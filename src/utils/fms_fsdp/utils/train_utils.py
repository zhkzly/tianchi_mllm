import os
from dataclasses import asdict
from functools import partial


try:
    import packaging.version
except ImportError:
    from pkg_resources import packaging  # type: ignore

import time
from datetime import timedelta

import torch.cuda.nccl as nccl
import torch.distributed as dist
from torch.distributed.fsdp import ShardingStrategy

from fms_fsdp.policies import *


def train(
    train_args,
    model,
    local_rank,
    rank,
    train_loader,
    val_loader,
    optimizer,
    scheduler,
    profiler,
    checkpointer,
    start_step,
    tokens_seen,
    train_sampler,
    epoch,
    tracker_fn,
):
    train_sampler.set_epoch(epoch)

    model.train()
    ddp_stats = torch.zeros(3).to(local_rank)

    start = time.time()
    loop_start = time.time()
    train_loss = -1
    # 指定了一个开始的节点
    for batch_idx, (input, label) in enumerate(train_loader, start=start_step + 1):
        if batch_idx > train_args.num_steps:
            break
        input = input.to(local_rank)
        label = label.to(local_rank)

        optimizer.zero_grad()
        output = model(input)
        output = output.logits if hasattr(output, "logits") else output
        ce_loss = torch.nn.CrossEntropyLoss()
        # 局部的损失，全局的梯度
        loss = ce_loss(output.view(-1, output.size(-1)), label.view(-1).long())

        loss.backward()
        ddp_stats[1] += model.clip_grad_norm_(train_args.grad_clip_thresh).item()
        optimizer.step()
        scheduler.step()

        ddp_stats[0] += loss.item()
        ddp_stats[2] += 1

        if profiler:
            profiler.step()

        if batch_idx % train_args.report_interval == 0:
            # 等价与 all_gather+ reduceOp.SUM,all_gather,是得到所有的结果，
            dist.all_reduce(ddp_stats, op=dist.ReduceOp.SUM)
            train_loss = ddp_stats[0] / ddp_stats[2]
            g_norm = ddp_stats[1] / ddp_stats[2]
            elapsed_time = time.time() - loop_start
            world_size = int(os.environ["WORLD_SIZE"])
            new_tokens_seen = (
                (batch_idx - start_step)
                * world_size
                * train_args.batch_size
                * train_args.seq_length
            )
            if rank == 0:
                total_tokens_seen = tokens_seen + new_tokens_seen
                current_loss = train_loss.item()
                current_lr = scheduler.get_last_lr()[0]
                current_gnorm = g_norm.item()
                current_step_time = (time.time() - start) / train_args.report_interval
                overall_step_time = elapsed_time / (batch_idx - start_step)
                current_throughput = int(
                    train_args.batch_size * train_args.seq_length / current_step_time
                )
                overall_throughput = int(
                    train_args.batch_size * train_args.seq_length / overall_step_time
                )
                reserved_mem = torch.cuda.max_memory_reserved(
                    device=torch.cuda.current_device()
                )
                allocated_mem = torch.cuda.max_memory_allocated(
                    device=torch.cuda.current_device()
                )

                print("step:", batch_idx)
                print("loss:", current_loss)
                print("LR:", current_lr)
                print("tokens seen:", total_tokens_seen)
                print("gradient norm:", current_gnorm)
                print("reserved memory:", reserved_mem)
                print("allocated memory:", allocated_mem)
                print("current step time:", current_step_time)
                print("overall step time:", overall_step_time)
                print("current token per gpu per sec:", current_throughput)
                print("overall token per gpu per sec:", overall_throughput)
                print(
                    "overall token per day:",
                    int(new_tokens_seen / elapsed_time * 3600 * 24),
                )
                if train_args.tracker:
                    vals_to_track = {
                        "learning rate": current_lr,
                        "loss": current_loss,
                        "gradient norm": current_gnorm,
                        "token seen": total_tokens_seen,
                        "current throughput (token per gpu per sec)": current_throughput,
                        "overall throughput (token per gpu per sec)": overall_throughput,
                        "gpu reserved memory": reserved_mem,
                        "gpu allocated memory": allocated_mem,
                    }
                    tracker_fn(vals_to_track, step=batch_idx)

            start = time.time()
            ddp_stats.zero_()
        torch.cuda.reset_peak_memory_stats(device=torch.cuda.current_device())

        if batch_idx % train_args.checkpoint_interval == 0:
            checkpointer.save(
                batch_idx,
                model,
                optimizer,
                None,
                tokens_seen=tokens_seen + new_tokens_seen,
            )

    return train_loss


def validate_fn(model, val_loader, epoch, train_args, rank, local_rank, tracker_fn):
    if rank == 0:
        print(f"Validating at epoch {epoch}...")
    with torch.no_grad():
        model.eval()
        val_stats = torch.zeros(2).to(local_rank)
        for val_batch_idx, (val_input, val_label) in enumerate(val_loader):
            val_input = val_input.to(local_rank)
            val_label = val_label.to(local_rank)
            val_output = model(val_input)
            val_output = (
                val_output.logits if hasattr(val_output, "logits") else val_output
            )
            val_loss = torch.nn.CrossEntropyLoss()(
                val_output.view(-1, val_output.size(-1)),
                val_label.view(-1).long(),
            )
            val_stats[0] += val_loss.item()
            val_stats[1] += val_input.shape[0]
        dist.reduce(val_stats, dst=0, op=dist.ReduceOp.SUM)
        if rank == 0:
            mean_val_loss = val_stats[0] / val_stats[1]
            print(f"Validation loss at epoch {epoch}: {val_loss.item()}")
            if train_args.tracker:
                if train_args.tracker == "wandb":
                    import wandb  # type: ignore

                    wandb.log({"validation loss": val_loss.item()}, step=epoch)
                elif train_args.tracker == "aim":
                    tracker_fn(
                        {"mean_validation loss": mean_val_loss.item()},
                        name="validation mean loss",
                        epoch=epoch,
                    )
        return mean_val_loss.item()


def setup():
    dist.init_process_group("nccl", timeout=timedelta(seconds=60 * 60))


def setup_environ_flags():
    os.environ["TORCH_SHOW_CPP_STACKTRACES"] = str(1)
    os.environ["NCCL_ASYNC_ERROR_HANDLING"] = str(1)


def get_mixed_precision_policy(train_args, rank):
    verify_bfloat_support = (
        torch.version.cuda
        and torch.cuda.is_bf16_supported()
        and packaging.version.parse(torch.version.cuda).release >= (11, 0)
        and dist.is_nccl_available()
        and nccl.version() >= (2, 10)
    )

    if train_args.mixed_precision:
        bf16_ready = verify_bfloat_support
        if bf16_ready:
            mixed_precision_policy = bfSixteen
            if rank == 0:
                print(f"bFloat16 enabled for mixed precision - using bfSixteen policy")
        else:
            mixed_precision_policy = fpSixteen
            if rank == 0:
                print(f"FP16 enabled")
    else:
        mixed_precision_policy = None

    return mixed_precision_policy


def get_policies(train_args, rank, block=None):
    """Get policies for mixed precision, wrapping, sharding, ac and param init function."""

    # mixed precision
    mixed_precision_policy = get_mixed_precision_policy(train_args, rank)

    # wrapping policy
    wrapping_policy = get_wrapper(train_args, block)

    # sharding strategy
    if train_args.sharding_strategy == "fsdp":
        sharding_strategy = ShardingStrategy.FULL_SHARD
    elif train_args.sharding_strategy == "hsdp":
        sharding_strategy = ShardingStrategy.HYBRID_SHARD
    elif train_args.sharding_strategy == "ddp":
        sharding_strategy = ShardingStrategy.NO_SHARD
    else:
        sharding_strategy = ShardingStrategy.FULL_SHARD
    if rank == 0:
        print(f"Sharding strategy = {train_args.sharding_strategy}")

    # ac handler
    apply_selective_ac = partial(apply_fsdp_checkpointing, block=block)

    # param init function
    if train_args.low_cpu_fsdp:
        param_init_fn = param_init_function
    else:
        param_init_fn = None

    return (
        mixed_precision_policy,
        wrapping_policy,
        sharding_strategy,
        apply_selective_ac,
        param_init_fn,
    )


def get_profiler(train_args, rank):
    if not train_args.use_profiler:
        return
    if train_args.profiler_rank0_only and rank != 0:
        return
    return torch.profiler.profile(
        activities=[
            torch.profiler.ProfilerActivity.CPU,
            torch.profiler.ProfilerActivity.CUDA,
        ],
        schedule=torch.profiler.schedule(wait=1, warmup=2, active=3, repeat=1),
        on_trace_ready=torch.profiler.tensorboard_trace_handler("profile_traces"),
        profile_memory=True,
        with_stack=False,
        record_shapes=True,
    )


def get_wrap_block(train_args, rank=0):
    set_block = set(train_args.set_block)
    if rank == 0:
        print(f"set_block = {set_block}")
    return set_block


def get_tracker(train_args, rank=0):
    if rank == 0:
        print(f"--> tracker is enabling!")
        if train_args.tracker:
            if train_args.tracker not in ["wandb", "aim"]:
                raise ValueError(f"tracker {train_args.tracker} not supported.")
            tracker_dir = train_args.tracker_dir
            project_name = train_args.tracker_project_name
            run_id = train_args.tracker_run_id

            if train_args.tracker == "wandb":
                try:
                    import wandb  # type: ignore
                except ImportError:
                    raise ImportError(
                        "tracker is set to wandb but wandb is not installed."
                    )

                print(f"--> wandb is enabled!")
                try:
                    wandb.init(
                        project=project_name,
                        dir=tracker_dir,
                        resume="allow",
                        id=run_id,
                    )
                except wandb.errors.UsageError:
                    raise ValueError(
                        "wandb failed to init, did you pass your wandb api key via WANDB_API_KEY?"
                    )
                wandb.config = asdict(train_args)
                tracker_fn = wandb.log

            if train_args.tracker == "aim":
                try:
                    from aim import Run  # type: ignore
                except ImportError:
                    raise ImportError("tracker is set to aim but aim is not installed.")

                print(f"--> aim is enabled!")
                run = Run(
                    experiment=project_name,
                    repo=tracker_dir,
                    run_hash=run_id,
                )
                run["hparams"] = asdict(train_args)
                tracker_fn = run.track

    else:
        tracker_fn = lambda *args, **kwargs: None

    return tracker_fn
