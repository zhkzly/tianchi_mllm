import math
import os

import torch
import torch.optim as optim

# from fms.models.llama import LLaMA, LLaMABlock
from torch import distributed as dist
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.optim.lr_scheduler import LambdaLR
from rich import print
from src.utils.fms_fsdp.utils.checkpointing_utils import Checkpointer
from src.utils.fms_fsdp.utils.config_utils import get_model_config
from src.utils.fms_fsdp.utils.dataloader_utils import get_data_loader, get_dummy_loader
from src.utils.fms_fsdp.utils.train_utils import (
    get_policies,
    get_profiler,
    setup,
    setup_environ_flags,
    train,
    get_wrap_block,
    validate_fn,
)
from src.utils.utils import get_model
from src.utils.fms_fsdp.configs.configs import ModelArgs, TrainArgs, PeftArgs, DataArgs
from transformers import HfArgumentParser
from peft import LoraConfig, get_peft_model

from src.utils.fms_fsdp.utils.dummy_dataloader import get_dataloaders
from src.utils.fms_fsdp.utils.dummy_datasets import CustomSftDataset, DataSftCollator

from transformers import AutoProcessor
from src.utils.utils import get_optimizer, get_scheduler
from tqdm import tqdm


def optuna_main(
    train_args: TrainArgs,
    model_args: ModelArgs,
    data_args: DataArgs,
    peft_args: PeftArgs,
):
    # get configs

    # ensure reproducibility
    torch.cuda.manual_seed(train_args.seed)
    torch.manual_seed(train_args.seed)

    # torchrun specific
    local_rank = int(os.environ["LOCAL_RANK"])
    rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])

    if rank == 0:
        print(f"--> running with these configs train_args: {train_args}\n")
        print(f"--> running with these configs model_args: {model_args}\n")
        print(f"--> running with these configs data_args: {data_args}\n")
        print(f"--> running with these configs peft_args: {peft_args}\n")

    # some setups
    setup()
    torch.cuda.set_device(local_rank)
    torch.cuda.empty_cache()
    setup_environ_flags()

    # get policy
    block = get_wrap_block(train_args, rank)
    (
        mixed_precision_policy,
        wrapping_policy,
        sharding_strategy_policy,
        apply_selective_ac,
        param_init_fn,
    ) = get_policies(train_args, rank, block)

    # get fms model
    llama_config = get_model_config(model_args.model_name)
    if train_args.low_cpu_fsdp and train_args.use_lora:
        if rank == 0:
            print(f"--> using Lora for low cpu fsdp and low rank...")
        with torch.device("meta"):
            model = get_model(model_args.model_name)
            lora_config = LoraConfig(
                r=peft_args.low_rank,
                target_modules=peft_args.target_modules,
                peft_type=peft_args.task_type,
                lora_alpha=peft_args.lora_alpha,
                bias=peft_args.bias,
            )
        model = get_peft_model(
            model=model, peft_config=lora_config, adapter_name="train_qwen2"
        )
    else:
        model = LLaMA(llama_config)
        model.reset_parameters()

    if rank == 0:
        total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"\n--> model has {total_params / 1e6} Million params\n")

    # get data loader
    if rank == 0:
        print("Constructing datasets...")
        print(f"--> using data_path: {data_args.data_path}")
        print(f"loading qwen2 processor...")
        print("Datasets constructed!")
    preprocessor = AutoProcessor.from_pretrained("Qwen/Qwen2-VL-2B-Instruct")
    train_dataset = CustomSftDataset(
        preprocessor=preprocessor,
        data_path=data_args.data_path,
        data_type="train",
        task_type="all",
    )
    val_dataset = CustomSftDataset(
        preprocessor=preprocessor,
        data_path=data_args.data_path,
        data_type="val",
        task_type="all",
    )
    data_collator = DataSftCollator(preprocessor=preprocessor)
    train_sampler, train_loader, val_loader = get_dataloaders(
        train_dataset,
        val_dataset,
        world_size=world_size,
        local_rank=rank,
        shuffle=train_args.shuffle,
        seed=train_args.seed,
        collator=data_collator,
        batch_size=train_args.batch_size,
        num_workers=train_args.num_workers,
    )

    if rank == 0:
        print("Datasets constructed!")
        print("successfully constructed data loaders!")

    # FSDP
    model = FSDP(
        model,
        auto_wrap_policy=wrapping_policy,
        mixed_precision=mixed_precision_policy,
        sharding_strategy=sharding_strategy_policy,
        use_orig_params=train_args.use_torch_compile,
        device_id=torch.cuda.current_device(),
        limit_all_gathers=True,
        # param_init_fn=param_init_fn,
    )
    # we need this post-fsdp call to avoid graph break with torch.compile, until we figure out a better solution.
    # model.rot_emb.compute_freqs_cis(
    #     torch.device("cuda", torch.cuda.current_device()),
    #     model.config.max_expected_seq_len,
    # )

    # fsdp activation checkpointing
    if train_args.fsdp_activation_checkpointing:
        if rank == 0:
            print(f"--> applying FSDP activation checkpointing...")
        apply_selective_ac(model, p=train_args.selective_checkpointing)

    # torch compile
    if train_args.use_torch_compile:
        if rank == 0:
            print(f"--> enabling torch compile...")
        # the default accumulated_cache_size_limit=64 is not enough for 70b model, so we make it 128 here
        torch._dynamo.config.accumulated_cache_size_limit = 128
        model = torch.compile(model)

    # Optimizer
    # optimizer = optim.AdamW(
    #     model.parameters(),
    #     lr=train_args.learning_rate,
    #     betas=(0.9, 0.95),
    #     weight_decay=0.1,
    # )
    optimizer = get_optimizer(model, train_args)

    # optionally load from checkpoint (when continue pretraining)
    checkpointer = Checkpointer(
        train_args.ckpt_save_path, 1000, train_args.sharding_strategy, rank, local_rank
    )
    model, optimizer, _, start_step, tokens_seen, is_resuming = checkpointer.load(
        model,
        optimizer,
        None,
        path=(
            os.path.join(train_args.ckpt_load_path, "checkpoints/")
            if not os.path.isfile(train_args.ckpt_load_path)
            else train_args.ckpt_load_path
        ),
        strict=False,
    )
    if not is_resuming:
        start_step = 0
        # Override loaded optim hyperparams with the current values
        for g in optimizer.param_groups:
            g["initial_lr"] = train_args.learning_rate

    # LR schedule
    if train_args.training_stage == "annealing":
        schedule = lambda x: 1 - x / train_args.num_steps
    else:
        warmup_interval = min(2000, train_args.num_steps // 20)
        schedule = lambda x: min(
            1 - (1 - min(x, warmup_interval) / warmup_interval) ** 2,
            0.1
            + 0.5
            * (1 - 0.1)
            * (
                1
                + math.cos(
                    min(x, train_args.num_steps) / train_args.num_steps * math.pi
                )
            ),
        )
    scheduler = LambdaLR(optimizer, lambda x: schedule(x + start_step))

    # profiler
    profiler = get_profiler(train_args, rank)

    # Train
    if rank == 0:
        print(f"Training for {train_args.num_steps} steps...")
    # if rank==0:
    #     pbar=tqdm(range(train_args.epochs),total=train_args.epochs,desc="trianing epochs----")
    # else:
    #     pbar=range(train_args.epochs)
    pbar = tqdm(
        range(train_args.epochs),
        total=total_steps_per_epoch,
        colour="blue",
        desc=f"traing epochs",
        disable=(rank != 0),
    )
    for epoch in pbar:
        train(
            train_args,
            model,
            local_rank,
            rank,
            train_loader,
            optimizer,
            scheduler,
            profiler,
            checkpointer,
            start_step,
            tokens_seen,
            train_sampler=train_sampler,
        )

    checkpointer.save_single_file(train_args.num_steps, model)
    dist.barrier()
    validate_loss = validate_fn(
        model,
        val_loader=val_loader,
        local_rank=local_rank,
        rank=rank,
        tracker_fn=tracker_fn,
    )
    return validate_loss


if __name__ == "__main__":
    # fire.Fire(main)
    train_args, data_args, peft_args, data_args = HfArgumentParser(
        (ModelArgs, TrainArgs, PeftArgs, DataArgs)
    ).parse_args_into_dataclasses()

    main(
        train_args=train_args,
        data_args=data_args,
        peft_args=peft_args,
        data_args=data_args,
    )
