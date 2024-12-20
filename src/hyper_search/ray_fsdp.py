import torch
import torch.distributed as dist
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from ray import tune
import ray
from src.hyper_search.run_optuna import optuna_main

# 定义训练函数


class RaySearch:
    def __init__(self, train_args, data_args, model_args, peft_args):
        self.train_args = train_args
        self.data_args = data_args
        self.model_args = model_args
        self.peft_args = peft_args

    def train_fn(self, ray_config):
        dist.init_process_group(backend="nccl")
        train_args.lr = ray_config["lr"]
        train_args.epochs = ray_config["epochs"]
        train_args.optimizer = ray_config["optimizer"]
        train_args.batch_size = ray_config["batch_size"]
        train_args.lr_scheduler = ray_config["lr_scheduler"]

        if self.rank == 0:
            print(f"config hyper_params")

        val_loss = optuna_main(train_args, data_args, model_args, peft_args)

        ray.train.report(metric={"val_loss": val_loss})


if __name__ == "__main__":
    # 使用 Ray Tune 运行分布式超参数搜索

    ray_search = RaySearch(train_args, data_args, model_args, peft_args)
    train_fn = ray_search.train_fn
    ayalysis = tune.run(
        train_fn,
        config={
            "lr": tune.loguniform(1e-5, 1e-3),  # 学习率范围
            "epochs": tune.randint(1, 10),  # 训练 epoch 范围
            "optimizer": tune.choice(["adam", "adamw"]),  # 优化器选择
            "batch_size": tune.choice([32, 64, 128]),  # 批大小选择
            "lr_scheduler": tune.choice(["cosine", "cosine_warmup"]),  # 学习率衰减策略
        },
        resources_per_trial={"cpu": 2, "gpu": 3},  # 每个 Trial 需要的资源
        name="tune_fsdp",  # 运行名称,
        num_samples=10,  # 超参数搜索的采样数
        local_dir=ray_args.log_dir,  # 结果保存路径
    )
    analysis = tune.run(...)
    print(
        "Best hyperparameters:",
        analysis.get_best_config(metric="mean_loss", mode="min"),
    )
