# user zhengkelong

from modified_model.models import Model
from modified_model.utils.data import MyDataset, Data

from torch.utils.data import DataLoader
import numpy as np
import torch
import os
import torch.optim as optim
import time
import torch.nn as nn
import matplotlib as mpl
import matplotlib.pyplot as plt
import logging
from tqdm import tqdm
from dataclasses import dataclass, field
from transformers import HfArgumentParser
from typing import Optional, List
from torch.utils.tensorboard import SummaryWriter
import math
from modified_model.utils.helper import get_optimizer, get_scheduler

from safetensors import safe_open
from safetensors.torch import save_file
from src.hyper_search.run_optuna import optuna_main
import torch.distributed as dist


fmt = "%(asctime)s - %(name)s - %(levelname)s - %(message)s-%(funcName)s-%(lineno)s"
logging.basicConfig(format=fmt)
run_logger = logging.getLogger(__name__)


# 实现多卡的超参数搜索，并使用 optuna 进行超参数搜索，
# 主要方法，是仅在主进程中创建optuna.create_study(),得到主要的参数后，通过dist的广播机制，将参数广播到各个进程中，
# 之后每个进程通过广播的参数进行模型的配置，训练的配置，之后进行训练，并记录相应的指标，最后进行模型的保存。


import optuna
from optuna.samplers import TPESampler
from optuna.integration import TorchDistributedTrial
from optuna.trial import TrialState


class OptunaFSDP:
    def __init__(self, model_args, train_args, data_args, peft_args, optuna_args, rank):
        self.model_args = model_args
        self.train_args = train_args
        self.data_args = data_args
        self.peft_args = peft_args
        self.optuna_args = optuna_args
        self.rank = rank
        self._set_up_optuna()

    def _set_up_optuna(self):
        self.optuna_group = dist.new_group(backend="gloo")

    def objective(self, trial):
        # 训练相关的，学习率，drop_out,epochs,optimizer,
        trial = TorchDistributedTrial(trial, group=self.optuna_group)

        train_args.lr = trial.suggest_float("lr", 1e-5, 1e-1, log=True)
        train_args.epochs = trial.suggest_int("epochs", 1, 50, step=10)
        train_args.optimizer = trial.suggest_categorical(
            "optimizer", ["adam", "adamw", "sgd"]
        )

        train_args.lr_scheduler = trial.suggest_categorical(
            "lr_scheduler", ["cosine", "step"]
        )
        if self.rank == 0:
            print(f"config hyper_params")

        val_loss = optuna_main(train_args, data_args, model_args, peft_args)

        trial.report(val_loss, step=epoch)
        if trial.should_prune():
            raise optuna.exceptions.TrialPruned()
        return val_loss

    def search(self):
        if self.rank == 0:
            storge = self.optuna_args.storage
            sampler = TPESampler(seed=self.optuna_args.seed)
            self.study = optuna.create_study(
                direction="minimize", sampler=sampler, storage=storge
            )
            self.study.optimize(self.objective, n_trials=self.optuna_args.n_trials)
            prued = self.study.get_trials(deepcopy=False, states=[TrialState.PRUNED])
            complete = self.study.get_trials(
                deepcopy=False, states=[TrialState.COMPLETE]
            )
            print("Study statistics: ")
            print("  Number of finished trials: ", len(self.study.trials))
            print("  Number of pruned trials: ", len(pruned))
            print("  Number of complete trials: ", len(complete))
            print("Best trial:")
            trial = self.study.best_trial
            print(f"the value of the best trial: {trial.value}")
            print("  Params: ")
            for key, value in trial.params.items():
                print(f"    {key}: {value}")
            dist.destroy_process_group(self.optuna_group)

        else:
            for _ in range(self.optuna_args.n_trials):
                try:
                    self.objective(None)
                except optuna.exceptions.TrialPruned:
                    pass
        dist.barrier()
        dist.barrier(self.optuna_group)
        dist.destroy_process_group(self.optuna_group)
        dist.destroy_process_group(self.optuna_group)


if __name__ == "__main__":
    dist.init_process_group()
    optuna_fsdp = OptunaFSDP(
        model_args, train_args, data_args, peft_args, optuna_args, rank
    )
    optuna_fsdp.search()
