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


fmt = "%(asctime)s - %(name)s - %(levelname)s - %(message)s-%(funcName)s-%(lineno)s"
logging.basicConfig(format=fmt)
run_logger = logging.getLogger(__name__)


@dataclass
class DataArgs:
    data_file: str = "/media/zkl/zkl_T7/preprocession/observe_prediction_512"
    data_save_path: str = "../datas"


@dataclass
class TrainArgs:

    model_save_path: str = "./modified_model/checkpoints"
    save_epoch_freq: int = 1
    epochs: int = 30
    start_epoch: int = 0
    batch_size: int = 32
    lr: float = 1e-2
    lrs: List[float] = field(default_factory=list)

    loss_iter_log: int = 2
    plot_test_fig: bool = True

    seed: int = 123
    warm_up: bool = True
    warmup_steps: int = 20
    lr_min: float = 1e-8
    scheduler: str = "cosine"
    optimizer: str = "adamw"

    device: str = "cuda"
    ratio: float = 0.7
    resume: bool = True
    saving_fig_log_freq: int = 10

    using_eRank: bool = False
    eRank_epochs: int = 20
    eRank_lr: float = 1e-2
    momentum: float = 0.9


# 采用 typing 中的，必须添加 List[int]
@dataclass
class ModelArgs:
    e_layer: int = 3
    e_layers: List[int] = field(default_factory=list)
    pred_len: int = 512
    output_attention: bool = False
    enc_in: int = 1
    d_models: List[int] = field(default_factory=list)
    d_model: int = 1024
    embed: str = "fixed"
    freq: str = "h"
    dropout: float = 0.1
    d_ff: Optional[int] = None
    activation: str = "glue"
    exp_setting: int = 2
    c_out: int = 1
    n_heads: int = 5
    factor: int = 3


# @dataclass
# class ModelArgs():
#     e_layer: int = 3
#     e_layers: List[int] = field(default_factory=lambda: [])
#     pred_len: int = 512
#     output_attention: bool = False
#     enc_in: int = 1
#     d_models: List[int] = field(default_factory=lambda: [])
#     d_model: int = 1024
#     embed: str = 'fixed'
#     freq: str = 'h'
#     dropout: float = 0.1
#     d_ff: Optional[int] = None
#     activation: str = 'glue'
#     exp_setting: int = 2
#     c_out: int = 1
#     n_heads: int = 5
#     factor: int = 3


def get_model(trial):
    train_args = TrainArgs()
    data_args = DataArgs()
    model_args = ModelArgs()
    model_args.e_layer = trial.suggest_int("e_layer", 1, 5, step=2)
    model_args.d_model = trial.suggest_int("d_model", 128, 512, 1024)
    model_args.n_heads = trial.suggest_int("n_heads", 1, 6, step=2)
    model_args.dropout = trial.suggest_float("dropout", 0.1, 0.5, step=0.1)
    model = Model(configs=model_args).to(device=train_args.device)
    train_args.optimizer = trial.suggest_categorical(
        "optimizer", ["adam", "adamw", "sgd"]
    )
    train_args.max_steps = train_args.epochs
    optimizer = get_optimizer(model=model, train_args=train_args)
    lr_scheduler = get_scheduler(optimizer=optimizer, train_args=train_args)
    lr = trial.suggest_float("lr", 1e-5, 1e-1, log=True)

    return model, optimizer, lr_scheduler


def objective(trial):
    model, optimizer, lr_scheduler = get_model(trial)
    train_args = TrainArgs()
    data_args = DataArgs()

    epochs = train_args.epochs
    start_epoch = train_args.start_epoch
    batch_size = train_args.batch_size

    dtype = torch.float32
    ratio = train_args.ratio
    seed = train_args.seed
    device = torch.device(train_args.device)

    data_file = data_args.data_file

    if not os.path.exists(data_args.data_save_path):
        os.mkdir(data_args.data_save_path)
    data_save_file = data_args.data_save_path
    loss = nn.MSELoss()

    tempt = os.path.join(os.path.join(data_save_file), "mapping.json")
    if os.path.exists(tempt):
        index_mapping_file = tempt
        load_from_file = os.path.join(data_save_file, "train_val_test.npz")
    else:
        load_from_file = None
        index_mapping_file = None

    dataset = Data(
        file_path=data_file,
        loading_from_file=load_from_file,
        index_mapping_file=index_mapping_file,
        ratio=ratio,
        seed=seed,
    )
    # input,label,depth
    # batch_size=dataset.train_len

    train_set_loader, val_set_loader, test_set_loader = dataset.get_dataloaders(
        batch_size=batch_size, dtype=dtype
    )
    _, _, test_set_loader_for_save_fig = dataset.get_dataloaders(
        batch_size=1, dtype=dtype
    )
    print(f"length of val_set_dataloader:{len(val_set_loader)}")
    epoch = 0
    for epoch in tqdm(
        range(start_epoch, epochs),
        initial=start_epoch,
        total=epochs - start_epoch,
        desc=f"epoch_{epoch}_total_{epochs}",
    ):
        total_iter = 0
        k = 0
        end_cost = 0.0
        for x, y, depth in tqdm(
            train_set_loader,
            total=len(train_set_loader),
            desc=f"epoch_{epoch}_total_{epochs}",
        ):
            run_logger.debug(f"the shape of x:{x.shape}")
            run_logger.debug(f"the shape of y:{y.shape}")
            x = x.unsqueeze(dim=-1).to(device)
            y = y.to(device)
            pred = model(x)
            run_logger.debug(f"the shape of pred.squeeze:{pred.squeeze().shape}")
            L = loss(y, pred.squeeze()) * 512

            optimizer.zero_grad()

            L.backward()
            optimizer.step()
            total_iter += 1

        lr_scheduler.step(step=epoch)

        model.eval()
        l = 0.0
        for x, y, depth in tqdm(
            test_set_loader, total=len(test_set_loader), desc=f"test_model"
        ):
            run_logger.debug(f"the shape of x:{x.shape}")
            with torch.no_grad():
                x = x.unsqueeze(dim=-1).to(device)
                y = y.to(device)
                pred = model(x)
                run_logger.debug(f"pred shape:{pred.shape}")
                L = loss(y, pred.squeeze()) * 512
                l = l + L.item() * x.shape[0]
        accuracy = l / len(test_set_loader.dataset)
        trial.report(accuracy, epoch)
        # Handle pruning based on the intermediate value.
        if trial.should_prune():
            raise optuna.exceptions.TrialPruned()
    return accuracy




class HyperSearch():
    def __init__(self, model_args, train_args, data_args):
        self.model_args = model_args
        self.train_args = train_args
        self.data_args = data_args

    def run(self):
        pass
    


    












if __name__ == "__main__":
    import optuna
    from optuna.storages import JournalStorage
    from optuna.trial import TrialState
    from optuna.storages.journal import JournalFileBackend
    from optuna.samplers import TPESampler

    sampler = TPESampler(seed=10)  # Make the sampler behave in a deterministic way.
    run_logger.setLevel(logging.INFO)
    # storage = JournalStorage(JournalFileBackend("./optuna_journal_storage.log"))
    storage = "sqlite:///./optuna.db"
    # storage = JournalStorage(JournalFileBackend("./optuna_journal_storage.log"))
    study = optuna.create_study(direction="minimize", storage=storage, sampler=sampler)
    study.optimize(objective, n_trials=300)

    pruned_trials = study.get_trials(deepcopy=False, states=[TrialState.PRUNED])
    complete_trials = study.get_trials(deepcopy=False, states=[TrialState.COMPLETE])

    print("Study statistics: ")
    print("  Number of finished trials: ", len(study.trials))
    print("  Number of pruned trials: ", len(pruned_trials))
    print("  Number of complete trials: ", len(complete_trials))

    print("Best trial:")
    trial = study.best_trial

    print("  Value: ", trial.value)

    print("  Params: ")
    for key, value in trial.params.items():
        print("    {}: {}".format(key, value))
