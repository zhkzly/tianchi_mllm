from dataclasses import dataclass,field
from typing import Optional, Union


@dataclass
class train_config:
    # model
    model_variant: str = "7b"
    ckpt_load_path: str = "/fsx/output/ckpt"
    ckpt_save_path: str = "/fsx/output/ckpt"

    # dataset and dataloader
    use_dummy_dataset: bool = False
    data_path: str = "/fsx/data"
    file_type: str = "arrow"
    col_name: str = "tokens"
    tokenizer_path: str = "/fsx/tokenizer"
    datasets: str = "lang=en/dataset=commoncrawl,lang=en/dataset=webhose,lang=en/dataset=github_clean,lang=de/dataset=wikipedia,lang=es/dataset=wikipedia,lang=fr/dataset=wikipedia,lang=ja/dataset=wikipedia,lang=pt/dataset=wikipedia,lang=en/dataset=wikimedia,lang=en/dataset=uspto,lang=en/dataset=pubmedcentral,lang=en/dataset=arxiv,lang=en/dataset=stackexchange"
    weights: str = "7725,500,550,28,17,22,25,8,100,500,175,250,100"
    seq_length: int = 4096
    vocab_size: int = 32000
    bos_token: Optional[int] = None
    eos_token: int = 0
    bol_token: Optional[int] = None
    eol_token: Optional[int] = None
    strip_tokens: str = ""
    logical_shards: int = 1024
    num_workers: int = 1

    # fsdp policies
    sharding_strategy: str = "hsdp"
    fsdp_activation_checkpointing: bool = False
    selective_checkpointing: Union[float, str] = 1  # percentage of blocks to apply ac
    mixed_precision: bool = True
    low_cpu_fsdp: bool = False

    # training spec
    batch_size: int = 2
    num_steps: int = 1000000
    training_stage: str = "initial"
    learning_rate: float = 3e-4
    grad_clip_thresh: float = 1.0
    seed: int = 2023

    # continued training spec
    resuming_dataset: bool = False

    # profiling
    use_profiler: bool = False
    profiler_rank0_only: bool = True

    # logging
    report_interval: int = 100
    checkpoint_interval: int = 10000
    tracker: Optional[str] = None  # None, "wandb", "aim"
    tracker_dir: str = "/fsx/aim_logs/llama"
    tracker_project_name: str = "llama"  # project name for a group of runs
    tracker_run_id: Optional[str] = None  # run id, for job resume purpose

    # compile
    use_torch_compile: bool = True

    # speculator training
    tp_size: int = 8
    model_arch: str = "embedllama"
    model_path: str = "/path/to/model/"
    n_speculator_heads: int = 3
    speculator_width: int = 4096
    speculator_tie_weights: bool = True
    speculator_scale_input: bool = True
    stage2_start_step: int = 15000
    stage2_prompt_length: int = 64
    stage2_batch_size: int = 96
    stage2_seq_length: int = 256




@dataclass
class TrainArgs:
    epochs: int = 10
    batch_size: int = 12
    lr: float = 1e-5
    optimizer: str = "adamw8bit"
    warm_up: bool = True
    lr_min: float = 1e-7
    warmup_steps: int = 100
    scheduler: str = "consine"
    max_steps: int = 1000

    log_freq_iter: int = 100
    save_freq_epoch: int = 2
    log_dir: str = "./log"

    model_save_path: str = "./huggingface/model"
    device: str = "cuda"
    shuffle: bool = True
    num_workers: int = 3

    seed: int = 32
    use_early_stop: bool = True
    early_stop_patience: int = 5
    early_stop_delta: float = 0.001

    # 显存优化层面
    use_model_compile: bool = True
    use_amp: bool = True
    use_accumlation: bool = True
    accumlation_steps: int = 1

    # lora
    use_lora: bool = True


@dataclass
class ModelArgs:
    model_name: str = "Qwen/Qwen2-VL-2B-Instruct"
    cache_dir: str = "huggingface"


@dataclass
class DataArgs:
    data_path: str = "datas/train"


@dataclass
class PeftArgs:
    low_rank: int = 8
    lora_alpha: float = 16
    target_modules: list[str] = field(default_factory=lambda: ["q_proj", "v_proj"])
    lora_dropout: float = 0.1
    bias: str = "none"
    task_type: str = "CAUSAL_LM"










