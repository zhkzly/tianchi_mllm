import wandb

wandb.init(
    project="mllm",
    dir="./logs",
    resume="allow",
)
from typing import Unpack

kwargs = {"text": "fs", "image": "fasf"}


def book(a: Unpack[kwargs]):
    print(a["text"])


text_fs = {"img": "fs", "sfaf": "fas"}
book(**text_fs)
