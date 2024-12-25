from argparse import ArgumentParser
import os
from pathlib import Path
from typing import TypeVar

from pydantic import BaseModel
import torch
from torch import nn
import wandb

T = TypeVar("T", bound=BaseModel)

def parse_args(cfg_t: type[T]) -> T:
    parser = ArgumentParser()
    for k, v in cfg_t.model_fields.items():
        if v.annotation == bool:
            parser.add_argument(
                f"--{k.replace('_', '-')}", default=v.default, action="store_true"
            )
        else:
            assert v.annotation is not None
            parser.add_argument(
                f"--{k.replace('_', '-')}", default=v.default, type=v.annotation
            )
    args = parser.parse_args()
    cfg = cfg_t(**args.__dict__)
    return cfg

def create_directory(out_dir_: str, meta: T) -> Path:
    assert wandb.run is not None
    for _ in range(100):
        if wandb.run.name != "":
            break
    if wandb.run.name != "":
        out_id = wandb.run.name
    else:
        out_id = "testing"

    out_dir = Path(out_dir_)
    exp_dir = out_dir / out_id
    try:
        os.mkdir(exp_dir)
    except OSError as e:
        print(e)
    with open(exp_dir / "meta.json", "w") as f:
        f.write(meta.model_dump_json())

    chkpt_path = exp_dir / "checkpoints"
    try:
        os.mkdir(chkpt_path)
    except OSError as e:
        print(e)
    return chkpt_path

def copy_params(src: nn.Module, dest: nn.Module):
    """
    Copies params from one model to another.
    """
    with torch.no_grad():
        for dest_, src_ in zip(dest.parameters(), src.parameters()):
            dest_.data.copy_(src_.data)
