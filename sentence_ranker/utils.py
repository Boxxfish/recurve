from argparse import ArgumentParser
import os
from pathlib import Path
from typing import Literal, TypeVar, get_args, get_origin
from transformers.modeling_outputs import CausalLMOutputWithPast # type: ignore
from transformers.models.llama import LlamaForCausalLM  # type: ignore

from pydantic import BaseModel
import torch
from torch import nn
import wandb

T = TypeVar("T", bound=BaseModel)

def parse_args(cfg_t: type[T]) -> T:
    parser = ArgumentParser()
    for k, v in cfg_t.model_fields.items():
        flag_name = f"--{k.replace('_', '-')}"
        if v.annotation == bool:
            parser.add_argument(
                flag_name, default=v.default, action="store_true"
            )
        elif get_origin(v.annotation) == Literal:
            choices = list(get_args(v.annotation))
            parser.add_argument(
                flag_name, default=v.default, choices=choices, type=type(choices[0])
            )
        else:
            assert v.annotation is not None
            parser.add_argument(
                flag_name, default=v.default, type=v.annotation
            )
    args = parser.parse_args()
    cfg = cfg_t(**args.__dict__)
    return cfg

def create_directory(out_dir_: str, meta: T) -> Path:
    for _ in range(100):
        if wandb.run.name not in ["" or None]:
            break
    if wandb.run.name not in ["" or None]:
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

def get_llm_logits(p_net: LlamaForCausalLM, input_ids: torch.Tensor, attn_masks: torch.Tensor) -> torch.Tensor:
    input_lens = attn_masks.byte().argmin(1) - 1
    model_out: CausalLMOutputWithPast = p_net.forward(input_ids, attn_masks, return_dict=True)
    return model_out.logits[torch.arange(0, model_out.logits.shape[0]), input_lens, :]

def copy_params(src: nn.Module, dest: nn.Module):
    """
    Copies params from one model to another.
    """
    with torch.no_grad():
        for dest_, src_ in zip(dest.parameters(), src.parameters()):
            dest_.data.copy_(src_.data)
