from argparse import ArgumentParser
import os
from pathlib import Path
from typing import Literal, Optional, Tuple, TypeVar, Union, get_args, get_origin
from transformers.modeling_outputs import CausalLMOutputWithPast  # type: ignore
from transformers.models.llama import LlamaForCausalLM  # type: ignore
from transformers import Cache

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
            parser.add_argument(flag_name, default=v.default, action="store_true")
        elif get_origin(v.annotation) == Literal:
            choices = list(get_args(v.annotation))
            parser.add_argument(
                flag_name, default=v.default, choices=choices, type=type(choices[0])
            )
        else:
            assert v.annotation is not None
            parser.add_argument(flag_name, default=v.default, type=v.annotation)
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
        f.write(meta.model_dump_json(indent=2))

    chkpt_path = exp_dir / "checkpoints"
    try:
        os.mkdir(chkpt_path)
    except OSError as e:
        print(e)
    return chkpt_path


def get_llm_logits(
    p_net: LlamaForCausalLM,
    input_ids: torch.Tensor,
    attn_masks: torch.Tensor,
    cache: Optional[Cache],
) -> Tuple[torch.Tensor, Union[Cache, None]]:
    # Cap the length of the input using the longest sequence in the batch
    input_lens = attn_masks.byte().argmin(1) - 1
    max_len = input_lens.amax().item() + 1

    model_out: CausalLMOutputWithPast = p_net.forward(
        input_ids[:, :max_len],
        attn_masks[:, :max_len],
        return_dict=True,
        past_key_values=cache,
        use_cache=cache is not None,
    )
    return (
        model_out.logits[torch.arange(0, model_out.logits.shape[0]), input_lens, :],
        model_out.past_key_values,
    )


def get_llm_scores(
    q_net: LlamaForCausalLM, input_ids: torch.Tensor, attn_masks: torch.Tensor
) -> torch.Tensor:
    """Given `input_ids` and `attn_masks` of shape (batch_size, num_candidates, max_seq_len), returns a set of scores of shape (batch_size, num_candidates)."""
    batch_size, num_candidates, _ = input_ids.shape
    input_lens = attn_masks.flatten(0, 1).byte().argmin(1) - 1
    max_len = input_lens.amax().item() + 1
    q_vals = q_net.forward(
        input_ids.flatten(0, 1)[:, :max_len], attn_masks.flatten(0, 1)[:, :max_len]
    ).logits.reshape(batch_size, num_candidates)
    return q_vals


def copy_params(src: nn.Module, dest: nn.Module):
    """
    Copies params from one model to another.
    """
    with torch.no_grad():
        for dest_, src_ in zip(dest.parameters(), src.parameters()):
            dest_.data.copy_(src_.data)
