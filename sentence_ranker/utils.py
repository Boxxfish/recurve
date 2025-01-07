from argparse import ArgumentParser
import os
from pathlib import Path
from typing import Literal, Optional, Tuple, TypeVar, Union, get_args, get_origin
from transformers.modeling_outputs import CausalLMOutputWithPast  # type: ignore
from transformers.models.llama import LlamaForCausalLM, LlamaForSequenceClassification  # type: ignore
from transformers import Cache, DynamicCache

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
    """Given `input_ids` and `attn_masks` of shape (batch_size, max_seq_len), returns a set of logits of shape (batch_size, vocab_size)."""
    # Cap the length of the input using the longest sequence in the batch
    input_lens = attn_masks.byte().argmin(1) - 1
    max_len = input_lens.amax().item() + 1

    model_out: CausalLMOutputWithPast = p_net(
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


def get_llm_logits_candidates(
    p_net: LlamaForCausalLM,
    input_ids: torch.Tensor,
    attn_masks: torch.Tensor,
) -> torch.Tensor:
    """
    Given `input_ids` and `attn_masks` of shape (batch_size, max_seq_len), returns a set of logits of shape
    (batch_size, max_seq_len, vocab_size).
    This is intended for computing logits for a set of completions with a common prefix.
    For memory reasons, the returned logits will always be on the CPU.
    """
    # Compute cached states for candidate prefixes
    batch_size, max_seq_len = input_ids.shape
    print(batch_size, max_seq_len)
    assert batch_size > 1
    prefix_idx = (input_ids[0] == input_ids[1]).byte().argmin().item() - 1
    cache = DynamicCache()
    lm_output = p_net(
        input_ids[:1, :prefix_idx],
        attn_masks[:1, :prefix_idx],
        past_key_values=cache,
        use_cache=True,
    )
    prefix_logits = torch.cat(
        [lm_output.logits.cpu()] * batch_size, dim=0
    )  # Shape: (batch_size, prefix_len, vocab_size)
    cache: DynamicCache = lm_output.past_key_values

    # Repeat cache for each candidate
    for i in range(len(cache.key_cache)):
        cache.key_cache[i] = torch.cat([cache.key_cache[i]] * batch_size, dim=0)
        cache.value_cache[i] = torch.cat([cache.value_cache[i]] * batch_size, dim=0)

    # Run inference over all candidates
    input_lens = attn_masks.byte().argmin(1) - 1
    max_len = input_lens.amax().item() + 1
    main_logits = p_net(
        input_ids[:, prefix_idx:max_len],
        attn_masks[:, :max_len],
        past_key_values=cache,
    ).logits.cpu()  # Shape: (batch_size, max_len - prefix_len, vocab_size)

    # Concat prefix, main, and padding logits
    vocab_size = main_logits.shape[2]
    padding = torch.zeros(
        [batch_size, max_seq_len - max_len, vocab_size],
        dtype=torch.float,
    )
    logits = torch.cat(
        [prefix_logits, main_logits, padding], 1
    )  # Shape: (batch_size, max_seq_len, vocab_size)
    return logits


def get_llm_scores(
    q_net: LlamaForSequenceClassification,
    input_ids: torch.Tensor,
    attn_masks: torch.Tensor,
    use_sigmoid: bool,
) -> torch.Tensor:
    """Given `input_ids` and `attn_masks` of shape (batch_size, num_candidates, max_seq_len), returns a set of scores of shape (batch_size, num_candidates)."""
    _, num_candidates, _ = input_ids.shape
    assert num_candidates > 1, "Must supply multiple candidates per batch row"
    input_lens = attn_masks.byte().argmin(-1) - 1

    q_vals = []
    for single_input_ids, single_attn_masks, single_input_lens in zip(
        input_ids, attn_masks, input_lens
    ):
        # Compute cached states for candidate prefixes
        prefix_idx = (
            single_input_ids[0] == single_input_ids[1]
        ).byte().argmin().item() - 1
        cache = DynamicCache()
        lm_output = q_net(
            single_input_ids[:1, :prefix_idx],
            single_attn_masks[:1, :prefix_idx],
            past_key_values=cache,
            use_cache=True,
        )
        cache: DynamicCache = lm_output.past_key_values

        # Repeat cache for each candidate
        for i in range(len(cache.key_cache)):
            cache.key_cache[i] = torch.cat([cache.key_cache[i]] * num_candidates, dim=0)
            cache.value_cache[i] = torch.cat(
                [cache.value_cache[i]] * num_candidates, dim=0
            )

        # Run inference over all candidates
        max_len = single_input_lens.amax().item() + 1
        single_q_vals = q_net(
            single_input_ids[:, prefix_idx:max_len],
            single_attn_masks[:, :max_len],
            past_key_values=cache,
        ).logits.squeeze(
            -1
        )  # Shape: (num_candidates)
        q_vals.append(single_q_vals)
    q_vals = torch.stack(q_vals, 0)
    if use_sigmoid:
        q_vals = q_vals.sigmoid()
    return q_vals


def copy_params(src: nn.Module, dest: nn.Module):
    """
    Copies params from one model to another.
    """
    with torch.no_grad():
        for dest_, src_ in zip(dest.parameters(), src.parameters()):
            dest_.data.copy_(src_.data)
