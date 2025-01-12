import random
from typing import *
from pydantic import BaseModel
import torch
from torch import Tensor
from transformers import PreTrainedTokenizer, AutoModelForCausalLM, DynamicCache
from torch.distributions import Categorical

from recurve.utils import get_llm_logits  # type: ignore


class DSItem(BaseModel):
    prompt: str
    answer: str


class Dataset(BaseModel):
    main_prompt: str  # Prompt prefix applied to all prompts in dataset
    items: List[DSItem]
    score_fn: str  # Python code that scores the model's output
    done_fn: str  # Python code that returns True when the model should stop generating
    split_fn: (
        str  # Python code that determines how the output should be split up for ranking
    )

    def get_done_score_fns(
        self,
    ) -> Tuple[Callable[[str], bool], Callable[[str, str], float]]:
        locs: Dict[str, Any] = {}
        exec(compile(self.done_fn + "\n\n" + self.score_fn, "", "exec"), locs)
        done_fn, score_fn = locs["done_fn"], locs["score_fn"]
        return done_fn, score_fn

    def get_split_fn(self) -> Callable[[str], bool]:
        locs: Dict[str, Any] = {}
        exec(compile(self.split_fn, "", "exec"), locs)
        split_fn = locs["split_fn"]
        return split_fn


LLAMA_TEMPLATE = """
{{ bos_token }}{% for message in messages -%}
{{ '<|start_header_id|>' + message['role'] + '<|end_header_id|>\n\n' + message['content'] | trim + '<|eot_id|>' }}
{%- endfor %}
{% if add_generation_prompt %}
{{ '<|start_header_id|>' + 'assistant' + '<|end_header_id|>\n\n' }}
{% endif %}
""".strip()


class DSEnvs:
    def __init__(
        self,
        ds: Dataset,
        tokenizer: PreTrainedTokenizer,
        max_seq_len: int,
        num_candidates: int,
        vocab_size: int,
    ):
        self.ds = ds
        self.tokenizer = tokenizer
        self.num_envs = 1
        self.max_seq_len = max_seq_len
        self.vocab_size = vocab_size
        self.input_ids = torch.zeros([self.num_envs, max_seq_len], dtype=torch.int)
        self.attn_masks = torch.zeros([self.num_envs, max_seq_len], dtype=torch.bool)
        self.candidate_input_ids = torch.zeros(
            [self.num_envs, num_candidates, max_seq_len], dtype=torch.int
        )
        self.candidate_attn_masks = torch.zeros(
            [self.num_envs, num_candidates, max_seq_len], dtype=torch.bool
        )
        self.candidate_logits = torch.zeros(
            [self.num_envs, num_candidates, max_seq_len, vocab_size], dtype=torch.float
        )
        self.candidate_masks = torch.zeros(
            [self.num_envs, num_candidates, max_seq_len], dtype=torch.bool
        )
        self.candidate_texts = [[None] * num_candidates for _ in range(self.num_envs)]
        self.num_candidates = num_candidates
        self.nexts = torch.zeros([self.num_envs], dtype=torch.int)
        self.done_fn, self.score_fn = ds.get_done_score_fns()
        self.split_fn = ds.get_split_fn()
        self.ds_idxs = [0] * self.num_envs

        # If using llama model, apply chat template
        if "llama" in str(tokenizer.name_or_path).lower():
            tokenizer.chat_template = LLAMA_TEMPLATE

    def step_ranker(
        self, actions: Tensor, lm: AutoModelForCausalLM, reset: bool = True
    ) -> Tuple[
        Tuple[Tensor, Tensor, Tensor, Tensor], List[float], List[bool], List[bool], dict
    ]:
        """Steps through the environment, for rankers."""
        self.input_ids = self.candidate_input_ids[
            torch.arange(0, self.num_envs), actions.int()
        ]
        self.attn_masks = self.candidate_attn_masks[
            torch.arange(0, self.num_envs), actions.int()
        ]
        self.nexts = self.attn_masks.int().argmin(1)
        texts = self.get_current_texts()
        dones = [
            self.done_fn(text) or next >= self.max_seq_len - 4
            for text, next in zip(texts, self.nexts)
        ]
        rewards = [
            (
                self.score_fn(
                    text,
                    self.ds.items[self.ds_idxs[i]].answer,
                )
                if done
                else 0.0
            )
            for i, (done, text) in enumerate(zip(dones, texts))
        ]
        truncs = [False] * self.num_envs
        for i, (done, trunc) in enumerate(zip(dones, truncs)):
            if (done or trunc) and reset:
                self._reset_env(i)
            self._gen_candidates(i, lm)
        states = (
            self.candidate_input_ids.clone(),
            self.candidate_attn_masks.clone(),
            self.candidate_logits.clone(),
            self.candidate_masks.clone(),
        )
        return (states, rewards, dones, truncs, {})

    def reset_ranker(
        self, lm: AutoModelForCausalLM
    ) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
        """Resets the environment, for rankers."""
        for i in range(self.num_envs):
            self._reset_env(i)
            self._gen_candidates(i, lm)
        return (
            self.candidate_input_ids.clone(),
            self.candidate_attn_masks.clone(),
            self.candidate_logits.clone(),
            self.candidate_masks.clone(),
        )

    def use_item_ranker(
        self, item_idx: int, lm: AutoModelForCausalLM
    ) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
        """Sets the dataset item to use for this episode, for rankers."""
        self._reset_env(0, item_idx)
        self._gen_candidates(0, lm)
        return (
            self.candidate_input_ids.clone(),
            self.candidate_attn_masks.clone(),
            self.candidate_logits.clone(),
            self.candidate_masks.clone(),
        )

    def get_current_texts(self) -> List[str]:
        """Returns the text in each environment."""
        return [
            self.tokenizer.decode(input_ids[:next])
            for input_ids, next in zip(self.input_ids, self.nexts)
        ]

    def _reset_env(self, i: int, ds_idx: Optional[int] = None):
        # Clear previous text
        self.input_ids[i, :] = self.tokenizer.pad_token_type_id
        self.attn_masks[i, :] = False
        self.nexts[i] = 0

        # Update text with new dataset item
        if ds_idx is None:
            ds_idx = random.randrange(0, len(self.ds.items))
        self.ds_idxs[i] = ds_idx
        ds_item = self.ds.items[ds_idx]
        instr = self.ds.main_prompt + "\n" + ds_item.prompt
        inpt = self.tokenizer.apply_chat_template(
            [
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": instr},
            ],
            add_generation_prompt=True,
            return_tensors="pt",
            return_dict=True,
        )
        num_toks = inpt.input_ids.shape[1]
        self.input_ids[i, :num_toks] = inpt.input_ids[0]
        self.attn_masks[i, :num_toks] = True
        self.nexts[i] = num_toks

    def _gen_candidates(self, i: int, lm: AutoModelForCausalLM):
        """Generates action candidates for the ranker."""
        device = lm.device
        for j in range(self.num_candidates):
            new_input_ids = self.input_ids[i].clone()
            new_attn_masks = self.attn_masks[i].clone()
            new_logits = torch.zeros([self.max_seq_len, self.vocab_size])
            new_candidate_mask = torch.zeros([self.max_seq_len], dtype=torch.bool)
            new_next = self.nexts[i].item()
            next_input_ids = new_input_ids
            next_attn_masks = new_attn_masks
            start_idx = new_next
            cache = DynamicCache()
            while True:
                logits, cache = get_llm_logits(
                    lm,
                    next_input_ids.unsqueeze(0).to(device),
                    next_attn_masks.unsqueeze(0).to(device),
                    cache,
                )
                logits = logits.cpu()
                input_id = Categorical(logits=logits).sample().squeeze().item()
                new_input_ids[new_next] = input_id
                new_attn_masks[new_next] = True
                new_logits[new_next] = logits.squeeze()
                new_candidate_mask[new_next] = True
                new_next += 1
                next_input_ids = new_input_ids[new_next - 1 :]
                next_attn_masks = new_attn_masks[new_next - 1 :]
                all_text = self.tokenizer.decode(new_input_ids[:new_next])
                text = self.tokenizer.decode(new_input_ids[start_idx:new_next])
                if (
                    self.split_fn(text)
                    or self.done_fn(all_text)
                    or new_next >= self.max_seq_len - 1
                ):
                    break
            self.candidate_texts[i][j] = text
            self.candidate_input_ids[i, j, :] = new_input_ids
            self.candidate_attn_masks[i, j, :] = new_attn_masks
            self.candidate_logits[i, j] = new_logits
            self.candidate_masks[i, j] = new_candidate_mask
