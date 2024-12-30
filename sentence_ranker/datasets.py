import random
from typing import *
from pydantic import BaseModel
import torch
from torch import Tensor
from transformers import PreTrainedTokenizer  # type: ignore


class DSItem(BaseModel):
    prompt: str
    answer: str


class Dataset(BaseModel):
    main_prompt: str  # Prompt prefix applied to all prompts in dataset
    items: List[DSItem]
    score_fn: str  # Python code that scores the model's output
    done_fn: str  # Python code that returns True when the model should stop generating

    def get_done_score_fns(self) -> Tuple[Callable[[Tensor], bool], Callable[[Tensor, str], float]]:
        locs: Dict[str, Any] = {}
        exec(compile(self.done_fn + "\n\n" + self.score_fn, "", "exec"), locs)
        done_fn, score_fn = locs["done_fn"], locs["score_fn"]
        return done_fn, score_fn

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
        num_envs: int,
        max_seq_len: int,
        done_fn: Callable[[Tensor], bool],
        score_fn: Callable[[Tensor, str], float],
    ):
        self.ds = ds
        self.tokenizer = tokenizer
        self.num_envs = num_envs
        self.max_seq_len = max_seq_len
        self.input_ids = torch.zeros([num_envs, max_seq_len], dtype=torch.int)
        self.attn_masks = torch.zeros([num_envs, max_seq_len], dtype=torch.bool)
        self.nexts = torch.zeros([num_envs], dtype=torch.int)
        self.done_fn = done_fn
        self.score_fn = score_fn
        self.ds_idxs = [0] * num_envs

    def step(
        self, actions: Tensor
    ) -> Tuple[Tuple[Tensor, Tensor], List[float], List[bool], List[bool], dict]:
        self.input_ids[torch.arange(0, self.num_envs), self.nexts] = actions.int()
        self.attn_masks[torch.arange(0, self.num_envs), self.nexts] = True
        self.nexts += 1
        texts = self.get_current_texts()
        dones = [
            self.done_fn(text)
            for text in texts
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
        truncs = [next == self.max_seq_len for next in self.nexts]
        for i, (done, trunc) in enumerate(zip(dones, truncs)):
            if done or trunc:
                self._reset_env(i)
        states = (self.input_ids.clone(), self.attn_masks.clone())
        return (states, rewards, dones, truncs, {})

    def reset(self) -> Tuple[Tensor, Tensor]:
        for i in range(self.num_envs):
            self._reset_env(i)
        self.ds_idxs = [
            random.randrange(0, len(self.ds.items)) for _ in range(self.num_envs)
        ]
        return self.input_ids, self.attn_masks
    
    def use_item(self, item_idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """Sets the dataset item to use for this episode."""
        self._reset_env(0, item_idx)
        return self.input_ids.clone(), self.attn_masks.clone()

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
        self.nexts[i] = 0  # Make sure this is updated when adding the prompt!

        # Update text with new dataset item
        if ds_idx is None:
            ds_idx = random.randrange(0, len(self.ds.items))
        self.ds_idxs[i] = ds_idx
        ds_item = self.ds.items[ds_idx]
        instr = self.ds.main_prompt + "\n" + ds_item.prompt
        inpt = self.tokenizer.apply_chat_template(
            [
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": instr}
            ],
            add_generation_prompt=True,
            return_tensors="pt",
            return_dict=True,
            chat_template=LLAMA_TEMPLATE,
        )
        self.input_ids[i, :len(inpt.input_ids[0])] = inpt.input_ids[0]
        self.attn_masks[i, :len(inpt.input_ids[0])] = True
        num_toks = inpt.input_ids.shape[1]
        self.nexts[i] = num_toks

if __name__ == "__main__":
    from yaml import load, Loader # type: ignore
    from transformers.models.llama import LlamaTokenizerFast  # type: ignore

    dataset_path = "datasets/best_lang.yaml"
    with open(dataset_path, "r") as f:
        data = load(f, Loader=Loader)
        dataset = Dataset.model_validate(data)
    done_fn, score_fn = dataset.get_done_score_fns()
    tokenizer = LlamaTokenizerFast.from_pretrained("meta-llama/Llama-3.2-1B-Instruct")
    envs = DSEnvs(dataset, tokenizer, 4, 256, done_fn, score_fn)
    torch.set_printoptions(threshold=10_000)

    envs.reset()
    envs.step(torch.tensor([0, 1, 4, 2]))
    envs.step(torch.tensor([1, 2, 3, 4]))
    envs.step(torch.tensor([2, 3, 4, 6]))
    print(envs.input_ids)
    print(envs.attn_masks)
    print(envs.nexts)