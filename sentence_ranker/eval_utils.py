from typing import *
from pydantic import BaseModel
import torch
from torch import nn
from torch.distributions import Categorical
from tqdm import tqdm
from transformers import AutoTokenizer

from sentence_ranker.datasets import DSEnvs, Dataset
from sentence_ranker.utils import get_llm_logits

class EvalResultOutput(BaseModel):
    text: str
    score: float

class EvalResult(BaseModel):
    item_idx: int
    outputs: List[EvalResultOutput]
    avg_score: float

class EvalResults(BaseModel):
    args: Any
    results: List[EvalResult]
    avg_score: float

def run_eval(args: BaseModel, dataset: Dataset, tokenizer: AutoTokenizer, max_seq_len: int, runs_per_item: int, p_net: nn.Module, device: torch.device, ranker_candidates: int) -> EvalResults:
    envs = DSEnvs(dataset, tokenizer, 1, max_seq_len, ranker_candidates)
    results = []
    avg_score_all = 0.0
    for item_idx in tqdm(range(len(dataset.items))):
        avg_item_reward = 0.0
        outputs = []
        for _ in range(runs_per_item):
            input_ids, attn_masks = envs.use_item(item_idx)
            while True:
                logits = get_llm_logits(p_net, input_ids.to(device=device), attn_masks.to(device=device))
                actions = Categorical(logits=logits).sample().to(device="cpu")
                text = envs.get_current_texts()[0]
                (input_ids, attn_masks), rewards, dones, truncs, _ = envs.step(actions)
                if dones[0] or truncs[0]:
                    avg_item_reward += rewards[0]
                    avg_score_all += rewards[0]
                    outputs.append(EvalResultOutput(text=text, score=rewards[0]))
                    break
        avg_item_reward = avg_item_reward / runs_per_item
        results.append(EvalResult(item_idx=item_idx, outputs=outputs, avg_score=avg_item_reward))
    avg_score_all = avg_score_all / (runs_per_item * len(dataset.items))
    return EvalResults(args=args, results=results, avg_score=avg_score_all)
