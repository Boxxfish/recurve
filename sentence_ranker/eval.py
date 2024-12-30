from pathlib import Path
from typing import *
from pydantic import BaseModel
import torch
from torch.distributions import Categorical
from tqdm import tqdm
from yaml import load, Loader # type: ignore
from transformers import AutoTokenizer, LlamaForCausalLM # type: ignore
from peft.peft_model import PeftModelForCausalLM

from sentence_ranker.train import ExpMeta
from sentence_ranker.datasets import DSEnvs, Dataset
from sentence_ranker.utils import get_llm_logits, parse_args


class Args(BaseModel):
    exp_path: str
    chkpt_label: str = "latest"
    out: str = "eval_results.json"
    dataset: str
    runs_per_item: int = 1
    device: str = "cuda"

class EvalResultOutput(BaseModel):
    text: str
    score: float

class EvalResult(BaseModel):
    item_idx: int
    outputs: List[EvalResultOutput]
    avg_score: float

class EvalResults(BaseModel):
    args: Args
    results: List[EvalResult]
    avg_score: float

def main():
    args = parse_args(Args)
    exp_dir = Path(args.exp_path)
    chkpt_dir = exp_dir / "checkpoints"
    with open(exp_dir / "meta.json", "r") as f:
        exp_meta = ExpMeta.model_validate_json(f.read())
    device = torch.device(args.device)

    # Load model
    p_net_dir = chkpt_dir / f"gen_p_net-{args.chkpt_label}"
    p_net_base = LlamaForCausalLM.from_pretrained(exp_meta.args.generator_base)
    p_net = PeftModelForCausalLM.from_pretrained(p_net_base, p_net_dir)
    p_net.to(device)
    tokenizer = AutoTokenizer.from_pretrained(exp_meta.args.generator_base)

    # Evaluate on dataset
    with open(args.dataset, "r") as f:
        data = load(f, Loader=Loader)
        dataset = Dataset.model_validate(data)
    done_fn, score_fn = dataset.get_done_score_fns()
    envs = DSEnvs(dataset, tokenizer, 1, exp_meta.args.max_seq_len, done_fn, score_fn)
    results = []
    avg_score_all = 0.0
    for item_idx in tqdm(range(len(dataset.items))):
        avg_item_reward = 0.0
        outputs = []
        for _ in range(args.runs_per_item):
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
        avg_item_reward = avg_item_reward / args.runs_per_item
        results.append(EvalResult(item_idx=item_idx, outputs=outputs, avg_score=avg_item_reward))
    avg_score_all = avg_score_all / (args.runs_per_item * len(dataset.items))

    with open(args.out, "w") as f:
        f.write(EvalResults(args=args, results=results, avg_score=avg_score_all).model_dump_json(indent=2))

if __name__ == "__main__":
    main()
