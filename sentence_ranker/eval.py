from pathlib import Path
from typing import *
from pydantic import BaseModel
import torch
from yaml import load, Loader # type: ignore
from transformers import AutoTokenizer, LlamaForCausalLM # type: ignore
from peft.peft_model import PeftModelForCausalLM

from sentence_ranker.eval_utils import run_eval
from sentence_ranker.train import ExpMeta
from sentence_ranker.datasets import Dataset
from sentence_ranker.utils import parse_args


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
    eval_results = run_eval(args, dataset, tokenizer, exp_meta.args.max_seq_len, args.runs_per_item, p_net, device)

    with open(args.out, "w") as f:
        f.write(eval_results.model_dump_json(indent=2))

if __name__ == "__main__":
    main()
