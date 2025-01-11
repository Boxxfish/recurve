from pathlib import Path
from typing import *
from pydantic import BaseModel
import torch
from yaml import load, Loader  # type: ignore
from transformers import AutoTokenizer, LlamaForCausalLM, LlamaForSequenceClassification, LlamaConfig  # type: ignore
from peft.peft_model import PeftModelForCausalLM, PeftModelForSequenceClassification

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


@torch.no_grad()
def main():
    args = parse_args(Args)
    exp_dir = Path(args.exp_path)
    chkpt_dir = exp_dir / "checkpoints"
    with open(exp_dir / "meta.json", "r") as f:
        exp_meta = ExpMeta.model_validate_json(f.read())
    device = torch.device(args.device)

    # Load models
    tokenizer = AutoTokenizer.from_pretrained(exp_meta.args.generator_base)

    p_net_dir = chkpt_dir / f"gen_p_net-{args.chkpt_label}"
    p_net = LlamaForCausalLM.from_pretrained(p_net_dir)
    p_net.eval()

    q_net_dir = chkpt_dir / f"ranker_q_net-{args.chkpt_label}"
    q_net_cfg = LlamaConfig.from_pretrained(exp_meta.args.ranker_base)
    q_net_cfg.num_labels = 1
    q_net_cfg.pad_token_id = tokenizer.pad_token_type_id
    q_net = LlamaForSequenceClassification.from_pretrained(q_net_dir)
    q_net.eval()

    # Evaluate on dataset
    with open(args.dataset, "r") as f:
        data = load(f, Loader=Loader)
        dataset = Dataset.model_validate(data)
    eval_results = run_eval(
        args,
        dataset,
        tokenizer,
        exp_meta.args.max_seq_len,
        args.runs_per_item,
        p_net,
        device,
        exp_meta.args.ranker_candidates,
        q_net,
        exp_meta.args.ranker_use_sigmoid,
    )

    with open(args.out, "w") as f:
        f.write(eval_results.model_dump_json(indent=2))


if __name__ == "__main__":
    main()
