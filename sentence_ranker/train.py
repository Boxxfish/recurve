from typing import *
from pydantic import BaseModel
import torch
from torch import Tensor
from tqdm import tqdm
from transformers.modeling_outputs import CausalLMOutputWithPast # type: ignore
from transformers.models.llama import LlamaForSequenceClassification, LlamaForCausalLM, LlamaTokenizerFast  # type: ignore
from torch.distributions import Categorical
from yaml import load # type: ignore
from yaml import Loader

from sentence_ranker.algorithms.ppo import train_ppo
from sentence_ranker.algorithms.replay_buffer import ReplayBuffer
from sentence_ranker.algorithms.rollout_buffer import RolloutBuffer
from sentence_ranker.datasets import DSEnvs, Dataset
from sentence_ranker.utils import create_directory, parse_args


class Args(BaseModel):
    exp_base_dir: str = "./runs"
    iterations: int = 1_000
    device: str = "cuda"
    dataset: str

    # Ranker settings
    ranker_base: str = "meta-llama/Llama-3.2-1B"

    # Generator settings
    generator_base: str = "meta-llama/Llama-3.2-1B-Instruct"
    max_seq_len: int = (
        256  # Max length of an input, including the prompt + special tokens
    )
    gen_num_envs: int = 8  # Number of parallel generations during sampling
    gen_num_steps: int = 512  # How many steps to take during sampling
    gen_train_mode: Literal["frozen", "sentence", "full"] = "frozen"
    gen_train_loop_iters: int = 10 # Number of times to run the generator training loop per iteration
    gen_train_iters: int = 2 # Number of training iterations per generatior training iteration
    gen_train_batch_size: int = 8 # Batch size for PPO
    gen_lambda: float = 0.95 # Lambda for GAE
    gen_epsilon: float = 0.2  # Probability ratio cutoff for PPO
    gen_p_lr: float = 0.0001
    gen_v_lr: float = 0.001


class ExpMeta(BaseModel):
    args: Args


class GeneratorTrainer:
    def __init__(self, args: Args, ds: Dataset):
        self.p_net = LlamaForCausalLM.from_pretrained(args.ranker_base)
        self.v_net = LlamaForSequenceClassification.from_pretrained(args.ranker_base)
        self.p_opt = torch.optim.Adam(self.p_net.parameters(), lr=args.gen_p_lr)
        self.v_opt = torch.optim.Adam(self.v_net.parameters(), lr=args.gen_v_lr)
        self.buffer = RolloutBuffer(
            args.max_seq_len,
            int(self.p_net.vocab_size),
            args.gen_num_envs,
            args.gen_num_steps,
        )
        self.tokenizer = LlamaTokenizerFast.from_pretrained(args.ranker_base)
        locs: Dict[str, Any] = {}
        exec(compile(ds.done_fn + "\n\n" + ds.score_fn, "", "exec"), locs)
        done_fn, score_fn = locs["done_fn"], locs["score_fn"]
        self.envs = DSEnvs(ds, self.tokenizer, args.gen_num_envs, args.max_seq_len, done_fn, score_fn)

def get_llm_logits(p_net: LlamaForCausalLM, input_ids: Tensor, attn_masks: Tensor) -> Tensor:
    model_out: CausalLMOutputWithPast = p_net.forward(input_ids, attn_masks, return_dict=True)
    return model_out.logits

class RankerTrainer:

    def __init__(self):
        pass
        # self.q_net = LlamaForSequenceClassification()
        # self.buffer = ReplayBuffer()


def main():
    # Set up experiment
    args = parse_args(Args)
    exp_meta = ExpMeta(args=args)
    create_directory(args.exp_base_dir, exp_meta)

    # Load dataset
    with open(args.dataset, "r") as f:
        data = load(f, Loader=Loader)
        dataset = Dataset.model_validate(data)

    # Set up trainers
    gen_trainer = GeneratorTrainer(args, dataset)

    device = torch.device(args.device)
    for _ in tqdm(range(args.iterations), position=0):
        ####
        ## Generator
        ####
        if args.gen_train_mode != "frozen":
            input_ids, attn_masks = gen_trainer.envs.reset()
            for _ in range(args.gen_train_loop_iters):
                # Sample with generator policy
                with torch.no_grad():
                    for _ in tqdm(range(args.gen_num_steps), position=1):
                        logits = get_llm_logits(gen_trainer.p_net, input_ids, attn_masks)
                        actions = Categorical(logits=logits).sample()
                        (input_ids_, attn_masks_), rewards, dones, truncs, _ = gen_trainer.envs.step(actions)
                        gen_trainer.buffer.insert_step(
                            input_ids, attn_masks,
                            actions.unsqueeze(-1),
                            logits,
                            rewards,
                            dones,
                            truncs,
                        )
                        input_ids, attn_masks = input_ids_, attn_masks_
                    gen_trainer.buffer.insert_final_step(input_ids, attn_masks)

                # Train generator
                total_p_loss, total_v_loss = train_ppo(
                    gen_trainer.p_net,
                    gen_trainer.v_net,
                    gen_trainer.p_opt,
                    gen_trainer.v_opt,
                    gen_trainer.buffer,
                    device,
                    args.gen_train_iters,
                    args.gen_train_batch_size,
                    1.0,
                    args.gen_lambda,
                    args.gen_epsilon,
                )
                gen_trainer.buffer.clear()


if __name__ == "__main__":
    main()
