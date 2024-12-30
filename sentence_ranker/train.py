import copy
from typing import *
from pydantic import BaseModel
import torch
from tqdm import tqdm
from transformers.models.llama import LlamaForSequenceClassification, LlamaForCausalLM, LlamaTokenizerFast  # type: ignore
from torch.distributions import Categorical
from yaml import load, Loader # type: ignore
import wandb
from peft import LoraConfig, LoraModel

from sentence_ranker.algorithms.ppo import train_ppo
from sentence_ranker.algorithms.replay_buffer import ReplayBuffer
from sentence_ranker.algorithms.rollout_buffer import RolloutBuffer
from sentence_ranker.datasets import DSEnvs, Dataset
from sentence_ranker.utils import create_directory, get_llm_logits, parse_args


class Args(BaseModel):
    exp_base_dir: str = "./runs"
    iterations: int = 1_000
    device: str = "cuda"
    dataset: str
    save_every: int = 10

    # Ranker settings
    ranker_base: str = "meta-llama/Llama-3.2-1B"

    # Generator settings
    generator_base: str = "meta-llama/Llama-3.2-1B-Instruct"
    generator_sft: str = "meta-llama/Llama-3.2-1B-Instruct"
    max_seq_len: int = (
        256  # Max length of an input, including the prompt + special tokens
    )
    gen_num_envs: int = 2  # Number of parallel generations during sampling
    gen_num_steps: int = 512  # How many steps to take during sampling
    gen_train_mode: Literal["frozen", "sentence", "full"] = "frozen"
    gen_train_loop_iters: int = 10 # Number of times to run the generator training loop per iteration
    gen_train_iters: int = 2 # Number of training iterations per generatior training iteration
    gen_train_batch_size: int = 8 # Batch size for PPO
    gen_lambda: float = 0.95 # Lambda for GAE
    gen_epsilon: float = 0.2  # Probability ratio cutoff for PPO
    gen_p_lr: float = 0.0001
    gen_v_lr: float = 0.001
    gen_grad_steps: int = 1 # Number of gradient steps to take during PPO
    gen_entropy_coeff: float = 0.0 # Entropy coefficient for PPO
    gen_sft_coeff: float = 0.1 # SFT coefficient for PPO (Prevents logit collapse)


class ExpMeta(BaseModel):
    args: Args


class GeneratorTrainer:
    def __init__(self, args: Args, ds: Dataset):
        self.tokenizer = LlamaTokenizerFast.from_pretrained(args.generator_base)

        # Policy model
        p_net_lora_cfg = LoraConfig(
            task_type="CAUSAL_LM",
            r=8,
            lora_alpha=32,
            lora_dropout=0.1,
        )
        self.p_net = LlamaForCausalLM.from_pretrained(args.generator_base)
        self.p_net.add_adapter(p_net_lora_cfg, "default")
        
        # Value model
        v_net_lora_cfg = LoraConfig(
            task_type="SEQ_CLS",
            r=8,
            lora_alpha=32,
            lora_dropout=0.1,
        )
        v_net_cfg = copy.deepcopy(self.p_net.config)
        v_net_cfg.num_labels = 1
        v_net_cfg.pad_token_id = self.tokenizer.pad_token_type_id
        self.v_net = LlamaForSequenceClassification.from_pretrained(args.generator_base, config=v_net_cfg)
        self.v_net.add_adapter(v_net_lora_cfg, "default")

        self.sft_net = LlamaForCausalLM.from_pretrained(args.generator_sft)

        self.p_opt = torch.optim.Adam(self.p_net.parameters(), lr=args.gen_p_lr)
        self.v_opt = torch.optim.Adam(self.v_net.parameters(), lr=args.gen_v_lr)
        self.buffer = RolloutBuffer(
            args.max_seq_len,
            int(self.p_net.vocab_size),
            args.gen_num_envs,
            args.gen_num_steps,
        )
        done_fn, score_fn = ds.get_done_score_fns()
        self.envs = DSEnvs(ds, self.tokenizer, args.gen_num_envs, args.max_seq_len, done_fn, score_fn)

class RankerTrainer:

    def __init__(self):
        pass
        # self.q_net = LlamaForSequenceClassification()
        # self.buffer = ReplayBuffer()


def main():
    args = parse_args(Args)
    
    # Wandb
    wandb.init(
        project="sentence-ranker",
        config=args.model_dump(),
    )

    # Set up experiment
    exp_meta = ExpMeta(args=args)
    chkpt_dir = create_directory(args.exp_base_dir, exp_meta)

    # Load dataset
    with open(args.dataset, "r") as f:
        data = load(f, Loader=Loader)
        dataset = Dataset.model_validate(data)

    # Set up trainers
    gen_trainer = GeneratorTrainer(args, dataset)

    device = torch.device(args.device)
    for train_step in tqdm(range(args.iterations), position=0):
        ####
        ## Generator
        ####
        if args.gen_train_mode != "frozen":
            input_ids, attn_masks = gen_trainer.envs.reset()
            avg_p_loss, avg_v_loss = 0.0, 0.0
            for _ in range(args.gen_train_loop_iters):
                # Sample with generator policy
                with torch.no_grad():
                    # Step through env with generator policy and collect transitions
                    print("Collecting generator transitions...")
                    gen_trainer.p_net.to(device)
                    for _ in tqdm(range(args.gen_num_steps), position=1):
                        logits = get_llm_logits(gen_trainer.p_net, input_ids.to(device=device), attn_masks.to(device=device))
                        actions = Categorical(logits=logits).sample().to(device="cpu")
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
                    gen_trainer.p_net.cpu()

                    # Run SFT model over transitions and collect logits
                    print("Collecting SFT logits...")
                    gen_trainer.sft_net.to(device)
                    for buffer_idx in tqdm(range(args.gen_num_steps), position=1):
                        input_ids = gen_trainer.buffer.input_ids[buffer_idx]
                        attn_masks = gen_trainer.buffer.attn_masks[buffer_idx]
                        logits = get_llm_logits(gen_trainer.sft_net, input_ids.to(device=device), attn_masks.to(device=device))
                        gen_trainer.buffer.sft_action_probs[buffer_idx].copy_(logits)

                    gen_trainer.sft_net.cpu()

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
                    args.gen_sft_coeff,
                    gradient_steps=args.gen_grad_steps,
                    entropy_coeff=args.gen_entropy_coeff,
                )
                gen_trainer.buffer.clear()
                avg_p_loss += total_p_loss
                avg_v_loss += total_v_loss
            
        # Log metrics
        avg_p_loss = avg_p_loss / args.gen_train_loop_iters
        avg_v_loss = avg_v_loss / args.gen_train_loop_iters
        wandb.log({
            "gen_avg_p_loss": avg_p_loss,
            "gen_avg_v_loss": avg_v_loss,
        })

        # Save artifacts
        if train_step % args.save_every == 0:
            for label in [str(train_step), "latest"]:
                gen_trainer.p_net.save_pretrained(str(chkpt_dir / f"gen_p_net-{label}"), from_pt=True)
                gen_trainer.v_net.save_pretrained(str(chkpt_dir / f"gen_v_net-{label}"), from_pt=True)

if __name__ == "__main__":
    main()
