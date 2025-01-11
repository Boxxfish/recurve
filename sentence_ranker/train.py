import copy
import random
from typing import *
from pydantic import BaseModel
import torch
from tqdm import tqdm
from transformers.models.llama import LlamaConfig, LlamaForSequenceClassification, LlamaForCausalLM, LlamaTokenizerFast  # type: ignore
from torch.distributions import Categorical
from yaml import load, Loader  # type: ignore
import wandb
from peft import LoraConfig

from sentence_ranker.algorithms.dqn import train_dqn
from sentence_ranker.algorithms.ppo import train_ppo
from sentence_ranker.algorithms.replay_buffer import ReplayBuffer
from sentence_ranker.datasets import DSEnvs, Dataset
from sentence_ranker.eval_utils import run_eval
from sentence_ranker.utils import (
    create_directory,
    get_llm_logits,
    get_llm_logits_candidates,
    get_llm_scores,
    parse_args,
)


class Args(BaseModel):
    exp_base_dir: str = "./runs"
    iterations: int = 10_000
    device: str = "cuda"
    dataset: str
    save_every: int = 10
    eval_every: int = 10
    num_eval_runs: int = 10

    # Ranker settings
    ranker_base: str = "meta-llama/Llama-3.2-1B"
    ranker_sample_steps: int = 4 # Number of steps the ranker takes when sampling
    ranker_candidates: int = (
        8  # Number of candidate sentences to generate on each timestep
    )
    ranker_buffer_size: int = (
        10_000  # Number of elements that can be stored in the buffer
    )
    ranker_target_update: int = 100  # Number of iterations before updating Q target
    ranker_q_epsilon: float = (
        1.0  # Epsilon for epsilon greedy strategy. This gets annealed over time.
    )
    ranker_train_iters: int = (
        16  # Number of training iterations per ranker training iteration
    )
    ranker_q_lr: float = 0.0001
    ranker_train_batch_size: int = 4  # Batch size for DQN
    ranker_grad_steps: int = 16  # Number of gradient steps to take during DQN
    ranker_use_sigmoid: bool = (
        False  # Whether Q values should be passed through the sigmoid function (Helps with overestimation)
    )

    # Generator settings
    generator_base: str = "meta-llama/Llama-3.2-1B-Instruct"
    generator_sft: str = "meta-llama/Llama-3.2-1B-Instruct"
    max_seq_len: int = (
        256  # Max length of an input, including the prompt + special tokens
    )
    gen_train_mode: Literal["frozen", "sentence"] = "sentence"
    gen_train_iters: int = (
        2  # Number of training iterations per generator training iteration
    )
    gen_lambda: float = 0.95  # Lambda for GAE
    gen_epsilon: float = 0.2  # Probability ratio cutoff for PPO
    gen_p_lr: float = 0.00001
    gen_v_lr: float = 0.0001
    gen_grad_steps: int = 1  # Number of gradient steps to take during PPO
    gen_entropy_coeff: float = 0.03  # Entropy coefficient for PPO
    gen_sft_coeff: float = 0.1  # SFT coefficient for PPO (Prevents logit collapse)
    gen_train_after: int = 400  # Number of steps to wait until generator training starts


class ExpMeta(BaseModel):
    args: Args


class GeneratorTrainer:
    def __init__(self, args: Args, ds: Dataset):
        self.tokenizer = LlamaTokenizerFast.from_pretrained(args.generator_base)

        # Policy model
        self.p_net = LlamaForCausalLM.from_pretrained(args.generator_base)
        if len(self.p_net.active_adapters()) == 0:
            p_net_lora_cfg = LoraConfig(
                task_type="CAUSAL_LM",
                r=8,
                lora_alpha=32,
                lora_dropout=0.1,
            )
            self.p_net.add_adapter(p_net_lora_cfg, "default")
        self.p_net.cpu()

        # Value model
        v_net_cfg = copy.deepcopy(self.p_net.config)
        v_net_cfg.num_labels = 1
        v_net_cfg.pad_token_id = self.tokenizer.pad_token_type_id
        self.v_net = LlamaForSequenceClassification.from_pretrained(
            args.generator_base, config=v_net_cfg
        )
        if len(self.v_net.active_adapter()) == 0:
            v_net_lora_cfg = LoraConfig(
                task_type="SEQ_CLS",
                r=8,
                lora_alpha=32,
                lora_dropout=0.1,
            )
            self.v_net.add_adapter(v_net_lora_cfg, "default")
        self.v_net.cpu()

        self.sft_net = LlamaForCausalLM.from_pretrained(args.generator_sft)
        self.sft_net.cpu()

        self.p_opt = torch.optim.Adam(self.p_net.parameters(), lr=args.gen_p_lr)
        self.v_opt = torch.optim.Adam(self.v_net.parameters(), lr=args.gen_v_lr)


class RankerTrainer:

    def __init__(self, args: Args, ds: Dataset):
        self.tokenizer = LlamaTokenizerFast.from_pretrained(args.ranker_base)

        # Q model
        q_net_cfg = LlamaConfig.from_pretrained(args.ranker_base)
        q_net_cfg.num_labels = 1
        q_net_cfg.pad_token_id = self.tokenizer.pad_token_type_id
        self.q_net = LlamaForSequenceClassification.from_pretrained(
            args.ranker_base, config=q_net_cfg
        )
        if len(self.q_net.active_adapters()) == 0:
            q_net_lora_cfg = LoraConfig(
                task_type="SEQ_CLS",
                r=8,
                lora_alpha=32,
                lora_dropout=0.1,
                modules_to_save=["score.weight"]
            )
            self.q_net.add_adapter(q_net_lora_cfg, "default")
        self.q_net.cpu()

        self.target_q_net = copy.deepcopy(self.q_net)
        self.target_q_net.cpu()

        self.q_opt = torch.optim.Adam(self.q_net.parameters(), lr=args.ranker_q_lr)
        vocab_size = int(LlamaConfig.from_pretrained(args.generator_base).vocab_size)
        self.buffer = ReplayBuffer(
            args.max_seq_len,
            args.ranker_candidates,
            args.ranker_buffer_size,
            vocab_size,
            args.ranker_sample_steps,
        )
        self.envs = DSEnvs(
            ds, self.tokenizer, args.max_seq_len, args.ranker_candidates, vocab_size
        )


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
    eval_dir = chkpt_dir.parent / "evals"
    eval_dir.mkdir(exist_ok=True)

    # Load dataset
    with open(args.dataset, "r") as f:
        data = load(f, Loader=Loader)
        dataset = Dataset.model_validate(data)

    # Set up trainers
    ranker_trainer = RankerTrainer(args, dataset)
    gen_trainer = GeneratorTrainer(args, dataset)

    # Fill replay buffer with experience
    device = torch.device(args.device)
    gen_trainer.p_net.to(device)
    with torch.no_grad():
        input_ids, attn_masks, logits, candidate_masks = ranker_trainer.envs.reset_ranker(gen_trainer.p_net)
        print("Filling replay buffer:")
        with tqdm(total=ranker_trainer.buffer.capacity) as pbar:
            while not ranker_trainer.buffer.filled:
                action = random.randrange(0, args.ranker_candidates)
                actions = torch.tensor([action])
                (next_input_ids, next_attn_masks, next_logits, next_candidate_masks), rewards, dones, _, _ = (
                    ranker_trainer.envs.step_ranker(actions, gen_trainer.p_net)
                )
                ranker_trainer.buffer.insert_step(
                    input_ids,
                    attn_masks,
                    next_input_ids,
                    next_attn_masks,
                    actions.unsqueeze(-1),
                    rewards,
                    dones,
                    None,
                    candidate_masks,
                    next_candidate_masks,
                    torch.zeros([1, args.ranker_candidates], dtype=torch.float), # Unecessary since we aren't training the generator yet
                )
                input_ids, attn_masks, logits, candidate_masks = next_input_ids, next_attn_masks, next_logits, next_candidate_masks
                pbar.update(1)
    gen_trainer.p_net.cpu()

    for train_step in tqdm(range(args.iterations), position=0):
        ####
        ## Collect experience
        ####
        percent_done = train_step / args.iterations
        with torch.no_grad():
            gen_trainer.p_net.to(device)
            input_ids, attn_masks, logits, candidate_masks = ranker_trainer.envs.reset_ranker(gen_trainer.p_net)
            gen_trainer.p_net.cpu()
            step_idxs: List[int] = []
            print("Collecting transitions...")
            for _ in range(args.ranker_sample_steps):
                # Compute Q values
                ranker_trainer.q_net.to(device)
                q_vals = get_llm_scores(
                    ranker_trainer.q_net,
                    input_ids.to(device),
                    attn_masks.to(device),
                    candidate_masks,
                    args.ranker_use_sigmoid,
                ).squeeze(
                    0
                ).cpu()  # Shape: (num_candidates)
                ranker_trainer.q_net.cpu()

                # Select action
                if random.random() < args.ranker_q_epsilon * max(
                    1.0 - percent_done, 0.05
                ):
                    action = random.randrange(0, args.ranker_candidates)
                else:
                    action = q_vals.argmax(0).item()
                actions = torch.tensor([action])

                # Perform a step in the environment
                gen_trainer.p_net.to(device)
                (next_input_ids, next_attn_masks, next_logits, next_candidate_masks), rewards, dones, _, _ = (
                    ranker_trainer.envs.step_ranker(actions, gen_trainer.p_net)
                )
                gen_trainer.p_net.cpu()

                # Compute candidate rewards
                candidate_rewards = q_vals
                if dones[0]:
                    candidate_rewards[action] = rewards[0]
                candidate_rewards.unsqueeze_(0)

                # Insert step results into buffer
                step_idxs.append(ranker_trainer.buffer.next)
                ranker_trainer.buffer.insert_step(
                    input_ids,
                    attn_masks,
                    next_input_ids,
                    next_attn_masks,
                    actions.unsqueeze(-1),
                    rewards,
                    dones,
                    logits,
                    candidate_masks,
                    next_candidate_masks,
                    candidate_rewards,
                )
                input_ids, attn_masks, logits, candidate_masks = next_input_ids, next_attn_masks, next_logits, next_candidate_masks

            # Run SFT model over transitions and collect logits
            print("Collecting SFT logits...")
            gen_trainer.sft_net.to(device)
            for logit_idx, buffer_idx in tqdm(enumerate(step_idxs), position=1):
                input_ids = ranker_trainer.buffer.input_ids[buffer_idx]
                attn_masks = ranker_trainer.buffer.attn_masks[buffer_idx]
                candidate_masks = ranker_trainer.buffer.candidate_masks[buffer_idx]
                logits = get_llm_logits_candidates(
                    gen_trainer.sft_net,
                    input_ids.to(device=device),
                    attn_masks.to(device=device),
                    candidate_masks,
                )
                ranker_trainer.buffer.sft_logits[logit_idx].copy_(logits)
            gen_trainer.sft_net.cpu()

        ####
        ## Ranker
        ####
        print("Training ranker...")
        avg_q_loss = 0.0
        total_q_loss = train_dqn(
            ranker_trainer.q_net,
            ranker_trainer.target_q_net,
            ranker_trainer.q_opt,
            ranker_trainer.buffer,
            device,
            args.ranker_train_iters,
            args.ranker_train_batch_size,
            1.0,
            args.ranker_grad_steps,
            args.ranker_use_sigmoid,
        )
        avg_q_loss += total_q_loss

        # Update Q target
        if train_step % args.ranker_target_update == 0:
            ranker_trainer.target_q_net.load_state_dict(
                ranker_trainer.q_net.state_dict()
            )

        ####
        ## Generator
        ####
        avg_p_loss, avg_v_loss = 0.0, 0.0
        if args.gen_train_mode != "frozen" and train_step >= args.gen_train_after:
            print("Training generator...")
            total_p_loss, total_v_loss = train_ppo(
                gen_trainer.p_net,
                gen_trainer.v_net,
                gen_trainer.p_opt,
                gen_trainer.v_opt,
                ranker_trainer.buffer,
                device,
                args.gen_train_iters,
                1.0,
                args.gen_lambda,
                args.gen_epsilon,
                args.gen_sft_coeff,
                args.gen_grad_steps,
                args.gen_entropy_coeff,
                step_idxs,
            )
            avg_p_loss += total_p_loss
            avg_v_loss += total_v_loss
        ranker_trainer.buffer.reset_logits()

        ####
        ## Metrics, evaluation, and saving
        ####
        # Log metrics
        log_dict = {
            "gen_avg_q_loss": avg_q_loss,
            "gen_avg_p_loss": avg_p_loss,
            "gen_avg_v_loss": avg_v_loss,
        }

        # Run eval
        if train_step % args.eval_every == 0:
            eval_results = run_eval(
                {},
                dataset,
                gen_trainer.tokenizer,
                args.max_seq_len,
                args.num_eval_runs,
                gen_trainer.p_net,
                device,
                args.ranker_candidates,
                ranker_trainer.q_net,
                args.ranker_use_sigmoid,
            )
            log_dict.update({"eval_score": eval_results.avg_score})
            for label in [str(train_step), "latest"]:
                with open(eval_dir / f"eval-{label}.json", "w") as f:
                    f.write(eval_results.model_dump_json(indent=2))

        wandb.log(log_dict)

        # Save artifacts
        if train_step % args.save_every == 0:
            for label in [str(train_step), "latest"]:
                ranker_trainer.q_net.save_pretrained(
                    str(chkpt_dir / f"ranker_q_net-{label}"), from_pt=True
                )
                gen_trainer.p_net.save_pretrained(
                    str(chkpt_dir / f"gen_p_net-{label}"), from_pt=True
                )
                gen_trainer.v_net.save_pretrained(
                    str(chkpt_dir / f"gen_v_net-{label}"), from_pt=True
                )


if __name__ == "__main__":
    main()
