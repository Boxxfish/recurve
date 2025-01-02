import copy
import random
from typing import *
from pydantic import BaseModel
import torch
from tqdm import tqdm
from transformers.models.llama import LlamaConfig, LlamaForSequenceClassification, LlamaForCausalLM, LlamaTokenizerFast  # type: ignore
from torch.distributions import Categorical
from yaml import load, Loader # type: ignore
import wandb
from peft import LoraConfig

from sentence_ranker.algorithms.dqn import train_dqn
from sentence_ranker.algorithms.ppo import train_ppo
from sentence_ranker.algorithms.replay_buffer import ReplayBuffer
from sentence_ranker.algorithms.rollout_buffer import RolloutBuffer
from sentence_ranker.datasets import DSEnvs, Dataset
from sentence_ranker.eval_utils import run_eval
from sentence_ranker.utils import create_directory, get_llm_logits, get_llm_scores, parse_args


class Args(BaseModel):
    exp_base_dir: str = "./runs"
    iterations: int = 100_000
    device: str = "cuda"
    dataset: str
    save_every: int = 10
    eval_every: int = 10
    num_eval_runs: int = 10

    # Ranker settings
    ranker_base: str = "meta-llama/Llama-3.2-1B"
    ranker_candidates: int = 10 # Number of candidate sentences to generate on each timestep
    ranker_train_loop_iters: int = 10 # Number of times to run the ranker training loop per iteration
    ranker_buffer_size: int = 10_000  # Number of elements that can be stored in the buffer
    ranker_target_update: int = 50  # Number of iterations before updating Q target
    ranker_q_epsilon: float = 1.0 # Epsilon for epsilon greedy strategy. This gets annealed over time.
    ranker_train_iters: int = 64 # Number of training iterations per ranker training iteration
    ranker_q_lr: float = 0.0001
    ranker_train_batch_size: int = 1 # Batch size for DQN
    ranker_grad_steps: int = 32 # Number of gradient steps to take during DQN

    # Generator settings
    generator_base: str = "meta-llama/Llama-3.2-1B-Instruct"
    generator_sft: str = "meta-llama/Llama-3.2-1B-Instruct"
    max_seq_len: int = (
        256  # Max length of an input, including the prompt + special tokens
    )
    gen_num_envs: int = 2  # Number of parallel generations during sampling
    gen_num_steps: int = 512  # How many steps to take during sampling
    gen_train_mode: Literal["frozen", "sentence", "full"] = "frozen"
    gen_train_loop_iters: int = 3 # Number of times to run the generator training loop per iteration
    gen_train_iters: int = 2 # Number of training iterations per generator training iteration
    gen_train_batch_size: int = 8 # Batch size for PPO
    gen_lambda: float = 0.95 # Lambda for GAE
    gen_epsilon: float = 0.2  # Probability ratio cutoff for PPO
    gen_p_lr: float = 0.00001
    gen_v_lr: float = 0.0001
    gen_grad_steps: int = 1 # Number of gradient steps to take during PPO
    gen_entropy_coeff: float = 0.0 # Entropy coefficient for PPO
    gen_sft_coeff: float = 0.1 # SFT coefficient for PPO (Prevents logit collapse)
    gen_train_after: int = 10 # Number of steps to wait until generator training starts


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
        self.envs = DSEnvs(ds, self.tokenizer, args.gen_num_envs, args.max_seq_len, 1)

class RankerTrainer:

    def __init__(self, args: Args, ds: Dataset):
        self.tokenizer = LlamaTokenizerFast.from_pretrained(args.ranker_base)

        # Q model
        q_net_lora_cfg = LoraConfig(
            task_type="SEQ_CLS",
            r=8,
            lora_alpha=32,
            lora_dropout=0.1,
        )
        q_net_cfg = copy.deepcopy(LlamaConfig.from_pretrained(args.ranker_base))
        q_net_cfg.num_labels = 1
        q_net_cfg.pad_token_id = self.tokenizer.pad_token_type_id
        self.q_net = LlamaForSequenceClassification.from_pretrained(args.ranker_base, config=q_net_cfg)
        self.q_net.add_adapter(q_net_lora_cfg, "default")
        
        self.target_q_net = copy.deepcopy(self.q_net)

        self.q_opt = torch.optim.Adam(self.q_net.parameters(), lr=args.ranker_q_lr)
        self.buffer = ReplayBuffer(
            args.max_seq_len,
            args.ranker_candidates,
            args.ranker_buffer_size,
        )
        self.envs = DSEnvs(ds, self.tokenizer, 1, args.max_seq_len, args.ranker_candidates)


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
        input_ids, attn_masks = ranker_trainer.envs.reset_ranker(gen_trainer.p_net)
        print("Filling replay buffer:")
        with tqdm(total=ranker_trainer.buffer.capacity) as pbar:
            while not ranker_trainer.buffer.filled:
                action = random.randrange(0, args.ranker_candidates)
                actions = torch.tensor([action])
                (next_input_ids, next_attn_masks), rewards, dones, truncs, _ = ranker_trainer.envs.step_ranker(actions, gen_trainer.p_net)
                ranker_trainer.buffer.insert_step(
                    input_ids,
                    attn_masks,
                    next_input_ids,
                    next_attn_masks,
                    actions.unsqueeze(-1),
                    rewards,
                    dones,
                    None,
                    None,
                )
                input_ids = next_input_ids
                attn_masks = next_attn_masks
                pbar.update(1)
    gen_trainer.p_net.cpu()

    for train_step in tqdm(range(args.iterations), position=0):
        ####
        ## Ranker
        ####
        percent_done = train_step / args.iterations
        avg_q_loss = 0.0
        for _ in range(args.ranker_train_loop_iters):
            # Collect experience
            gen_trainer.p_net.to(device)
            input_ids, attn_masks = ranker_trainer.envs.reset_ranker(gen_trainer.p_net)
            gen_trainer.p_net.cpu()
            with torch.no_grad():
                while True:
                    if (
                        random.random() < args.ranker_q_epsilon * max(1.0 - percent_done, 0.05)
                    ):
                        action = random.randrange(0, args.ranker_candidates)
                    else:
                        ranker_trainer.q_net.to(device)
                        q_vals = get_llm_scores(ranker_trainer.q_net, input_ids.to(device), attn_masks.to(device)).squeeze(0) # Shape: (num_candidates)
                        ranker_trainer.q_net.cpu()
                        action = q_vals.argmax(0).cpu().item()
                    actions = torch.tensor([action])
                    gen_trainer.p_net.to(device)
                    (next_input_ids, next_attn_masks), rewards, dones, truncs, _ = ranker_trainer.envs.step_ranker(actions, gen_trainer.p_net)
                    gen_trainer.p_net.cpu()
                    ranker_trainer.buffer.insert_step(
                        input_ids,
                        attn_masks,
                        next_input_ids,
                        next_attn_masks,
                        actions.unsqueeze(-1),
                        rewards,
                        dones,
                        None,
                        None,
                    )
                    input_ids = next_input_ids
                    attn_masks = next_attn_masks

                    if dones[0] or truncs[0]:
                        break

            total_q_loss = train_dqn(
                ranker_trainer.q_net,
                ranker_trainer.target_q_net,
                ranker_trainer.q_opt,
                ranker_trainer.buffer,
                device,
                args.ranker_train_iters,
                args.ranker_train_batch_size,
                1.0,
                args.ranker_grad_steps
            )
            avg_q_loss += total_q_loss

        # Update Q target
        if train_step % args.ranker_target_update == 0:
            ranker_trainer.target_q_net.load_state_dict(ranker_trainer.q_net.state_dict())

        ####
        ## Generator
        ####
        avg_p_loss, avg_v_loss = 0.0, 0.0
        if args.gen_train_mode != "frozen" and train_step > args.gen_train_after:
            input_ids, attn_masks = gen_trainer.envs.reset()
            for _ in range(args.gen_train_loop_iters):
                # Sample with generator policy
                with torch.no_grad():
                    # Step through env with generator policy and collect transitions
                    print("Collecting generator transitions...")
                    gen_trainer.p_net.to(device)
                    for _ in tqdm(range(args.gen_num_steps), position=1):
                        logits, _ = get_llm_logits(gen_trainer.p_net, input_ids.to(device=device), attn_masks.to(device=device))
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
                        logits, _ = get_llm_logits(gen_trainer.sft_net, input_ids.to(device=device), attn_masks.to(device=device))
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
        avg_q_loss = avg_q_loss / args.ranker_train_loop_iters
        avg_p_loss = avg_p_loss / args.gen_train_loop_iters
        avg_v_loss = avg_v_loss / args.gen_train_loop_iters
        log_dict = {
            "gen_avg_q_loss": avg_q_loss,
            "gen_avg_p_loss": avg_p_loss,
            "gen_avg_v_loss": avg_v_loss,
        }

        # Run eval
        if train_step % args.eval_every == 0:
            eval_results = run_eval({}, dataset, gen_trainer.tokenizer, args.max_seq_len, args.num_eval_runs, gen_trainer.p_net, device, args.ranker_candidates, ranker_trainer.q_net)
            log_dict.update({
                "eval_score": eval_results.avg_score
            })
            for label in [str(train_step), "latest"]:
                with open(eval_dir / f"eval-{label}.json", "w") as f:
                    f.write(eval_results.model_dump_json(indent=2))

        wandb.log(log_dict)

        # Save artifacts
        if train_step % args.save_every == 0:
            for label in [str(train_step), "latest"]:
                ranker_trainer.q_net.save_pretrained(str(chkpt_dir / f"ranker_q_net-{label}"), from_pt=True)
                gen_trainer.p_net.save_pretrained(str(chkpt_dir / f"gen_p_net-{label}"), from_pt=True)
                gen_trainer.v_net.save_pretrained(str(chkpt_dir / f"gen_v_net-{label}"), from_pt=True)

if __name__ == "__main__":
    main()
