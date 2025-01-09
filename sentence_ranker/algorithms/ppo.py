import copy
import random
from typing import List, Tuple

import torch
from torch import nn
from torch.distributions import Categorical
from tqdm import tqdm
from transformers import DynamicCache, LlamaForSequenceClassification, LlamaForCausalLM

from sentence_ranker.utils import get_llm_logits

from .replay_buffer import ReplayBuffer


def train_ppo(
    p_net: LlamaForCausalLM,
    v_net: LlamaForSequenceClassification,
    p_opt: torch.optim.Optimizer,
    v_opt: torch.optim.Optimizer,
    buffer: ReplayBuffer,
    device: torch.device,
    train_iters: int,
    discount: float,
    lambda_: float,
    epsilon: float,
    sft_coeff: float,
    gradient_steps: int,
    entropy_coeff: float,
    step_idxs: List[int],
) -> Tuple[float, float]:
    """
    Performs the PPO training loop. Returns a tuple of total policy loss and
    total value loss.
    This loop has been optimized for ArCHer style training.
    """
    p_net.cpu()
    v_net.cpu()

    p_net.train()
    v_net_frozen = copy.deepcopy(v_net)
    v_net.train()

    total_v_loss = 0.0
    total_p_loss = 0.0

    p_opt.zero_grad()
    v_opt.zero_grad()

    optim_step = 0
    for _ in tqdm(range(train_iters), position=1):
        shuffled_step_idxs = copy.deepcopy(step_idxs)
        random.shuffle(shuffled_step_idxs)
        for step_idx in shuffled_step_idxs:
            # Get data from step
            input_ids = buffer.input_ids[
                step_idx
            ]  # Shape: (num_candidates, max_seq_len)
            attn_masks = buffer.attn_masks[
                step_idx
            ]  # Shape: (num_candidates, max_seq_len)
            logits = buffer.logits[
                step_idx
            ]  # Shape: (num_candidates, max_seq_len, vocab_size)
            sft_logits = buffer.sft_logits[
                step_idx
            ]  # Shape: (num_candidates, max_seq_len, vocab_size)
            candidate_masks = buffer.candidate_masks[
                step_idx
            ]  # Shape: (num_candidates, max_seq_len)
            candidate_rewards = buffer.candidate_rewards[
                step_idx
            ]  # Shape: (num_candidates)
            actions = input_ids[candidate_masks]  # Shape: (num_tokens)

            input_lens = attn_masks.byte().argmin(1) - 1  # Shape: (num_candidates)
            max_len = input_lens.amax() + 1

            num_candidates, max_seq_len, vocab_size = logits.shape

            with torch.no_grad():
                returns = torch.zeros(
                    [max_seq_len, num_candidates],
                    dtype=torch.float,
                )
                advantages = torch.zeros(
                    [max_seq_len, num_candidates],
                    dtype=torch.float,
                )
                v_net_frozen.to(device)
                step_returns: torch.Tensor = (
                    v_net_frozen(
                        input_ids.to(device),
                        attn_masks.to(device),
                    )
                    .logits.squeeze()
                    .cpu()
                )  # Shape: (num_candidates)

                # Compute state value estimates
                # TODO: Only compute minimum necessary state values
                # prefix_idx = candidate_masks[0].byte().argmax(0).item()
                cache = DynamicCache()
                state_values_all = torch.zeros(
                    [num_candidates, max_seq_len], dtype=torch.float
                )
                for i in range(max_seq_len):
                    v_output = v_net_frozen(
                        input_ids[:, i : i + 1].to(device),
                        attn_masks[:, : i + 1].to(device),
                        past_key_values=cache,
                    )
                    state_values_all[:, i] = v_output.logits.squeeze().cpu()
                    cache = v_output.past_key_values
                v_net_frozen.cpu()

                # Calculate advantage estimates and rewards to go
                state_values = step_returns.clone()
                step_advantages = torch.zeros([num_candidates], dtype=torch.float)
                for i in reversed(range(max_seq_len)):
                    dones = (input_lens == i).to(torch.float)  # Shape: (num_candidates)
                    rewards = candidate_rewards * dones  # Shape: (num_candidates)
                    inv_dones = 1.0 - dones  # Shape: (num_candidates)
                    prev_state_values = state_values_all[
                        :, i
                    ]  # Shape: (num_candidates)
                    # Delta is the difference between the 1 step bootstrap (reward +
                    # value prediction of next state) and the value prediction of
                    # the current state
                    delta = (
                        rewards
                        + discount * inv_dones * state_values
                        - prev_state_values
                    )
                    state_values = prev_state_values
                    step_advantages = (
                        delta + discount * lambda_ * inv_dones * step_advantages
                    )
                    step_returns = state_values + step_advantages
                    advantages[i] = step_advantages
                    returns[i] = step_returns

                # Extract token-level advantages and returns
                advantages = advantages.T[candidate_masks]  # Shape: (num_tokens)
                returns = returns.T[candidate_masks]  # Shape: (num_tokens)

            # Train policy network
            p_net.to(device)
            with torch.no_grad():
                logits[~candidate_masks] = 1.0
                old_act_log_probs = Categorical(
                    logits=logits
                ).logits  # Shape: (num_candidates, max_seq_len, vocab_size)
                old_act_log_probs = old_act_log_probs[
                    candidate_masks
                ]  # Shape: (num_tokens, vocab_size)
                old_act_log_probs = old_act_log_probs[
                    torch.nn.functional.one_hot(
                        actions.long(), num_classes=vocab_size
                    ).bool()
                ]  # Shape: (num_tokens)

            # TODO: Precompute the prefix when calculating new logits
            new_logits = p_net(
                input_ids[:, :max_len].to(device), attn_masks[:, :max_len].to(device)
            ).logits.cpu()  # Shape: (num_candidates, max_len, vocab_size)
            new_logits = torch.cat(
                [
                    new_logits,
                    torch.zeros(
                        [num_candidates, max_seq_len - max_len, vocab_size],
                        dtype=torch.float,
                    ),
                ],
                dim=1,
            )  # Shape: (num_candidates, max_seq_len, vocab_size)
            new_logits[~candidate_masks] = 1.0
            new_act_distr = Categorical(logits=new_logits)
            new_act_log_probs = (
                new_act_distr.logits
            )  # Shape: (num_candidates, max_seq_len, vocab_size)
            new_act_log_probs = new_act_log_probs[
                candidate_masks
            ]  # Shape: (num_tokens, vocab_size)
            new_act_log_probs = new_act_log_probs[
                torch.nn.functional.one_hot(
                    actions.long(), num_classes=vocab_size
                ).bool()
            ]  # Shape: (num_tokens)

            # Clipped PPO loss term
            term1 = (new_act_log_probs - old_act_log_probs).exp() * advantages
            term2 = (1.0 + epsilon * advantages.sign()) * advantages
            ppo_term = -term1.min(term2).mean()

            # Entropy term
            entropy_term = -entropy_coeff * new_act_distr.entropy().mean()

            # SFT term
            sft_logits[~candidate_masks] = 1.0
            sft_act_distr = Categorical(logits=sft_logits)
            sft_term = (
                sft_coeff
                * torch.distributions.kl_divergence(new_act_distr, sft_act_distr).mean()
            )

            p_loss = (ppo_term + entropy_term + sft_term + sft_term) / gradient_steps
            p_loss.backward()
            total_p_loss += p_loss.item()

            if (optim_step + 1) % gradient_steps == 0:
                p_opt.step()
                p_opt.zero_grad()
            p_net.cpu()

            # Train value network
            v_net.to(device)
            cache = DynamicCache()
            state_values_all = torch.zeros(
                [num_candidates, max_seq_len], dtype=torch.float
            )
            # TODO: Extract producing state values for all tokens into its own function, and precompute prefixes
            for i in range(max_len):
                v_output = v_net(
                    input_ids[:, i : i + 1].to(device),
                    attn_masks[:, : i + 1].to(device),
                    past_key_values=cache,
                )
                state_values_all[:, i] = v_output.logits.squeeze().cpu()
                cache = v_output.past_key_values
            state_values = state_values_all[candidate_masks]  # Shape: (num_tokens)
            diff = state_values - returns
            v_loss = (diff * diff).mean() / gradient_steps
            v_loss.backward()
            total_v_loss += v_loss.item()

            if (optim_step + 1) % gradient_steps == 0:
                v_opt.step()
                v_opt.zero_grad()
            v_net.cpu()

            optim_step += 1

    p_net.cpu()
    v_net.cpu()
    p_net.eval()
    v_net.eval()

    return (total_p_loss, total_v_loss)
