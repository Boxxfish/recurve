import copy
from typing import Tuple

import torch
from torch import nn
from torch.distributions import Categorical
from tqdm import tqdm

from sentence_ranker.utils import get_llm_logits

from .rollout_buffer import RolloutBuffer


def train_ppo(
    p_net: nn.Module,
    v_net: nn.Module,
    p_opt: torch.optim.Optimizer,
    v_opt: torch.optim.Optimizer,
    buffer: RolloutBuffer,
    device: torch.device,
    train_iters: int,
    train_batch_size: int,
    discount: float,
    lambda_: float,
    epsilon: float,
    sft_coeff: float,
    gradient_steps: int = 1,
    entropy_coeff: float = 0.0,
) -> Tuple[float, float]:
    """
    Performs the PPO training loop. Returns a tuple of total policy loss and
    total value loss.

    Args:
        gradient_steps: Number of batches to step through before before
        adjusting weights.
        use_masks: If True, masks are passed to the model.
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

    for _ in tqdm(range(train_iters), position=1):
        batches = buffer.samples(train_batch_size, discount, lambda_, v_net_frozen.to(device), device)
        v_net_frozen.cpu()
        for (
            i,
            (prev_input_ids, prev_attn_masks, actions, logits, sft_logits, returns, advantages, action_masks),
        ) in enumerate(batches):
            # Move batch to device if applicable
            prev_input_ids = prev_input_ids.to(device=device)
            prev_attn_masks = prev_attn_masks.to(device=device)
            actions = actions.to(device=device)
            logits = logits.to(device=device)
            sft_logits = sft_logits.to(device=device)
            returns = returns.to(device=device)
            advantages = advantages.to(device=device)

            # Train policy network
            p_net.to(device)
            with torch.no_grad():
                old_act_log_probs = Categorical(logits=logits).log_prob(
                    actions.squeeze()
                )
            new_logits, _ = get_llm_logits(p_net, prev_input_ids, prev_attn_masks)
            new_act_distr = Categorical(logits=new_logits)
            new_act_log_probs = new_act_distr.log_prob(actions.squeeze())
            
            # Clipped PPO loss term
            term1 = (new_act_log_probs - old_act_log_probs).exp() * advantages.squeeze()
            term2 = (1.0 + epsilon * advantages.squeeze().sign()) * advantages.squeeze()
            ppo_term = -term1.min(term2).mean()

            # Entropy term
            entropy_term = -entropy_coeff * new_act_distr.entropy().mean()

            # SFT term
            sft_act_distr = Categorical(logits=sft_logits)
            sft_term = sft_coeff * torch.distributions.kl_divergence(new_act_distr, sft_act_distr).mean()
            
            p_loss = (
                ppo_term + entropy_term + sft_term + sft_term
            ) / gradient_steps
            p_loss.backward()
            total_p_loss += p_loss.item()
            
            if (i + 1) % gradient_steps == 0:
                p_opt.step()
                p_opt.zero_grad()
            p_net.cpu()

            # Train value network
            v_net.to(device)
            diff = v_net(prev_input_ids, prev_attn_masks).logits.squeeze() - returns
            v_loss = (diff * diff).mean() / gradient_steps
            v_loss.backward()
            total_v_loss += v_loss.item()

            if (i + 1) % gradient_steps == 0:
                v_opt.step()
                v_opt.zero_grad()
            v_net.cpu()

    p_net.cpu()
    v_net.cpu()
    p_net.eval()
    v_net.eval()

    return (total_p_loss, total_v_loss)
