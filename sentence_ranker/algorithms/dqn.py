from typing import Tuple

import torch
from torch import nn

from .replay_buffer import ReplayBuffer

INF = 10e8


def train_dqn(
    q_net: nn.Module,
    q_net_target: nn.Module,
    q_opt: torch.optim.Optimizer,
    buffer: ReplayBuffer,
    device: torch.device,
    train_iters: int,
    train_batch_size: int,
    discount: float,
    gradient_steps: int,
) -> float:
    """
    Performs the DQN training loop.
    Returns the total Q loss.
    """
    total_q_loss = 0.0
    q_net.train()
    q_net.cpu()
    q_net_target.cpu()

    q_opt.zero_grad()

    total_q_loss = 0.0
    for i in range(train_iters):
        prev_input_ids, prev_attn_masks, input_ids, attn_masks, actions, rewards, dones, _, _ = buffer.sample(
            train_batch_size
        )
        num_candidates = prev_input_ids.shape[1]

        # Move batch to device if applicable
        prev_input_ids = prev_input_ids.to(device=device)
        prev_attn_masks = prev_attn_masks.to(device=device)
        input_ids = input_ids.to(device=device)
        attn_masks = attn_masks.to(device=device)
        actions = actions.to(device=device)
        rewards = rewards.to(device=device)
        dones = dones.to(device=device)

        # Train q network
        with torch.no_grad():
            q_net.to(device)
            next_actions = q_net.forward(input_ids.flatten(0, 1), attn_masks.flatten(0, 1)).logits.reshape(train_batch_size, num_candidates).argmax(1).squeeze(0)
            q_net.cpu()

            q_net_target.to(device)
            q_target = rewards.unsqueeze(1) + discount * q_net_target.forward(
                input_ids.flatten(0, 1), attn_masks.flatten(0, 1)
            ).logits.reshape(train_batch_size, num_candidates).detach().gather(1, next_actions.unsqueeze(1)) * (1.0 - dones.unsqueeze(1))
            q_net_target.cpu()
        
        q_net.to(device)
        diff = q_net.forward(prev_input_ids.flatten(0, 1), prev_attn_masks.flatten(0, 1)).logits.reshape(train_batch_size, num_candidates).gather(1, actions) - q_target
        q_loss = (diff * diff).mean()
        q_loss.backward()

        if (i + 1) % gradient_steps == 0:
            q_opt.step()
            q_opt.zero_grad()
        
        total_q_loss += q_loss.item()
        q_net.cpu()

    q_net.eval()
    return total_q_loss
