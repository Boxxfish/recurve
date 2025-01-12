"""
A replay buffer for use with off policy algorithms.
"""

from typing import List, Optional, Tuple

import torch


class ReplayBuffer:
    """
    Stores transitions and generates mini batches.
    """

    def __init__(
        self,
        max_seq_len: int,
        num_candidates: int,
        capacity: int,
        vocab_size: int,
        sample_steps: int,
    ):
        input_ids_shape = torch.Size([capacity, num_candidates, max_seq_len])
        attn_mask_shape = torch.Size([capacity, num_candidates, max_seq_len])
        logits_shape = torch.Size(
            [sample_steps, num_candidates, max_seq_len, vocab_size]
        )
        action_shape = torch.Size([capacity, 1])
        self.capacity = capacity
        self.next = 0
        d = torch.device("cpu")
        self.input_ids = torch.zeros(
            input_ids_shape, dtype=torch.int, device=d, requires_grad=False
        )
        self.attn_masks = torch.zeros(
            attn_mask_shape, dtype=torch.bool, device=d, requires_grad=False
        )
        self.next_input_ids = torch.zeros(
            input_ids_shape, dtype=torch.int, device=d, requires_grad=False
        )
        self.next_attn_masks = torch.zeros(
            attn_mask_shape, dtype=torch.bool, device=d, requires_grad=False
        )
        self.logits = torch.zeros(
            logits_shape, dtype=torch.float, device=d, requires_grad=False
        )
        self.sft_logits = torch.zeros(
            logits_shape, dtype=torch.float, device=d, requires_grad=False
        )
        self.logit_buffer_idxs = {}
        self.next_logit = 0
        self.candidate_masks = torch.zeros(
            attn_mask_shape, dtype=torch.bool, device=d, requires_grad=False
        )
        self.next_candidate_masks = torch.zeros(
            attn_mask_shape, dtype=torch.bool, device=d, requires_grad=False
        )
        self.candidate_rewards = torch.zeros(
            [capacity, num_candidates], dtype=torch.float, device=d, requires_grad=False
        )
        self.actions = torch.zeros(
            action_shape, dtype=torch.int64, device=d, requires_grad=False
        )
        self.rewards = torch.zeros(
            [capacity], dtype=torch.float, device=d, requires_grad=False
        )
        # Technically this is the "terminated" flag
        self.dones = torch.zeros(
            [capacity], dtype=torch.float, device=d, requires_grad=False
        )
        self.filled = False

    def insert_step(
        self,
        input_ids: torch.Tensor,
        attn_masks: torch.Tensor,
        next_input_ids: torch.Tensor,
        next_attn_masks: torch.Tensor,
        actions: torch.Tensor,
        rewards: List[float],
        dones: List[bool],
        logits: Optional[torch.Tensor],
        candidate_masks: torch.Tensor,
        next_candidate_masks: torch.Tensor,
        candidate_rewards: torch.Tensor,
    ):
        """
        Inserts a transition from each environment into the buffer. Make sure
        more data than steps aren't inserted.
        """
        batch_size = len(dones)
        assert batch_size == 1
        d = torch.device("cpu")
        with torch.no_grad():
            indices = torch.arange(
                self.next,
                (self.next + batch_size),
            ).remainder(self.capacity)
            self.input_ids.index_copy_(0, indices, input_ids)
            self.attn_masks.index_copy_(0, indices, attn_masks)
            self.next_input_ids.index_copy_(0, indices, next_input_ids)
            self.next_attn_masks.index_copy_(0, indices, next_attn_masks)
            self.candidate_masks.index_copy_(0, indices, candidate_masks)
            self.next_candidate_masks.index_copy_(0, indices, next_candidate_masks)
            self.actions.index_copy_(0, indices, actions)
            self.candidate_rewards.index_copy_(0, indices, candidate_rewards)
            self.rewards.index_copy_(
                0, indices, torch.tensor(rewards, dtype=torch.float, device=d)
            )
            self.dones.index_copy_(
                0, indices, torch.tensor(dones, dtype=torch.float, device=d)
            )

            # Logits are updated in their own way due to being cleared each training iteration
            if logits is not None:
                logit_indices = torch.arange(
                    self.next_logit,
                    (self.next_logit + batch_size),
                )
                self.logits.index_copy_(0, logit_indices, logits)
                self.logit_buffer_idxs[self.next] = self.next_logit
                self.next_logit += 1

        self.next = (self.next + batch_size) % self.capacity
        if self.next == 0:
            self.filled = True

    def reset_logits(self):
        """Resets policy and SFT logits, which are used for on-policy training."""
        self.next_logit = 0
        self.logit_buffer_idxs = {}

    def sample(self, batch_size: int) -> Tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
    ]:
        """
        Generates minibatches of experience.
        """
        with torch.no_grad():
            indices = torch.randint(
                self.capacity,
                [batch_size],
                dtype=torch.int,
            )
            rand_input_ids = self.input_ids.index_select(0, indices)
            rand_attn_masks = self.attn_masks.index_select(0, indices)
            rand_next_input_ids = self.next_input_ids.index_select(0, indices)
            rand_next_attn_masks = self.next_attn_masks.index_select(0, indices)
            rand_actions = self.actions.index_select(0, indices)
            rand_rewards = self.rewards.index_select(0, indices)
            rand_dones = self.dones.index_select(0, indices)
            rand_candidate_masks = self.candidate_masks.index_select(0, indices)
            rand_next_candidate_masks = self.next_candidate_masks.index_select(0, indices)
            return (
                rand_input_ids,
                rand_attn_masks,
                rand_next_input_ids,
                rand_next_attn_masks,
                rand_actions,
                rand_rewards,
                rand_dones,
                rand_candidate_masks,
                rand_next_candidate_masks,
            )
