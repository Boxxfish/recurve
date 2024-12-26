from pydantic import BaseModel
from transformers.models.llama import LlamaForSequenceClassification, LlamaForCausalLM

from sentence_ranker.algorithms.replay_buffer import ReplayBuffer
from sentence_ranker.algorithms.rollout_buffer import RolloutBuffer
from sentence_ranker.utils import create_directory, parse_args


class Args(BaseModel):
    exp_base_dir: str = "./runs"

    # Ranker settings
    ranker_base: str = "meta-llama/Llama-3.2-1B"

    # Generator settings
    generator_base: str = "meta-llama/Llama-3.2-1B-Instruct"
    max_input_len: int = 256    # Max length of an input, including the prompt + special tokens
    num_envs: int = 8           # Number of parallel generations during sampling
    num_steps: int = 512        # How many steps to take during sampling


class ExpMeta(BaseModel):
    args: Args


class GeneratorTrainer:
    def __init__(self, args: Args):
        self.p_net = LlamaForCausalLM.from_pretrained(args.ranker_base)
        self.v_net = LlamaForSequenceClassification.from_pretrained(args.ranker_base)
        self.buffer = RolloutBuffer(
            args.max_input_len,
            int(self.p_net.vocab_size),
            args.num_envs,
            args.num_steps,
        )


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

    # Set up trainers
    gen_trainer = GeneratorTrainer()

if __name__ == "__main__":
    main()
