## `recurve` - Train ___s with Hierarchical RL

This repo uses the [ArCHer (Zhou et al., 2024)](https://yifeizhou02.github.io/archer.io/) framework to train ___s hierarchically, by decomposing tasks into a high-level *utternace* component and low-level *token* component. This allows for substantially better sample efficiency and faster policy improvement times.

### Quickstart

Use [Poetry](https://python-poetry.org/) to set up the environment.

```bash
poetry install
eval $(poetry env activate)
```

To track metrics, ensure that you've logged into Weights and Biases. You can now train your models like so:

```bash
python -m recurve.train --dataset datasets/best_lang.yaml
```

Checkpoints, evaluations, and experiment metadata will be stored by default in `./runs`. To evaluate trained models, run the following command:

```bash
python recurve/eval.py --exp-path ./runs/NAME_OF_RUN --dataset datasets/best_lang.yaml --chkpt-label latest
```

### Dataset Format

Datasets are stored under `./datasets`, and have the following format:

```yaml
main_prompt: "You are a helpful assistant." # Prepended to the prompt of each item in the dataset.
items: # List of items in the dataset.
  - prompt: Create a list of 3 compiled programming languages, with a short description of each on one line. Then, tell me the best programming language in the form "The best programming language is ____.". # The context given to the model at the beginning.
    answer: Rust # The gold answer.
score_fn: | # A Python function that takes in the entire context + gold answer, and returns a score. This is run after the episode finishes.
  def score_fn(text: str, answer: str):
    output = text.split("assistant")[-1]
    out_ans = output.split("best programming language")[-1]
    return answer in out_ans
done_fn: | # A Python function that takes in the entire context and returns whether the episode should terminate. This is run after each token is generated.
  def done_fn(text: str):
    output = text.split("assistant")[-1]
    if "best programming language" not in output:
      return False
    out_ans = output.split("best programming language")[-1]
    return "." in out_ans
split_fn: | # A Python function that takes in the current generation and returns whether the generation should stop. This is run after each token is generated.
  def split_fn(output: str):
    return "\n" in output
```

### Why ArCHer?

Since 2022, token-level PPO and its derived algorithms have been the dominant RL algorithm used to train language models. Unfortunately, not only is PPO sample inefficient, but token-level RL methods in general scale poorly to long generations.

An alternative to token-level approaches is *utterance-level* approaches, such as [CHAI (Verma et al., 2022)](https://siddharthverma314.github.io/research/chai-acl-2022/). Rather than using RL to modify token probabilities, a frozen generator model produces a set of utterances, and the algorithm uses an off-policy algorithm to learn to rerank them. In practice, this has been shown to be substantially more sample efficient than PPO. However, the generator cannot be modified to produce better utterances during the training process, it must be trained upfront.

ArCHer combines the best of both worlds. Rather than directly using a token-level method or only using an utterance-level method, it uses the utterance-level method for high level planning, and uses the values produced by it to guide a token-level method. This substantially reduces the length of episodes from the perspective of the token-level generator.

### Differences from the ArCHer paper

- The generator directly uses Q-values from the ranker, rather than advantages.
- Instead of training the generator with REINFORCE, PPO is used instead.
- The target network is updated every `n` steps instead of using Polyak weight averaging.
- For computational efficiency, only decoder networks are used, as opposed to using encoder networks for value prediction (the paper uses RoBERTa).

### Why `recurve`?

All things being equal, a recurve bow will result in more efficient energy transfer into an arrow than a straight bow due to its curved design. In a similar vein, using the ArCHer framework, ___s can be trained more efficiently than standard token-level methods.

### License

This repo is MIT licensed.