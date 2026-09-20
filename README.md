# Predictive control

Experiments on learning representations for reinforcement learning. The question is when predicting future features helps a policy learn and transfer, and when prediction quality is a misleading proxy for control.

The first experiment is deliberately small: Stable-Baselines3 PPO on MiniGrid Empty with randomized starts. It establishes a working RL training loop before adding a predictive objective. No JEPA model or new algorithm is implemented yet.

## Run the baseline

Python 3.11; the initial configuration uses CPU and one PyTorch thread. With [uv](https://docs.astral.sh/uv/):

```bash
uv venv --python 3.11
uv pip install -r requirements-cpu.txt --extra-index-url https://download.pytorch.org/whl/cpu
uv pip install --no-deps -e .
.venv/bin/python -m predictive_control.train --seed 0 --steps 32768 --output runs/empty-seed0
```

Choose a fresh output directory for each run; existing results are never overwritten. Repeat with seeds 1 and 2 to inspect training variability. The environment is `MiniGrid-Empty-Random-5x5-v0`, with the native seven actions and reward. Observations are the **full symbolic grid**, flattened and divided by 10, including agent position and heading. They are not RGB images. The MLP has separate policy/value networks with two 64-unit hidden layers.

Each run writes:

- `config.json` and `manifest.json`: settings, software versions, commit and source hashes;
- `development.json`: raw episodes before training and every 8,192 interactions;
- `progress.csv` and `monitor/`: PPO diagnostics and training episode returns;
- `final.json`: final-checkpoint evaluation and a uniform random-action baseline;
- `model.zip`: the final checkpoint.

Evaluation uses a separate environment, deterministic policy actions, and fixed seeds (10,000 onward for development, 20,000 onward for final evaluation). These are new random seeds, **not unseen layouts or necessarily unseen states**: the tiny room has a finite set of starts. Success means reaching the goal, not merely surviving until the time limit. No checkpoint selection uses the final evaluation. The output checkpoint can be loaded with `stable_baselines3.PPO.load`.

## What comes after calibration

Choose a task with meaningful variation, freeze its evaluation protocol, then compare RL alone against representation pretraining with and without future prediction. Count pretraining data collection as environment interaction and report compute separately. Check representation collapse, transfer and policy performance rather than relying on prediction loss alone.

The general idea has substantial prior art: [SPR](https://arxiv.org/abs/2007.05929), [representation pretraining for RL](https://arxiv.org/abs/2106.04799), and [TACO](https://arxiv.org/abs/2306.13229). An auxiliary prediction loss is not by itself a new contribution, nor does it make PPO model-based RL. The next research question must be narrower than this repository's working theme.

## Checks

```bash
.venv/bin/pytest -q
.venv/bin/ruff check .
.venv/bin/ruff format --check .
```

Tests cover observable heading, goal termination, time-limit truncation, and reproducible random-policy evaluation. CI also runs a short training job; learning quality is assessed separately in the experiment results.
