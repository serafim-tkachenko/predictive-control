# Predictive control

Does learning to predict future states help an RL agent learn and generalize?
This repository starts with PPO navigation baselines in MiniGrid. Predictive
representation learning is the next step; there is no JEPA implementation yet.

## Current results

The small Empty room is solved reliably when actions are sampled from the trained
policy. Choosing the most likely action can trap the same policy in a loop.
[State audit](evidence/empty-audit-20260920/RESULTS.md).

Crossing adds a wall with a movable opening. Its 42 layouts are split by geometry:
28 for training, 7 for development, 7 for testing. After 524,288 interactions per seed:

| PPO seed | Train success | Test success |
|---|---:|---:|
| 0 | 48.7% | 36.2% |
| 1 | 66.5% | 55.8% |
| 2 | 46.9% | 32.6% |
| Random actions | 5.0% | 6.25% |

These scores use sampled actions. Learning on the training maps is still weak, so
our next experiment will examine the baseline before adding a prediction objective.
The current input is a flattened symbolic grid, not a rendered image.

[Crossing results](evidence/crossing-v1/RESULTS.md) ·
[Protocol](protocols/crossing-v1.md) ·
[Layouts](protocols/crossing-layouts.png) ·
[Working notes and next session](docs/notes/2026-09-20.md)

## Run

Python 3.11, with [uv](https://docs.astral.sh/uv/):

```bash
uv venv --python 3.11
uv pip install -r requirements-cpu.txt --extra-index-url https://download.pytorch.org/whl/cpu
uv pip install --no-deps -e .
.venv/bin/python -m predictive_control.train_crossing --seed 0 --output runs/crossing-v1-seed0
```

This runs 524,288 interactions on CPU. The initial machine took about 106 seconds,
including development evaluations. Choose a new output directory for each run.
[Reproduction commands and saved files](docs/reproducing.md).

## Research context

[SPR](https://arxiv.org/abs/2007.05929),
[representation pretraining for RL](https://arxiv.org/abs/2106.04799), and
[TACO](https://arxiv.org/abs/2306.13229) already study predictive representations for
control. The open question here needs to be more specific than adding an auxiliary
prediction loss. For now, the work is baseline development and evaluation.

## Checks

```bash
.venv/bin/pytest -q
.venv/bin/ruff check .
.venv/bin/ruff format --check .
```

CI checks both training entry points. Tests also cover environment transitions,
layout separation, reachability and evaluation reproducibility.
