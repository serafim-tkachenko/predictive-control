# Reproducing the experiments

Run these commands from the repository root after the [README setup](../README.md#run).
The commands assume a fresh clone; existing run directories are never overwritten.
The runs below use CPU and one PyTorch thread.

## Empty room

The short first pilot used 32,768 interactions:

```bash
.venv/bin/python -m predictive_control.train --seed 0 --steps 32768 --output runs/empty-seed0
```

The three longer runs used 262,144 interactions each:

```bash
for seed in 0 1 2; do
  .venv/bin/python -m predictive_control.train --seed "$seed" --steps 262144 --output "runs/empty-262k-seed$seed"
done
```

This is `MiniGrid-Empty-Random-5x5-v0`. PPO sees the full symbolic grid, including
position and heading, flattened and divided by 10. The native seven actions and
reward are unchanged. Policy and value networks each have two 64-unit hidden layers.

The original evaluation takes the most likely action on 32 seeded episodes. Different
seeds can produce the same start. The later audit covers all 32 position/heading
combinations and evaluates both sampled and greedy execution:

```bash
.venv/bin/python -m predictive_control.audit runs/empty-262k-seed0/model.zip runs/empty-262k-seed1/model.zip runs/empty-262k-seed2/model.zip --output runs/state-audit
.venv/bin/python scripts/validate_audit.py --audit runs/state-audit
.venv/bin/python scripts/summarize_audit.py runs/state-audit
```

The validation command runs 1,024 episodes per trained policy and 1,024 with random
actions. The exact calculation in the audit only applies to the deterministic Empty
room and stationary policies.

## Crossing

The [protocol](../protocols/crossing-v1.md) fixes the settings and map split. The split
file is [crossing-v1.json](../src/predictive_control/data/crossing-v1.json).

```bash
for seed in 0 1 2; do
  .venv/bin/python -m predictive_control.train_crossing --seed "$seed" --output "runs/crossing-v1-seed$seed"
done
.venv/bin/python scripts/evaluate_crossing.py --output runs/crossing-evaluation
.venv/bin/python scripts/summarize_crossing.py runs/crossing-evaluation
```

All three runs must finish before final evaluation. The evaluator checks checkpoint
and split hashes. It uses 32 sampled episodes per map, plus one greedy episode per
map. Action sampling uses local RNGs and does not consume training's RNG state.

The seven test layouts have now been inspected. They remain useful for reproduction,
but further tuning on their results cannot be presented as a fresh test.

## Files

| File | Contents |
|---|---|
| `config.json` | Environment, PPO settings, seed and observation description |
| `manifest.json` | Package versions, source hashes and Git revision |
| `development.json` | Intermediate scores and individual episodes |
| `progress.csv`, `monitor/` | PPO diagnostics and training episode returns |
| `model.zip` | Final checkpoint; load with `stable_baselines3.PPO.load` |
| `final.json` | Empty-room final evaluation |
| `completed.json` | Crossing training duration and checkpoint hash |
| `evaluation-*.json` | Crossing final scores and individual episodes |

Checkpoints and monitor logs stay in local `runs/`. The `evidence/` directory contains
the recorded scores, configurations, training diagnostics and hashes used in reports.

To regenerate the committed reports without training:

```bash
.venv/bin/python scripts/summarize.py evidence/empty-20260920
.venv/bin/python scripts/summarize_audit.py evidence/empty-audit-20260920
.venv/bin/python scripts/summarize_crossing.py evidence/crossing-v1
```
