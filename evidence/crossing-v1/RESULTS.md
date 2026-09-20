# Crossing baseline v1

Three independent PPO training runs, each with 524,288 interactions, followed the
[frozen protocol](../../protocols/crossing-v1.md). The task is native MiniGrid
SimpleCrossingS9N1: one wall with a gap, fixed start/goal, full symbolic observations.
No predictive pretraining, RGB encoder, reward shaping or new RL method is involved.

Geometries are split 28 train / 7 development / 7 test, with no overlap in full-grid
hashes. This tests unseen wall/opening combinations within the same generator, not
arbitrary maze transfer. Training only samples the 28 train maps.

| Run | Train sampled | Dev sampled | Test sampled | Test greedy | Test sampled return |
|---|---:|---:|---:|---:|---:|
| crossing-v1-seed0 | 48.7% | 28.1% | 36.2% | 0.0% | 0.208 |
| crossing-v1-seed1 | 66.5% | 52.7% | 55.8% | 0.0% | 0.319 |
| crossing-v1-seed2 | 46.9% | 33.9% | 32.6% | 0.0% | 0.179 |
| random | 5.0% | 4.9% | 6.2% | — | 0.021 |

Final sampled evaluation uses 32 episodes per map: 896 train, 224 development, 224 test.
Each map has equal weight. The random baseline is shared, not repeated independent
evidence. Greedy evaluation uses one episode per map because the dynamics and policy
are deterministic. Repeated action samples do not increase the number of distinct test
geometries beyond seven; no significance or broad generalization claim is made.

![Learning curves and per-map results](crossing-baseline.png)

Median train success is 48.7%: this did not clear the predeclared 80% feasibility
gate. That gate determines whether this baseline is ready for a representation
comparison; it is not a statistical test or a literature benchmark threshold.

All three final checkpoints were fixed before test evaluation. No training budget,
hyperparameter or checkpoint was selected using these test results. The test set is
now exposed: further tuning informed by this report must be labeled accordingly.

Intermediate evaluations are small diagnostics, made at rollout boundaries before the
next PPO update. The final table uses saved checkpoints after all updates and a fresh
action-sampling seed. Evaluation uses independent local RNGs, so it does not perturb
training. Raw episodes, curves, configuration, training diagnostics, checkpoint hashes
and split hashes are retained. Checkpoints remain in the local `runs/` directories.
