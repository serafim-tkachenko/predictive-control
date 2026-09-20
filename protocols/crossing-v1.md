# Crossing baseline v1

Frozen before training on this task. This is a baseline feasibility experiment, not
a claim of novelty or a comparison with JEPA.

## Task and data

Native MiniGrid-SimpleCrossingS9N1-v0, MiniGrid 3.1.0. A 9x9 room contains one wall
with one opening; the start is (1,1), facing east, and the goal is (7,7). The native
324-step horizon, sparse reward, and seven actions remain unchanged. We use the full
symbolic grid, flattened and divided by 10. No pixels, language, bonus reward or
oracle-distance features are supplied.

There are 42 geometries: three vertical and three horizontal wall positions, each
with seven possible openings. Enumerate native environment seeds until all geometry
hashes are found, retain the first seed per geometry, sort by SHA-256, shuffle using
NumPy generator seed 20260920, and assign 28/7/7 to train/development/test. The actual
seed/hash lists are committed in `src/predictive_control/data/crossing-v1.json`.
The hashes cover the full grid encoding, including the fixed goal and walls, but not
the agent marker. Map sets are disjoint by geometry, not merely by random seed.
This is a within-generator split, not evidence of broad out-of-distribution transfer.

Training samples uniformly from the 28 train maps at episode reset. No pretraining
data is collected yet. Future representation comparisons must use this same access
restriction, count their data collection and declare whether encoders are frozen.

## Fixed baseline

PPO from Stable-Baselines3 2.9.0, CPU, one PyTorch thread, 8 environments, separate
64x64 policy/value MLPs. Rollouts 128 steps/environment; batch size 256; 4 epochs;
learning rate 0.0003; gamma 0.99; GAE lambda 0.95; clipping 0.2; entropy weight 0.01.
Train seeds 0, 1, 2 from scratch for 524,288 interactions each. No hyperparameter
search or best-checkpoint selection. Preserve the final checkpoint regardless of score.

Primary evaluation samples from the policy distribution with independent NumPy RNGs,
so evaluation does not consume the training RNG. Greedy evaluation is a secondary
diagnostic, following the documented Empty-room failure analysis. Evaluate before
training and every 131,072 steps on train (one episode/map) and development (four/map),
with action seed 900. Intermediate curves are diagnostics, not precise estimates.

After all three runs finish, evaluate their final checkpoints on train, development
and test maps with 32 sampled episodes/map and action seed 1900. Also evaluate greedy
actions once/map (the environment is deterministic), and a uniform-random policy with
the same sampled episode budgets. Keep all per-episode records. Report success, return,
episode length and per-map success, keeping model seeds separate. Repeated episodes
on seven test maps do not constitute hundreds of independent generalization examples.

The first inspection of test performance is a descriptive assessment of this frozen
baseline. Any subsequent tuning influenced by it must treat this test set as exposed.

## Decision after the run

If median sampled train success is below 80%, investigate baseline learning before
adding predictive objectives. If it clears that feasibility gate, inspect map-level
failures and transfer before defining the representation comparison. This threshold
is an engineering choice, not a statistical significance criterion. We do not extend
the budget or change settings after observing test scores within this experiment.
