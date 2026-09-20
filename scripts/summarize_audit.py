"""Summarize the exhaustive audit; all displayed values come from recorded evidence."""

import argparse
import json
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("audit", type=Path)
    args = parser.parse_args()
    summary = json.loads((args.audit / "summary.json").read_text())
    validation = json.loads((args.audit / "rollout-validation.json").read_text())
    rows, greedy, sampled = [], [], []
    for row in summary["runs"]:
        raw = json.loads((args.audit / f"{row['run']}.json").read_text())
        greedy.append(row["greedy_success"])
        sampled.append(row["sampled_success"])
        rows.append(
            f"| {row['run']} | {100 * greedy[-1]:.3f}% | "
            f"{100 * sampled[-1]:.5f}% | {row['cycle_failures']} | "
            f"{np.mean(raw['sampled']['length']):.2f} |"
        )
    random_success = np.mean(summary["random"]["success"])
    fig, ax = plt.subplots(figsize=(7, 4), layout="constrained")
    positions = np.arange(len(rows))
    ax.bar(positions - 0.18, greedy, 0.36, label="Most likely action")
    ax.bar(positions + 0.18, sampled, 0.36, label="Sample from policy")
    ax.axhline(random_success, color="black", linestyle="--", label="Uniform random actions")
    ax.set(
        xticks=positions,
        xticklabels=["seed 0", "seed 1", "seed 2"],
        ylim=(0, 1.2),
        ylabel="Exact success probability within 100 steps",
        title="Same checkpoints, different execution rules",
    )
    ax.legend(loc="upper center", fontsize=8)
    fig.savefig(args.audit / "execution-rules.png", dpi=150)
    plt.close(fig)
    example = json.loads((args.audit / "empty-262k-seed2.json").read_text())
    index = summary["states"].index([3, 1, 0])
    forward, right = example["probabilities"][index][2], example["probabilities"][index][1]
    validation_rows = []
    for row in validation["runs"]:
        validation_rows.append(
            f"| {row['run']} | {row['observed']['success']:.3%} | "
            f"{row['exact']['return']:.4f} | {row['observed']['return']:.4f} |"
        )
    report = (
        """# Why some PPO evaluations failed

The saved policies perform well when actions are sampled from their learned distribution.
Taking only the most likely action creates deterministic cycles for some starts.
This is a behavioral diagnosis in one tiny room, not a new RL result or evidence about JEPA.

## Complete state coverage

The room has 8 non-goal starting cells and 4 headings: 32 valid initial states.
All 32 have distinct observations. We obtained all seven one-step transitions for each
state from the actual environment. For each checkpoint we extracted its categorical
action probabilities, then propagated probability mass for the native 100-step horizon.
This integrates action sampling exactly (up to floating-point error), rather than
estimating it from a small rollout sample. It assumes this deterministic room and a
stationary feed-forward policy; it is not an evaluator for arbitrary environments.

| Run | Greedy success | Sampled success | Greedy cycle failures / 32 | Sampled mean steps |
|---|---:|---:|---:|---:|
"""
        + "\n".join(rows)
        + f"""

Uniform random actions achieve {100 * random_success:.3f}% exact success under the same
uniform weighting over starts. The old 65.6% random result was a finite rollout estimate
on a different weighted set of starts; it was not an exact baseline.

The previous development seeds covered {summary["development_coverage"]["unique_states"]}/32
distinct states, and final seeds covered {summary["final_coverage"]["unique_states"]}/32.
That explains why some scores change under complete coverage. This is a diagnostic
re-evaluation of previously inspected checkpoints, not a new held-out generalization test.

![Execution rules](execution-rules.png)

## A concrete failure

For seed 2, starting at (1,1) facing east, the greedy policy moves to (2,1), then (3,1),
then keeps trying to move into the east wall. At (3,1,east), forward has probability
{forward:.3f}, while turning right has probability {right:.3f}. Taking argmax always repeats
the wall collision. Sampling retains the chance of turning, so the state is not a trap.
All greedy failures in these checkpoints enter repeated position/heading cycles.

## Independent rollout check

We also executed {validation["repeats_per_state"]} stochastic episodes per initial state
(1,024 per policy) using SB3's actual action sampler and the real environment, without
the probability-transition calculation. Seed 777 fixes action sampling. All three
trained policies succeeded in all these sampled episodes; this does not prove perfect
success, and the exact calculation above retains their small failure probabilities.

| Policy | Observed success | Exact expected return | Observed mean return |
|---|---:|---:|---:|
"""
        + "\n".join(validation_rows)
        + """

## Decision

The sampled policy is an adequate calibration baseline on this room. Increasing the
training budget just to repair greedy execution is not the next priority. Keep both
execution rules visible in subsequent experiments and choose the primary rule before
comparing methods. PPO was trained with sampled actions; greedy execution is a separate
deployment choice, not a bug in the old evaluation.

This does not establish optimal paths, larger-room performance, visual learning, or
generalization. Before adding predictive pretraining, the next baseline needs a task
with meaningful variation and a fixed data/evaluation protocol. This small solved room
is useful for checks, but has little headroom for a representation-learning claim.

No policy was retrained in this audit. Existing checkpoints and their hashes are retained.
Tests cover closed-form absorption, distinct observations, shortest-path reachability,
and detection of a wall-collision cycle. Raw per-state probabilities, trajectories and
independent rollout episodes are included alongside this report.
"""
    )
    (args.audit / "RESULTS.md").write_text(report, encoding="utf-8")


if __name__ == "__main__":
    main()
