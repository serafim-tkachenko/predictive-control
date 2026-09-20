"""Regenerate the calibration table and plot from committed episode records."""

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("evidence", type=Path)
    args = parser.parse_args()
    rows = []
    fig, ax = plt.subplots(figsize=(8, 4.5), layout="constrained")
    for path in sorted(args.evidence.glob("*/final.json")):
        config = json.loads((path.parent / "config.json").read_text())
        result = json.loads(path.read_text())
        development = json.loads((path.parent / "development.json").read_text())
        trained, random = result["trained"], result["random"]
        rows.append(
            f"| {config['seed']} | {result['steps']:,} | "
            f"{trained['success_rate']:.1%} | {trained['mean_return']:.3f} | "
            f"{random['success_rate']:.1%} | {random['mean_return']:.3f} |"
        )
        if result["steps"] > 32768:
            ax.plot(
                [r["steps"] for r in development],
                [r["success_rate"] for r in development],
                label=f"seed {config['seed']}",
            )
    if not rows:
        raise SystemExit("No final.json files found")
    ax.set(
        xlabel="Training interactions",
        ylabel="Development success rate",
        ylim=(-0.02, 1.02),
        title="PPO calibration: full symbolic MiniGrid Empty",
    )
    ax.grid(alpha=0.2)
    ax.legend()
    fig.savefig(args.evidence / "learning-curve.png", dpi=150)
    plt.close(fig)
    report = (
        """# First PPO calibration

This checks that the training loop learns in a small environment. It does not test JEPA,
world-model planning, transfer, or a novel method.

| Training seed | Interactions | PPO success | PPO return | Random success | Random return |
|---|---:|---:|---:|---:|---:|
"""
        + "\n".join(rows)
        + """

Each final score uses 32 episodes with seeds 20,000–20,031 and the final checkpoint,
with deterministic PPO actions. Random actions use one fixed RNG stream, shared across
runs: its repeated score is one comparator, not independent baseline replications.
The layout and goal are fixed; starts vary. Different evaluation seeds do not guarantee
different states from training. These results establish neither generalization nor
statistical significance. No confidence interval is inferred from pooled training seeds.

The 32,768-interaction seed-0 pilot was inspected first. Its poor result motivated
increasing the budget to 262,144 and running seeds 0, 1, 2 from scratch with unchanged
hyperparameters. This was an adaptive calibration decision, not a preregistered test.
The small run and longer seed-0 run are not independent replications.

![Development curves](learning-curve.png)

Development evaluation uses 32 fixed episodes with seeds 10,000–10,031. Periodic
callbacks evaluate at rollout boundaries before the following PPO update; the final
table evaluates the saved checkpoint after all updates. Training returns use sampled
actions and should not be equated with deterministic evaluation performance.

Per-run folders contain raw evaluation episodes, PPO diagnostics, configuration,
source hashes and checkpoint hashes. Checkpoints remain in the local `runs/` folders;
they can be regenerated with the README commands. No pretrained JEPA weights were used.

Next: inspect failed trajectories and establish adequate baseline coverage before moving
to a larger room or a predictive representation. Do not tune on these final episodes
and continue to call them an untouched test set.
"""
    )
    (args.evidence / "RESULTS.md").write_text(report, encoding="utf-8")


if __name__ == "__main__":
    main()
