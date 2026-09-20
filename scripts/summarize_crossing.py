"""Generate the crossing baseline report and its map-level evaluation figure."""

import argparse
import json
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("evidence", type=Path)
    args = parser.parse_args()
    rows, training = [], []
    fig, axes = plt.subplots(1, 2, figsize=(11, 4), layout="constrained")
    test_map_rows = []
    names = [f"crossing-v1-seed{s}" for s in range(3)] + ["random"]
    for name in names:
        values = []
        for partition in ["train", "development", "test"]:
            data = json.loads((args.evidence / name / f"evaluation-{partition}.json").read_text())
            sampled = data["sampled"]
            values.append(f"{sampled['success_rate']:.1%}")
            if partition == "train" and name != "random":
                training.append(sampled["success_rate"])
            if partition == "test":
                map_ids = sorted({e["geometry_sha256"] for e in sampled["episodes"]})
                test_map_rows.append(
                    [
                        np.mean(
                            [
                                e["success"]
                                for e in sampled["episodes"]
                                if e["geometry_sha256"] == key
                            ]
                        )
                        for key in map_ids
                    ]
                )
                greedy = f"{data['greedy']['success_rate']:.1%}" if name != "random" else "—"
                mean_return = sampled["mean_return"]
        rows.append(f"| {name} | {' | '.join(values)} | {greedy} | {mean_return:.3f} |")
        if name != "random":
            curve = json.loads((args.evidence / name / "development.json").read_text())
            axes[0].plot(
                [r["steps"] for r in curve],
                [r["development"]["success_rate"] for r in curve],
                label=name[-5:],
            )
    axes[0].set(
        title="Development during training (28 episodes)",
        xlabel="Training interactions",
        ylabel="Sampled success",
        ylim=(-0.02, 1.02),
    )
    axes[0].legend()
    axes[0].grid(alpha=0.2)
    matrix = np.array(test_map_rows)
    im = axes[1].imshow(matrix, vmin=0, vmax=1, cmap="viridis", aspect="auto")
    axes[1].set(
        title="Final test success by geometry (32 episodes/map)",
        xticks=range(7),
        xticklabels=[key[:5] for key in map_ids],
        yticks=range(4),
        yticklabels=["seed 0", "seed 1", "seed 2", "random"],
        xlabel="Geometry SHA-256 prefix",
    )
    for i in range(4):
        for j in range(7):
            axes[1].text(
                j,
                i,
                f"{matrix[i, j]:.0%}",
                ha="center",
                va="center",
                color="black" if matrix[i, j] > 0.55 else "white",
                fontsize=8,
            )
    fig.colorbar(im, ax=axes[1], label="Success probability estimate")
    fig.savefig(args.evidence / "crossing-baseline.png", dpi=150)
    plt.close(fig)
    median = float(np.median(training))
    decision = "cleared" if median >= 0.8 else "did not clear"
    report = (
        """# Crossing baseline v1

PPO learns something useful, but still fails on many training maps. The current
baseline needs work before comparing representation-learning objectives.

Three seeds, 524,288 interactions each, using the
[settings fixed before training](../../protocols/crossing-v1.md). The environment is
MiniGrid SimpleCrossingS9N1: a wall with a gap, fixed start and goal. The MLP receives
the full symbolic grid. The 42 geometries are split 28/7/7 into train/development/test.

| Run | Train sampled | Dev sampled | Test sampled | Test greedy | Test sampled return |
|---|---:|---:|---:|---:|---:|
"""
        + "\n".join(rows)
        + f"""

Scores use the final checkpoint and 32 sampled episodes per map, with equal weight
per map. Greedy execution needs only one episode per map here. There are seven test
geometries; repeated action samples do not make the test set larger. The random row
comes from one shared baseline evaluation.

![Learning curves and per-map results](crossing-baseline.png)

Median train success is {median:.1%}. This {decision} the 80% threshold chosen before
training for deciding whether to proceed with a representation comparison. Next,
inspect failures and test the observation encoding; its role is still a hypothesis.

All three checkpoints were saved before test evaluation. The budget and settings
were unchanged. These test maps have now been inspected and should be treated as
exposed in follow-up work.

The curves use small intermediate evaluations before PPO updates. The table uses
the final saved models and a fresh action seed. Evaluation has its own RNGs. The
folders beside this report contain per-episode scores, configurations, source and
checkpoint hashes, and training diagnostics. Model files remain in local `runs/`.
"""
    )
    (args.evidence / "RESULTS.md").write_text(report, encoding="utf-8")


if __name__ == "__main__":
    main()
