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

Three independent PPO training runs, each with 524,288 interactions, followed the
[frozen protocol](../../protocols/crossing-v1.md). The task is native MiniGrid
SimpleCrossingS9N1: one wall with a gap, fixed start/goal, full symbolic observations.
No predictive pretraining, RGB encoder, reward shaping or new RL method is involved.

Geometries are split 28 train / 7 development / 7 test, with no overlap in full-grid
hashes. This tests unseen wall/opening combinations within the same generator, not
arbitrary maze transfer. Training only samples the 28 train maps.

| Run | Train sampled | Dev sampled | Test sampled | Test greedy | Test sampled return |
|---|---:|---:|---:|---:|---:|
"""
        + "\n".join(rows)
        + f"""

Final sampled evaluation uses 32 episodes per map: 896 train, 224 development, 224 test.
Each map has equal weight. The random baseline is shared, not repeated independent
evidence. Greedy evaluation uses one episode per map because the dynamics and policy
are deterministic. Repeated action samples do not increase the number of distinct test
geometries beyond seven; no significance or broad generalization claim is made.

![Learning curves and per-map results](crossing-baseline.png)

Median train success is {median:.1%}: this {decision} the predeclared 80% feasibility
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
"""
    )
    (args.evidence / "RESULTS.md").write_text(report, encoding="utf-8")


if __name__ == "__main__":
    main()
