"""Evaluate frozen final checkpoints only after all protocol training runs finish."""

import argparse
import hashlib
import json
import shutil
from pathlib import Path

import torch
from stable_baselines3 import PPO

from predictive_control.crossing import SPLIT_PATH, load_split
from predictive_control.train import write_json
from predictive_control.train_crossing import evaluate_layouts


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--runs", type=Path, default=Path("runs"))
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    sources = [args.runs / f"crossing-v1-seed{seed}" for seed in range(3)]
    split_hash = hashlib.sha256(SPLIT_PATH.read_bytes()).hexdigest()
    for source in sources:
        completed = json.loads((source / "completed.json").read_text())
        manifest = json.loads((source / "manifest.json").read_text())
        assert completed["steps"] == 524288, "Protocol requires all complete final checkpoints"
        assert manifest["split_sha256"] == split_hash
        assert (
            completed["checkpoint_sha256"]
            == hashlib.sha256((source / "model.zip").read_bytes()).hexdigest()
        )
    args.output.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(1)
    split = load_split()
    for source in sources + [None]:
        name = source.name if source else "random"
        output = args.output / name
        output.mkdir()
        model = PPO.load(source / "model.zip", device="cpu") if source else None
        if source:
            for file in list(source.glob("*.json")) + [source / "progress.csv"]:
                shutil.copy2(file, output / file.name)
        for partition in ["train", "development", "test"]:
            result = {
                "sampled": evaluate_layouts(model, split[partition], repeats=32, action_seed=1900)
            }
            if model is not None:
                result["greedy"] = evaluate_layouts(
                    model, split[partition], repeats=1, action_seed=1900, greedy=True
                )
            write_json(output / f"evaluation-{partition}.json", result)
            print(name, partition, "sampled", result["sampled"]["success_rate"], flush=True)
    write_json(
        args.output / "evaluation-manifest.json",
        {
            "split_sha256": split_hash,
            "evaluator_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            "sampled_repeats_per_map": 32,
            "action_seed": 1900,
            "note": "Final checkpoints only; no policy tuning or checkpoint selection on test maps",
        },
    )


if __name__ == "__main__":
    main()
