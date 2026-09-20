"""Check the exact calculation against actual environment rollouts and SB3 sampling."""

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from stable_baselines3 import PPO

from predictive_control.audit import enumerate_room
from predictive_control.envs import make_env
from predictive_control.train import write_json


def rollout_validation(model, states, repeats):
    env = make_env()
    rng = np.random.default_rng(777)
    torch.manual_seed(777)
    episodes = []
    try:
        for x, y, direction in states:
            env.unwrapped.agent_start_pos = (x, y)
            env.unwrapped.agent_start_dir = direction
            for repeat in range(repeats):
                obs, _ = env.reset(seed=repeat)
                total, length = 0.0, 0
                while True:
                    action = (
                        int(rng.integers(7))
                        if model is None
                        else int(model.predict(obs, deterministic=False)[0])
                    )
                    obs, reward, terminated, truncated, _ = env.step(action)
                    total += reward
                    length += 1
                    if terminated or truncated:
                        episodes.append(
                            {
                                "state": [x, y, direction],
                                "repeat": repeat,
                                "success": bool(terminated and reward > 0),
                                "return": float(total),
                                "length": length,
                            }
                        )
                        break
    finally:
        env.close()
    return {
        "success": float(np.mean([e["success"] for e in episodes])),
        "return": float(np.mean([e["return"] for e in episodes])),
        "length": float(np.mean([e["length"] for e in episodes])),
        "episodes": episodes,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--audit", required=True, type=Path)
    parser.add_argument("--runs", default=Path("runs"), type=Path)
    parser.add_argument("--repeats", default=32, type=int)
    args = parser.parse_args()
    if args.repeats <= 0:
        parser.error("repeats must be positive")
    output = args.audit / "rollout-validation.json"
    if output.exists():
        parser.error("validation already exists")
    torch.set_num_threads(1)
    states, _, _, _ = enumerate_room()
    summary = json.loads((args.audit / "summary.json").read_text())
    records = []
    for row in summary["runs"] + [{"run": "random"}]:
        name = row["run"]
        if name == "random":
            model, expected = None, summary["random"]
        else:
            model = PPO.load(args.runs / name / "model.zip", device="cpu")
            expected = json.loads((args.audit / f"{name}.json").read_text())["sampled"]
        observed = rollout_validation(model, states, args.repeats)
        expected_means = {key: float(np.mean(values)) for key, values in expected.items()}
        record = {"run": name, "exact": expected_means, "observed": observed}
        records.append(record)
        print(
            name,
            "exact",
            expected_means,
            "observed",
            {k: v for k, v in observed.items() if k != "episodes"},
            flush=True,
        )
    write_json(output, {"action_seed": 777, "repeats_per_state": args.repeats, "runs": records})


if __name__ == "__main__":
    main()
