"""PPO reference run with sampled-action evaluation on geometry-disjoint maps."""

import argparse
import hashlib
import importlib.metadata
import subprocess
import time
from pathlib import Path

import numpy as np
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.logger import configure

from predictive_control.crossing import ENV_ID, SPLIT_PATH, LayoutPool, geometry_hash, load_split
from predictive_control.envs import make_env
from predictive_control.train import write_json


def evaluate_layouts(model, layouts, repeats=4, action_seed=900, greedy=False):
    """Batch inference; each episode has a local RNG, never perturbing training RNG."""
    environments, observations, rows, generators = [], [], [], []
    try:
        for layout in layouts:
            for repeat in range(repeats):
                env = make_env(ENV_ID)
                environments.append(env)
                obs, _ = env.reset(seed=layout["seed"])
                assert geometry_hash(env) == layout["geometry_sha256"]
                observations.append(obs)
                generators.append(np.random.default_rng([action_seed, layout["seed"], repeat]))
                rows.append(
                    {
                        "geometry_sha256": layout["geometry_sha256"],
                        "map_seed": layout["seed"],
                        "repeat": repeat,
                        "return": 0.0,
                        "length": 0,
                        "success": False,
                    }
                )
        active = list(range(len(environments)))
        while active:
            if model is None:
                probabilities = np.full((len(active), 7), 1 / 7)
            else:
                with torch.no_grad():
                    tensor, _ = model.policy.obs_to_tensor(
                        np.stack([observations[i] for i in active])
                    )
                    probabilities = model.policy.get_distribution(tensor).distribution.probs
                    probabilities = probabilities.cpu().numpy().astype(np.float64)
                    probabilities /= probabilities.sum(axis=1, keepdims=True)
            remaining = []
            for i, probs in zip(active, probabilities, strict=True):
                action = int(probs.argmax()) if greedy else int(generators[i].choice(7, p=probs))
                obs, reward, terminated, truncated, _ = environments[i].step(action)
                observations[i] = obs
                rows[i]["return"] += float(reward)
                rows[i]["length"] += 1
                if terminated or truncated:
                    rows[i].update(
                        success=bool(terminated and reward > 0),
                        terminated=bool(terminated),
                        truncated=bool(truncated),
                    )
                else:
                    remaining.append(i)
            active = remaining
        return {
            "success_rate": float(np.mean([r["success"] for r in rows])),
            "mean_return": float(np.mean([r["return"] for r in rows])),
            "mean_length": float(np.mean([r["length"] for r in rows])),
            "action_mode": "greedy" if greedy else "sampled",
            "action_seed": action_seed,
            "episodes": rows,
        }
    finally:
        for env in environments:
            env.close()


class CrossingEvaluation(BaseCallback):
    def __init__(self, split, output):
        super().__init__()
        self.split, self.output, self.rows = split, output, []

    def record(self):
        # Equal 28-episode budgets: one sample per train map, four per dev map.
        row = {
            "steps": self.num_timesteps,
            "train": evaluate_layouts(self.model, self.split["train"], repeats=1),
            "development": evaluate_layouts(self.model, self.split["development"], repeats=4),
        }
        self.rows.append(row)
        write_json(self.output / "development.json", self.rows)
        print(
            f"step={self.num_timesteps} train={row['train']['success_rate']:.3f} "
            f"dev={row['development']['success_rate']:.3f}",
            flush=True,
        )

    def _on_training_start(self):
        self.record()

    def _on_step(self):
        if self.num_timesteps % 131072 == 0:
            self.record()
        return True


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--steps", type=int, default=524288)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    if args.steps <= 0 or args.steps % 1024:
        parser.error("steps must be a positive multiple of 1024")
    args.output.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(1)
    split = load_split()
    parameters = {
        "n_steps": 128,
        "batch_size": 256,
        "n_epochs": 4,
        "learning_rate": 3e-4,
        "gamma": 0.99,
        "gae_lambda": 0.95,
        "clip_range": 0.2,
        "ent_coef": 0.01,
        "policy_kwargs": {"net_arch": {"pi": [64, 64], "vf": [64, 64]}},
        "device": "cpu",
        "seed": args.seed,
    }
    write_json(
        args.output / "config.json",
        {
            "env": ENV_ID,
            "steps": args.steps,
            "seed": args.seed,
            "n_envs": 8,
            "ppo": parameters,
            "split": split,
            "observation": "full symbolic grid / 10, flattened",
            "reward": "native sparse reward; no shaping",
            "primary_evaluation": "sampled",
        },
    )
    commit = subprocess.run(
        ["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=True
    )
    dirty = subprocess.run(
        ["git", "status", "--porcelain"], capture_output=True, text=True, check=True
    )
    write_json(
        args.output / "manifest.json",
        {
            "git_commit": commit.stdout.strip(),
            "git_dirty": bool(dirty.stdout.strip()),
            "versions": {
                n: importlib.metadata.version(n)
                for n in ["torch", "numpy", "gymnasium", "minigrid", "stable-baselines3"]
            },
            "split_sha256": hashlib.sha256(SPLIT_PATH.read_bytes()).hexdigest(),
            "source_sha256": {
                p.name: hashlib.sha256(p.read_bytes()).hexdigest()
                for p in Path(__file__).parent.glob("*.py")
            },
        },
    )
    env = make_vec_env(
        lambda: LayoutPool(split["train"]),
        n_envs=8,
        seed=args.seed,
        monitor_dir=str(args.output / "monitor"),
    )
    try:
        model = PPO("MlpPolicy", env, **parameters)
        model.set_logger(configure(str(args.output), ["csv"]))
        callback = CrossingEvaluation(split, args.output)
        start = time.perf_counter()
        model.learn(total_timesteps=args.steps, callback=callback)
        model.save(args.output / "model")
        write_json(
            args.output / "completed.json",
            {
                "steps": model.num_timesteps,
                "training_and_eval_seconds": time.perf_counter() - start,
                "checkpoint_sha256": hashlib.sha256(
                    (args.output / "model.zip").read_bytes()
                ).hexdigest(),
            },
        )
        # No test-map evaluation during training or checkpoint selection.
        print(f"finished seed {args.seed}", flush=True)
    finally:
        env.close()


if __name__ == "__main__":
    main()
