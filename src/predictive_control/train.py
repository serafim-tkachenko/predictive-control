"""PPO calibration run with separate evaluation environments and raw episode records."""

import argparse
import hashlib
import importlib.metadata
import json
import platform
import subprocess
import time
from pathlib import Path

import numpy as np
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.logger import configure

from predictive_control.envs import make_env


def evaluate(model, env_id, seeds, action_seed=0):
    """A fresh environment per evaluation; positive terminal reward is Empty success."""
    env = make_env(env_id)
    rng = np.random.default_rng(action_seed)
    episodes = []
    try:
        for seed in seeds:
            obs, _ = env.reset(seed=seed)
            total, length = 0.0, 0
            while True:
                action = (
                    int(rng.integers(env.action_space.n))
                    if model is None
                    else int(model.predict(obs, deterministic=True)[0])
                )
                obs, reward, terminated, truncated, _ = env.step(action)
                total += float(reward)
                length += 1
                if terminated or truncated:
                    episodes.append(
                        {
                            "seed": seed,
                            "return_": total,
                            "length": length,
                            "success": bool(terminated and reward > 0),
                            "terminated": bool(terminated),
                            "truncated": bool(truncated),
                        }
                    )
                    break
    finally:
        env.close()
    return {
        "success_rate": float(np.mean([e["success"] for e in episodes])),
        "mean_return": float(np.mean([e["return_"] for e in episodes])),
        "mean_length": float(np.mean([e["length"] for e in episodes])),
        "episodes": episodes,
    }


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")


class EvaluationCallback(BaseCallback):
    def __init__(self, env_id, output, frequency, episodes):
        super().__init__()
        self.env_id, self.output, self.frequency = env_id, output, frequency
        self.seeds = list(range(10_000, 10_000 + episodes))
        self.rows = []

    def record(self):
        score = evaluate(self.model, self.env_id, self.seeds)
        row = dict(steps=self.num_timesteps, **score)
        self.rows.append(row)
        write_json(self.output / "development.json", self.rows)
        print(
            f"steps={self.num_timesteps} success={score['success_rate']:.3f} "
            f"return={score['mean_return']:.3f}",
            flush=True,
        )

    def _on_training_start(self):
        self.record()

    def _on_step(self):
        if self.num_timesteps % self.frequency == 0:
            self.record()
        return True


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--steps", type=int, default=32768)
    parser.add_argument("--eval-episodes", type=int, default=32)
    parser.add_argument(
        "--env", default="MiniGrid-Empty-Random-5x5-v0", choices=["MiniGrid-Empty-Random-5x5-v0"]
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.steps <= 0 or args.steps % 1024 or args.eval_episodes <= 0:
        parser.error("steps must be a positive multiple of 1024; eval-episodes must be positive")
    args.output.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(1)
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
    config = {
        "env": args.env,
        "seed": args.seed,
        "total_steps": args.steps,
        "n_envs": 8,
        "eval_episodes": args.eval_episodes,
        "observation": "full symbolic grid / 10, flattened",
        "action_space": "all 7 native actions",
        "ppo": parameters,
    }
    write_json(args.output / "config.json", config)
    versions = {
        name: importlib.metadata.version(name)
        for name in ["stable-baselines3", "torch", "gymnasium", "minigrid", "numpy"]
    }
    commit = subprocess.run(
        ["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=False
    )
    dirty = subprocess.run(
        ["git", "status", "--porcelain"], capture_output=True, text=True, check=False
    )
    write_json(
        args.output / "manifest.json",
        {
            "python": platform.python_version(),
            "platform": platform.platform(),
            "versions": versions,
            "git_commit": commit.stdout.strip() or None,
            "git_dirty": bool(dirty.stdout.strip()),
            "source_sha256": {
                p.name: hashlib.sha256(p.read_bytes()).hexdigest()
                for p in Path(__file__).parent.glob("*.py")
            },
        },
    )
    development_seeds = list(range(10_000, 10_000 + args.eval_episodes))
    write_json(
        args.output / "random_development.json",
        evaluate(None, args.env, development_seeds, action_seed=123),
    )
    env = make_vec_env(
        lambda: make_env(args.env),
        n_envs=8,
        seed=args.seed,
        monitor_dir=str(args.output / "monitor"),
    )
    try:
        model = PPO("MlpPolicy", env, **parameters)
        model.set_logger(configure(str(args.output), ["csv"]))
        callback = EvaluationCallback(args.env, args.output, 8192, args.eval_episodes)
        start = time.perf_counter()
        model.learn(total_timesteps=args.steps, callback=callback)
        elapsed = time.perf_counter() - start
        model.save(args.output / "model")
        checkpoint = args.output / "model.zip"
        # Final checkpoint, never selected on evaluation success.
        seeds = list(range(20_000, 20_000 + args.eval_episodes))
        write_json(
            args.output / "final.json",
            {
                "steps": model.num_timesteps,
                "training_and_dev_eval_seconds": elapsed,
                "checkpoint_sha256": hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
                "trained": evaluate(model, args.env, seeds),
                "random": evaluate(None, args.env, seeds, action_seed=456),
            },
        )
        print(
            f"Finished seed={args.seed}, steps={model.num_timesteps}, seconds={elapsed:.1f}",
            flush=True,
        )
    finally:
        env.close()


if __name__ == "__main__":
    main()
