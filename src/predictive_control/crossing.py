"""Fixed, geometry-disjoint splits of the native one-wall crossing environment."""

import hashlib
import json
from pathlib import Path

import gymnasium as gym
import numpy as np

from predictive_control.envs import make_env

ENV_ID = "MiniGrid-SimpleCrossingS9N1-v0"
SPLIT_PATH = Path(__file__).resolve().parent / "data/crossing-v1.json"


def geometry_hash(env):
    return hashlib.sha256(env.unwrapped.grid.encode().tobytes()).hexdigest()


def generate_split():
    env = make_env(ENV_ID)
    found = {}
    try:
        for seed in range(10000):
            env.reset(seed=seed)
            key = geometry_hash(env)
            found.setdefault(key, {"seed": seed, "geometry_sha256": key})
            if len(found) == 42:
                break
    finally:
        env.close()
    if len(found) != 42:
        raise RuntimeError("Expected six wall positions/orientations times seven openings")
    layouts = [found[key] for key in sorted(found)]
    np.random.default_rng(20260920).shuffle(layouts)
    return {
        "env": ENV_ID,
        "generator_seed": 20260920,
        "train": layouts[:28],
        "development": layouts[28:35],
        "test": layouts[35:],
    }


def load_split():
    return json.loads(SPLIT_PATH.read_text())


class LayoutPool(gym.Wrapper):
    """Choose uniformly from specified native map seeds on each episode reset."""

    def __init__(self, layouts):
        super().__init__(make_env(ENV_ID))
        self.layouts = layouts
        self.sampler = np.random.default_rng(0)

    def reset(self, *, seed=None, options=None):
        if seed is not None:
            self.sampler = np.random.default_rng(seed)
        layout = self.layouts[int(self.sampler.integers(len(self.layouts)))]
        obs, info = self.env.reset(seed=layout["seed"], options=options)
        info["geometry_sha256"] = geometry_hash(self.env)
        if info["geometry_sha256"] != layout["geometry_sha256"]:
            raise RuntimeError("Environment geometry no longer matches the frozen split")
        return obs, info
