import numpy as np
import torch
from stable_baselines3 import PPO

from predictive_control.crossing import ENV_ID, LayoutPool, generate_split, load_split
from predictive_control.envs import make_env
from predictive_control.train_crossing import evaluate_layouts


def test_split_reproducible_and_geometry_disjoint():
    split = load_split()
    assert generate_split() == split
    sets = [{r["geometry_sha256"] for r in split[key]} for key in ["train", "development", "test"]]
    assert list(map(len, sets)) == [28, 7, 7]
    assert len(set.union(*sets)) == 42


def test_training_pool_never_uses_validation_geometries():
    split = load_split()
    env = LayoutPool(split["train"])
    expected = {r["geometry_sha256"] for r in split["train"]}
    try:
        observed = [env.reset(seed=seed)[1]["geometry_sha256"] for seed in range(100)]
        assert set(observed) <= expected
        assert len(set(observed)) > 20
    finally:
        env.close()


def test_sampled_evaluation_reproducible_without_changing_training_rng():
    torch.set_num_threads(1)
    env = make_env(ENV_ID)
    try:
        model = PPO("MlpPolicy", env, seed=3, n_steps=8, batch_size=8)
        state = torch.get_rng_state().clone()
        layouts = load_split()["development"][:1]
        a = evaluate_layouts(model, layouts, repeats=1)
        assert torch.equal(state, torch.get_rng_state())
        b = evaluate_layouts(model, layouts, repeats=1)
        assert a == b
        assert np.isfinite(a["mean_return"])
    finally:
        env.close()
