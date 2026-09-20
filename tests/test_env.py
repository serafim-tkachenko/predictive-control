import numpy as np

from predictive_control.envs import make_env
from predictive_control.train import evaluate


def test_heading_visible_and_goal_terminates():
    env = make_env("MiniGrid-Empty-5x5-v0")
    try:
        obs, _ = env.reset(seed=0)
        assert env.observation_space.contains(obs)
        turned, *_ = env.step(0)
        assert not np.array_equal(obs, turned)
        env.reset(seed=0)
        # Fixed start (1,1), east-facing; goal (3,3).
        for action in [2, 2, 1, 2, 2]:
            obs, reward, terminated, truncated, _ = env.step(action)
        assert terminated and not truncated and reward > 0
    finally:
        env.close()


def test_time_limit_is_not_success():
    env = make_env(max_steps=1)
    try:
        env.reset(seed=0)
        _, reward, terminated, truncated, _ = env.step(0)
        assert truncated and not terminated and reward == 0
    finally:
        env.close()


def test_random_evaluation_reproducible():
    a = evaluate(None, "MiniGrid-Empty-Random-5x5-v0", [100, 101], action_seed=4)
    b = evaluate(None, "MiniGrid-Empty-Random-5x5-v0", [100, 101], action_seed=4)
    assert a == b
