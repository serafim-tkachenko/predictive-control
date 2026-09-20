import gymnasium as gym
import minigrid  # noqa: F401 -- registers environments
import numpy as np
from minigrid.wrappers import FullyObsWrapper, ImgObsWrapper


class FlatSymbolicObservation(gym.ObservationWrapper):
    """Full categorical grid, flattened and scaled; these are not RGB pixels."""

    def __init__(self, env):
        super().__init__(env)
        size = int(np.prod(env.observation_space.shape))
        self.observation_space = gym.spaces.Box(0.0, 1.0, (size,), dtype=np.float32)

    def observation(self, observation):
        return observation.astype(np.float32).reshape(-1) / 10.0


def make_env(env_id="MiniGrid-Empty-Random-5x5-v0", **kwargs):
    # FullyObs includes the agent location and heading in the grid encoding.
    return FlatSymbolicObservation(ImgObsWrapper(FullyObsWrapper(gym.make(env_id, **kwargs))))
