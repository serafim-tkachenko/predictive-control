"""Render the actual native layouts in each frozen partition."""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from predictive_control.crossing import ENV_ID, load_split
from predictive_control.envs import make_env


def main():
    fig, axes = plt.subplots(6, 7, figsize=(11, 10), layout="constrained")
    split = load_split()
    env = make_env(ENV_ID)
    try:
        for row, key, offset in [
            (0, "train", 0),
            (1, "train", 7),
            (2, "train", 14),
            (3, "train", 21),
            (4, "development", 0),
            (5, "test", 0),
        ]:
            for col in range(7):
                layout = split[key][offset + col]
                env.reset(seed=layout["seed"])
                axes[row, col].imshow(env.unwrapped.get_frame(highlight=False, tile_size=12))
                axes[row, col].axis("off")
                axes[row, col].set_title(f"{key} {layout['geometry_sha256'][:5]}", fontsize=8)
    finally:
        env.close()
    fig.suptitle("Crossing v1: 28 train / 7 development / 7 test geometries", fontsize=14)
    fig.savefig(Path("protocols/crossing-layouts.png"), dpi=140)
    plt.close(fig)


if __name__ == "__main__":
    main()
