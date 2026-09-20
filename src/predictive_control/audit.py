"""Exhaustive behavioral audit of the 32 starts in MiniGrid Empty Random 5x5.

Finite-horizon policy evaluation integrates action sampling exactly, without rollouts.
This is specific to this deterministic room and a stationary, feed-forward policy.
"""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import torch
from stable_baselines3 import PPO

from predictive_control.envs import make_env
from predictive_control.train import write_json


def enumerate_room():
    states = [
        (x, y, d) for x in range(1, 4) for y in range(1, 4) if (x, y) != (3, 3) for d in range(4)
    ]
    index = {state: i for i, state in enumerate(states)}
    successors = np.full((len(states), 7), -1, dtype=int)
    observations = []
    env = make_env()
    try:
        for i, (x, y, direction) in enumerate(states):
            env.unwrapped.agent_start_pos = (x, y)
            env.unwrapped.agent_start_dir = direction
            obs, _ = env.reset(seed=0)
            observations.append(obs)
            for action in range(7):
                env.reset(seed=0)
                _, reward, terminated, truncated, _ = env.step(action)
                assert not truncated
                if terminated:
                    assert reward > 0
                else:
                    state = (*env.unwrapped.agent_pos, env.unwrapped.agent_dir)
                    successors[i, action] = index[state]
        horizon = env.unwrapped.max_steps
    finally:
        env.close()
    return states, np.stack(observations), successors, horizon


def exact_metrics(probabilities, successors, horizon):
    """Return per-start success, return and length for the native reward/time limit."""
    count = len(successors)
    transition = np.zeros((count, count), dtype=np.float64)
    goal = np.zeros(count, dtype=np.float64)
    for state in range(count):
        for action, probability in enumerate(probabilities[state]):
            target = successors[state, action]
            if target < 0:
                goal[state] += probability
            else:
                transition[state, target] += probability
    assert np.allclose(transition.sum(axis=1) + goal, 1)
    # Each row tracks surviving probability mass for one possible initial state.
    mass = np.eye(count)
    success, reward, length = (np.zeros(count) for _ in range(3))
    for step in range(1, horizon + 1):
        length += mass.sum(axis=1)
        hits = mass @ goal
        success += hits
        reward += hits * (1 - 0.9 * step / horizon)
        mass = mass @ transition
    return {"success": success.tolist(), "return": reward.tolist(), "length": length.tolist()}


def greedy_trajectories(probabilities, successors, states, horizon):
    actions = probabilities.argmax(axis=1)
    rows = []
    for initial in range(len(states)):
        state, seen, trace = initial, {}, []
        cycle_start = None
        for step in range(horizon):
            if state in seen:
                cycle_start = seen[state]
                break
            seen[state] = step
            action = int(actions[state])
            trace.append({"state": list(states[state]), "action": action})
            state = int(successors[state, action])
            if state < 0:
                break
        rows.append(
            {
                "initial_state": list(states[initial]),
                "success": state < 0,
                "cycle_start": cycle_start,
                "trace": trace,
            }
        )
    return rows


def seed_coverage(seeds):
    env = make_env()
    try:
        result = []
        for seed in seeds:
            env.reset(seed=seed)
            result.append([int(v) for v in (*env.unwrapped.agent_pos, env.unwrapped.agent_dir)])
        return {"states": result, "unique_states": len(set(map(tuple, result)))}
    finally:
        env.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("checkpoints", nargs="+", type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(1)
    states, observations, successors, horizon = enumerate_room()
    records = []
    for checkpoint in args.checkpoints:
        model = PPO.load(checkpoint, device="cpu")
        with torch.no_grad():
            tensor, _ = model.policy.obs_to_tensor(observations)
            probs = model.policy.get_distribution(tensor).distribution.probs.cpu().numpy()
        # Renormalize float32 probabilities in float64 before repeated multiplication.
        probs = probs.astype(np.float64)
        probs /= probs.sum(axis=1, keepdims=True)
        greedy = np.eye(7)[probs.argmax(axis=1)]
        trajectories = greedy_trajectories(probs, successors, states, horizon)
        result = {
            "run": checkpoint.parent.name,
            "checkpoint_sha256": hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
            "probabilities": probs.tolist(),
            "greedy": exact_metrics(greedy, successors, horizon),
            "sampled": exact_metrics(probs, successors, horizon),
            "greedy_trajectories": trajectories,
        }
        write_json(args.output / f"{checkpoint.parent.name}.json", result)
        records.append(
            {
                "run": result["run"],
                "greedy_success": float(np.mean(result["greedy"]["success"])),
                "sampled_success": float(np.mean(result["sampled"]["success"])),
                "cycle_failures": sum(t["cycle_start"] is not None for t in trajectories),
            }
        )
    summary = {
        "states": states,
        "successors": successors.tolist(),
        "horizon": horizon,
        "weighting": "uniform over all 32 valid position/heading combinations",
        "development_coverage": seed_coverage(range(10000, 10032)),
        "final_coverage": seed_coverage(range(20000, 20032)),
        "random": exact_metrics(np.full((32, 7), 1 / 7), successors, horizon),
        "runs": records,
        "audit_source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    }
    write_json(args.output / "summary.json", summary)
    print(json.dumps(records, indent=2))
    print(
        "Unique evaluation starts:",
        summary["development_coverage"]["unique_states"],
        summary["final_coverage"]["unique_states"],
    )


if __name__ == "__main__":
    main()
