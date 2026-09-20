import numpy as np

from predictive_control.audit import enumerate_room, exact_metrics, greedy_trajectories


def test_exact_evaluation_matches_geometric_absorption():
    result = exact_metrics(np.array([[0.25, 0.75]]), np.array([[-1, 0]]), horizon=4)
    hits = np.array([0.75 ** (step - 1) * 0.25 for step in range(1, 5)])
    assert np.isclose(result["success"][0], 1 - 0.75**4)
    assert np.isclose(result["return"][0], sum(hits * (1 - 0.9 * np.arange(1, 5) / 4)))
    assert np.isclose(result["length"][0], sum(0.75**step for step in range(4)))


def test_all_states_distinguishable_and_shortest_path_reaches_goal():
    states, observations, successors, horizon = enumerate_room()
    assert len(states) == len(np.unique(observations, axis=0)) == 32
    distance = np.full(32, np.inf)
    for _ in range(32):
        distance = np.array(
            [min(1 if t < 0 else 1 + distance[t] for t in row) for row in successors]
        )
    assert np.isfinite(distance).all()
    actions = [np.argmin([0 if t < 0 else distance[t] for t in row]) for row in successors]
    policy = np.eye(7)[actions]
    result = exact_metrics(policy, successors, horizon)
    assert np.allclose(result["success"], 1)
    assert np.allclose(result["length"], distance)
    assert all(t["success"] for t in greedy_trajectories(policy, successors, states, horizon))


def test_greedy_wall_collision_is_detected_as_cycle():
    states, _, successors, horizon = enumerate_room()
    forward = np.tile(np.eye(7)[2], (32, 1))
    traces = greedy_trajectories(forward, successors, states, horizon)
    row = traces[states.index((1, 1, 2))]  # Heading west into the wall.
    assert not row["success"] and row["cycle_start"] == 0
    assert len(row["trace"]) == 1
