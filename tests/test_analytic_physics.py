"""Compare graph physics features with independent rigid-rod geometry."""

import numpy as np
import pytest
import torch

from env.mujoco_builder import PendulumConfig
from graph.graph_builder import build_graph
from models.cgat._physics import (
    compute_inertia_coupling,
    compute_gravity_torques,
    compute_hamiltonian,
)
from training.ppo_utils import batch_obs, obs_to_tensor


def observation(lengths, masses, angles):
    n = len(lengths)
    graph = build_graph(
        PendulumConfig(n, lengths, masses, 1.2), 0, 0, angles, np.zeros(n)
    )
    return obs_to_tensor(
        batch_obs(
            [
                dict(
                    node_features=graph.node_features,
                    edge_index=graph.edge_index,
                    edge_features=graph.edge_features,
                    n_nodes=np.array([n + 1]),
                    n_edges=np.array([2 * n]),
                )
            ]
        ),
        torch.device("cpu"),
    )


@pytest.mark.parametrize("n", [1, 2, 3])
def test_physics_features_match_relative_coordinate_geometry(n):
    lengths = np.linspace(0.3, 0.8, n)
    masses = np.linspace(0.1, 0.4, n)
    angles = np.linspace(0.15, -0.25, n)
    obs = observation(lengths, masses, angles)

    def com(q, rod):
        absolute = np.cumsum(q[1:])
        segments = (
            np.column_stack([np.sin(absolute), np.cos(absolute)]) * lengths[:, None]
        )
        return np.array([q[0], 0]) + segments[:rod].sum(axis=0) + segments[rod] / 2

    def potential(q):
        return sum(masses[i] * 9.81 * com(q, i)[1] for i in range(n))

    q = np.r_[0, angles]
    eps = 1e-5
    mass = np.zeros((n + 1, n + 1))
    mass[0, 0] = 1.2
    for i in range(n):
        jac = np.column_stack(
            [
                (
                    com(q + np.eye(n + 1)[j] * eps, i)
                    - com(q - np.eye(n + 1)[j] * eps, i)
                )
                / (2 * eps)
                for j in range(n + 1)
            ]
        )
        angular = np.zeros(n + 1)
        angular[1 : i + 2] = 1
        mass += masses[i] * jac.T @ jac + masses[i] * lengths[i] ** 2 / 12 * np.outer(
            angular, angular
        )
    expected = mass / np.sqrt(np.outer(np.diag(mass), np.diag(mass)))
    np.testing.assert_allclose(
        compute_inertia_coupling(obs)[0].numpy(), expected, atol=2e-6
    )
    torque = np.array(
        [
            -(
                potential(q + np.eye(n + 1)[j] * eps)
                - potential(q - np.eye(n + 1)[j] * eps)
            )
            / (2 * eps)
            for j in range(n + 1)
        ]
    )
    np.testing.assert_allclose(
        compute_gravity_torques(obs)[0].numpy(), torque / 100, atol=1e-7
    )
    assert compute_hamiltonian(obs).item() == pytest.approx(potential(q) / 20, abs=1e-7)
