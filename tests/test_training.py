"""Policy gradient and rollout math regression tests."""

import numpy as np
import pytest
import torch

from env.pendulum_env import VariablePendulumEnv
from models.cgat import load_cgat_variant
from models.gnn_mpnn_ppo import GNNMPNNPPOPolicy
from models.gnn_transformer_ppo import GNNTransformerPPOPolicy
from models.mlp_ppo import MLPPPOPolicy
from training.ppo_utils import RolloutBuffer, batch_obs, compute_ppo_loss, obs_to_tensor


@pytest.mark.parametrize(
    "kind",
    [
        "mlp",
        "mpnn",
        "transformer",
        "base",
        "perhead",
        "directional",
        "gravity",
        "perc",
        "no_physics",
        "shuffled",
    ],
)
def test_policy_mixed_topology_backward(kind):
    env = VariablePendulumEnv(n_links_range=(1, 3), max_links=3)
    try:
        obs = obs_to_tensor(
            batch_obs([env.reset(seed=s)[0] for s in range(3)]), torch.device("cpu")
        )
    finally:
        env.close()
    if kind == "mlp":
        policy = MLPPPOPolicy(hidden=32, max_links=3, dropout=0)
    elif kind == "mpnn":
        policy = GNNMPNNPPOPolicy(hidden=32, n_layers=1, max_links=3, dropout=0)
    elif kind == "transformer":
        policy = GNNTransformerPPOPolicy(
            hidden=32, n_layers=1, n_heads=2, max_links=3, dropout=0
        )
    else:
        policy = load_cgat_variant(
            kind, hidden=32, n_icga_layers=1, n_heads=2, max_links=3
        )
    action, log_prob, entropy, value = policy.get_action_and_value(obs)
    assert action.shape == (3, 1)
    assert (action.abs() <= policy.max_force).all()
    assert all(torch.isfinite(t).all() for t in [action, log_prob, entropy, value])
    (log_prob.mean() + value.mean()).backward()
    assert all(
        torch.isfinite(p.grad).all() for p in policy.parameters() if p.grad is not None
    )


def test_gae_does_not_cross_episode_boundary():
    buffer = RolloutBuffer(3, 1, 2, 2, gamma=0.9, gae_lambda=1)
    buffer.rewards[:, 0] = [1, 2, 100]
    buffer.values[:, 0] = [0, 0, 0]
    buffer.dones[:, 0] = [0, 1, 0]
    buffer.compute_gae(np.array([10]))
    np.testing.assert_allclose(buffer.returns[:, 0], [2.8, 2, 109])


def test_single_sample_ppo_minibatch_has_finite_loss():
    env = VariablePendulumEnv(n_links_range=(1, 1), max_links=3)
    try:
        obs = obs_to_tensor(batch_obs([env.reset(seed=0)[0]]), torch.device("cpu"))
    finally:
        env.close()
    policy = MLPPPOPolicy(hidden=32, max_links=3, dropout=0)
    action, logp, _, _ = policy.get_action_and_value(obs)
    loss, *_ = compute_ppo_loss(
        policy,
        obs,
        action.detach(),
        logp.detach(),
        torch.ones(1),
        torch.ones(1),
        0.2,
        0.5,
        0.01,
    )
    assert torch.isfinite(loss)
    loss.backward()
