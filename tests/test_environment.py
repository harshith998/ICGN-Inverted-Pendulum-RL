"""Environment contract, deterministic resets, and episode boundaries."""

import numpy as np
import pytest
from gymnasium.utils.env_checker import check_env

from env.pendulum_env import VariablePendulumEnv


def test_gymnasium_contract():
    env = VariablePendulumEnv(n_links_range=(1, 3))
    try:
        check_env(env, skip_render_check=True)
    finally:
        env.close()


def test_ood_observation_is_in_declared_space():
    env = VariablePendulumEnv(
        n_links_range=(1, 1),
        link_length_range=(0.05, 0.1),
        link_mass_range=(0.01, 0.05),
    )
    try:
        obs, _ = env.reset(seed=7)
        assert (obs["edge_features"] < 0).any()
        assert env.observation_space.contains(obs)
    finally:
        env.close()


def test_failure_on_last_step_does_not_receive_win_bonus():
    env = VariablePendulumEnv(
        n_links_range=(1, 1),
        max_episode_steps=1,
        min_episode_steps=0,
        termination_angle=0.001,
        angle_noise=0.5,
    )
    try:
        env.reset(seed=2)
        _, _, terminated, truncated, info = env.step(np.array([0.0]))
        assert terminated and truncated
        assert "win_bonus" not in info["reward_components"]
    finally:
        env.close()


@pytest.mark.parametrize("action", [[np.nan], [np.inf], [0, 1]])
def test_invalid_action_rejected(action):
    env = VariablePendulumEnv(n_links_range=(1, 1))
    try:
        env.reset(seed=0)
        with pytest.raises(ValueError, match="finite force"):
            env.step(action)
    finally:
        env.close()
