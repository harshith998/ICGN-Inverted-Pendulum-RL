"""Evaluation must be reproducible and must not disguise software failures as scores."""

import copy

import numpy as np
import pytest
import torch
import yaml

from eval import eval_cgat, eval_dqn, eval_ppo


@pytest.fixture
def cfg():
    with open("configs/smoke.yaml") as stream:
        return yaml.safe_load(stream)


@pytest.mark.parametrize("module", [eval_cgat, eval_dqn, eval_ppo])
def test_eval_environment_honors_reward_and_initialization(module, cfg):
    cfg["init"] = {"angle_noise": 0, "vel_noise": 0}
    cfg["rewards"] = {"alive_bonus": 9}
    env = module.make_fixed_env(cfg, 0.5, 0.2)
    try:
        env.reset(seed=0)
        assert np.all(env._mj_data.qpos == 0)
        assert env.reward_config == cfg["rewards"]
    finally:
        env.close()


class BrokenPolicy:
    def get_deterministic_action(self, *args, **kwargs):
        raise RuntimeError("broken policy")

    def get_action(self, *args, **kwargs):
        raise RuntimeError("broken policy")


@pytest.mark.parametrize("module", [eval_cgat, eval_dqn, eval_ppo])
def test_policy_error_is_not_a_zero_reward(module, cfg):
    env = module.make_fixed_env(cfg, 0.5, 0.2)
    try:
        with pytest.raises(RuntimeError, match="broken policy"):
            if module is eval_dqn:
                module.eval_point(
                    BrokenPolicy(), env, 1, np.array([0]), torch.device("cpu")
                )
            else:
                module.eval_point(BrokenPolicy(), env, 1, torch.device("cpu"))
    finally:
        env.close()


def test_seeded_evaluation_repeats(cfg):
    policy = eval_ppo.RandomPPOPolicy(max_force=20)
    scores = []
    for _ in range(2):
        eval_ppo.set_seed(12)
        env = eval_ppo.make_fixed_env(cfg, 0.5, 0.2)
        try:
            scores.append(eval_ppo.eval_point(policy, env, 2, torch.device("cpu")))
        finally:
            env.close()
    assert scores[0] == scores[1]
