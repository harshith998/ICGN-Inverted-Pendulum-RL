"""Shared observation batching, rollout storage, and PPO objective."""

import numpy as np
import torch
import torch.nn.functional as F

from graph.graph_builder import NODE_FEAT_DIM, EDGE_FEAT_DIM


def set_seed(seed: int | None):
    if seed is None:
        return
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def seed_suffix(seed: int | None) -> str:
    return "" if seed is None else f"_seed{seed}"


# ---------------------------------------------------------------------------
# Observation helpers
# ---------------------------------------------------------------------------


def batch_obs(obs_list: list) -> dict:
    """Stack N per-env obs dicts into a single batched obs dict (B=N)."""
    return {
        "node_features": np.stack([o["node_features"] for o in obs_list]),
        "edge_index": np.stack([o["edge_index"] for o in obs_list]),
        "edge_features": np.stack([o["edge_features"] for o in obs_list]),
        "n_nodes": np.stack([o["n_nodes"] for o in obs_list]),
        "n_edges": np.stack([o["n_edges"] for o in obs_list]),
    }


def obs_to_tensor(obs_batch: dict, device: torch.device) -> dict:
    """Convert batched numpy obs dict to tensors on device."""
    return {
        "node_features": torch.tensor(
            obs_batch["node_features"], dtype=torch.float32
        ).to(device),
        "edge_index": torch.tensor(obs_batch["edge_index"], dtype=torch.int64).to(
            device
        ),
        "edge_features": torch.tensor(
            obs_batch["edge_features"], dtype=torch.float32
        ).to(device),
        "n_nodes": torch.tensor(obs_batch["n_nodes"], dtype=torch.int64).to(device),
        "n_edges": torch.tensor(obs_batch["n_edges"], dtype=torch.int64).to(device),
    }


# ---------------------------------------------------------------------------
# Rollout Buffer  (T × N layout)
# ---------------------------------------------------------------------------


class RolloutBuffer:
    """
    On-policy buffer for N parallel environments.
    Layout: (rollout_steps, n_envs, ...).
    GAE is computed per-env along the time axis.
    Flattened to (T*N, ...) when yielding mini-batches.
    """

    def __init__(
        self,
        rollout_steps: int,
        n_envs: int,
        max_nodes: int,
        max_edges: int,
        gamma: float,
        gae_lambda: float,
    ):
        self.rollout_steps = rollout_steps
        self.n_envs = n_envs
        self.gamma = gamma
        self.gae_lambda = gae_lambda

        T, N = rollout_steps, n_envs
        self.node_feat = np.zeros((T, N, max_nodes, NODE_FEAT_DIM), dtype=np.float32)
        self.edge_index = np.zeros((T, N, 2, max_edges), dtype=np.int64)
        self.edge_feat = np.zeros((T, N, max_edges, EDGE_FEAT_DIM), dtype=np.float32)
        self.n_nodes = np.zeros((T, N, 1), dtype=np.int64)
        self.n_edges = np.zeros((T, N, 1), dtype=np.int64)

        self.actions = np.zeros((T, N), dtype=np.float32)
        self.log_probs = np.zeros((T, N), dtype=np.float32)
        self.rewards = np.zeros((T, N), dtype=np.float32)
        self.values = np.zeros((T, N), dtype=np.float32)
        self.dones = np.zeros((T, N), dtype=np.float32)

        self.returns = np.zeros((T, N), dtype=np.float32)
        self.advantages = np.zeros((T, N), dtype=np.float32)

        self.pos = 0

    def store(
        self,
        obs_list: list,
        actions: np.ndarray,
        log_probs: np.ndarray,
        rewards: np.ndarray,
        values: np.ndarray,
        dones: np.ndarray,
    ):
        """Store one timestep of data from all N envs. obs_list has length N."""
        t = self.pos
        for n, obs in enumerate(obs_list):
            self.node_feat[t, n] = obs["node_features"]
            self.edge_index[t, n] = obs["edge_index"]
            self.edge_feat[t, n] = obs["edge_features"]
            self.n_nodes[t, n] = obs["n_nodes"]
            self.n_edges[t, n] = obs["n_edges"]
        self.actions[t] = actions
        self.log_probs[t] = log_probs
        self.rewards[t] = rewards
        self.values[t] = values
        self.dones[t] = dones
        self.pos += 1

    def compute_gae(self, last_values: np.ndarray):
        """
        GAE-λ computed jointly across all N envs.
        last_values: (N,) bootstrapped V(s_{T+1}) from the final state.
        Done flags mask the carry-over between episodes correctly.
        """
        gae = np.zeros(self.n_envs, dtype=np.float32)
        for t in reversed(range(self.rollout_steps)):
            next_val = (
                last_values if t == self.rollout_steps - 1 else self.values[t + 1]
            )
            delta = (
                self.rewards[t]
                + self.gamma * next_val * (1.0 - self.dones[t])
                - self.values[t]
            )
            gae = delta + self.gamma * self.gae_lambda * (1.0 - self.dones[t]) * gae
            self.advantages[t] = gae
            self.returns[t] = gae + self.values[t]

    def generate_batches(self, batch_size: int, device: torch.device):
        """Flatten (T, N) → (T*N,) and yield shuffled mini-batches as tensor dicts."""
        T, N = self.rollout_steps, self.n_envs
        total = T * N
        indices = np.random.permutation(total)

        # Reshape (T, N, ...) → (T*N, ...)
        node_feat_f = self.node_feat.reshape(total, *self.node_feat.shape[2:])
        edge_index_f = self.edge_index.reshape(total, *self.edge_index.shape[2:])
        edge_feat_f = self.edge_feat.reshape(total, *self.edge_feat.shape[2:])
        n_nodes_f = self.n_nodes.reshape(total, 1)
        n_edges_f = self.n_edges.reshape(total, 1)
        actions_f = self.actions.reshape(total)
        log_probs_f = self.log_probs.reshape(total)
        returns_f = self.returns.reshape(total)
        adv_f = self.advantages.reshape(total)

        for start in range(0, total, batch_size):
            idx = indices[start : start + batch_size]
            obs_batch = {
                "node_features": torch.tensor(node_feat_f[idx], dtype=torch.float32).to(
                    device
                ),
                "edge_index": torch.tensor(edge_index_f[idx], dtype=torch.int64).to(
                    device
                ),
                "edge_features": torch.tensor(edge_feat_f[idx], dtype=torch.float32).to(
                    device
                ),
                "n_nodes": torch.tensor(n_nodes_f[idx], dtype=torch.int64).to(device),
                "n_edges": torch.tensor(n_edges_f[idx], dtype=torch.int64).to(device),
            }
            yield (
                obs_batch,
                torch.tensor(actions_f[idx], dtype=torch.float32)
                .to(device)
                .unsqueeze(1),
                torch.tensor(log_probs_f[idx], dtype=torch.float32).to(device),
                torch.tensor(returns_f[idx], dtype=torch.float32).to(device),
                torch.tensor(adv_f[idx], dtype=torch.float32).to(device),
            )

    def reset(self):
        self.pos = 0


# ---------------------------------------------------------------------------
# PPO loss
# ---------------------------------------------------------------------------


def compute_ppo_loss(
    policy,
    obs,
    actions,
    old_log_probs,
    returns,
    advantages,
    clip_epsilon,
    value_coef,
    entropy_coef,
):
    """Clipped surrogate objective + value loss + entropy bonus."""
    _, new_log_probs, entropy, values = policy.get_action_and_value(obs, action=actions)

    # Normalise advantages within mini-batch
    advantages = (advantages - advantages.mean()) / (
        advantages.std(unbiased=False) + 1e-8
    )

    ratio = (new_log_probs - old_log_probs).exp()
    surr1 = ratio * advantages
    surr2 = ratio.clamp(1.0 - clip_epsilon, 1.0 + clip_epsilon) * advantages
    policy_loss = -torch.min(surr1, surr2).mean()

    value_loss = F.mse_loss(values.squeeze(-1), returns)
    entropy_loss = -entropy.mean()

    total_loss = policy_loss + value_coef * value_loss + entropy_coef * entropy_loss
    return total_loss, policy_loss.item(), value_loss.item(), (-entropy_loss).item()
