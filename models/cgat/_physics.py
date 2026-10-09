"""
Shared physics helpers for all CGAT variants.
All functions are pure (no learned parameters).
"""

import torch

NODE_FEAT_DIM = 9
EDGE_FEAT_DIM = 2

# ── Denormalisation constants (must match graph/graph_builder.py) ──────────────
_LEN_MIN         = 0.3;  _LEN_RANGE       = 0.9
_MASS_MIN        = 0.1;  _MASS_RANGE      = 1.9
_CART_MASS_MIN   = 0.5;  _CART_MASS_RANGE = 2.5
_ANG_VEL         = 10.0
_G               = 9.81
_G_TORQUE_SCALE  = 100.0   # normalises gravity torques to ~[-1, 1]
_H_SCALE         = 20.0    # normalises potential energy


def _rod_tensors(obs: dict):
    """
    Extract per-rod (L, m, sin_th, cos_th, theta_dot) and validity mask.
    Returns: L, m, sin_th, cos_th, theta_dot — each (B, max_links); rod_valid (B, max_links) bool
    """
    node_feats = obs["node_features"].float()   # (B, max_nodes, 9)
    edge_feats = obs["edge_features"].float()   # (B, max_edges, 2)
    n_nodes    = obs["n_nodes"].long()

    B, max_nodes, _ = node_feats.shape
    max_links = max_nodes - 1
    device    = node_feats.device

    rod_ef    = edge_feats[:, 0::2, :]
    L         = rod_ef[..., 0] * _LEN_RANGE  + _LEN_MIN
    m         = rod_ef[..., 1] * _MASS_RANGE + _MASS_MIN

    n_links   = n_nodes.squeeze(-1) - 1
    rod_valid = torch.arange(max_links, device=device).unsqueeze(0) < n_links.unsqueeze(1)

    L = L * rod_valid.float()
    m = m * rod_valid.float()

    # MuJoCo observations contain relative hinge angles. Analytic rod
    # geometry uses absolute orientations, obtained by cumulative summation.
    relative = torch.atan2(node_feats[:, 1:, 3], node_feats[:, 1:, 4])
    absolute = torch.cumsum(relative, dim=1)
    sin_th = torch.sin(absolute)
    cos_th = torch.cos(absolute)
    theta_dot = node_feats[:, 1:, 5] * _ANG_VEL

    return L, m, sin_th, cos_th, theta_dot, rod_valid


def compute_inertia_coupling(obs: dict) -> torch.Tensor:
    """
    Analytically compute normalised Lagrangian mass matrix M̃(q).
    M̃ᵢⱼ = Mᵢⱼ / √(Mᵢᵢ · Mⱼⱼ) ∈ [-1, 1].
    Returns (B, max_nodes, max_nodes).
    """
    L, m, sin_th, cos_th, _, rod_valid = _rod_tensors(obs)
    B         = L.shape[0]
    max_links = L.shape[1]
    max_nodes = max_links + 1
    device    = L.device

    node_feats = obs["node_features"].float()
    m_cart     = node_feats[:, 0, 8] * _CART_MASS_RANGE + _CART_MASS_MIN

    distal      = torch.flip(torch.cumsum(torch.flip(m, [1]), 1), [1])
    distal_excl = distal - m

    M = torch.zeros(B, max_nodes, max_nodes, device=device)
    M[:, 0, 0] = m_cart + m.sum(dim=1)

    M_0j = L * cos_th * (distal - m / 2) * rod_valid.float()
    M[:, 0, 1:] = M_0j
    M[:, 1:, 0] = M_0j

    M_diag = L ** 2 * (m / 3 + distal_excl) * rod_valid.float()
    for j in range(max_links):
        M[:, j+1, j+1] = M_diag[:, j]

    for j in range(max_links):
        for k in range(j+1, max_links):
            cos_jk = cos_th[:, j] * cos_th[:, k] + sin_th[:, j] * sin_th[:, k]
            M_jk   = L[:, j] * L[:, k] * cos_jk * (distal[:, k] - m[:, k] / 2)
            valid  = (rod_valid[:, j] & rod_valid[:, k]).float()
            M[:, j+1, k+1] = M[:, k+1, j+1] = M_jk * valid

    # q_absolute = T q_relative, so M_relative = T.T M_absolute T.
    transform = torch.eye(max_nodes, device=device, dtype=M.dtype)
    transform[1:, 1:] = torch.tril(torch.ones(max_links, max_links, device=device))
    M = transform.T @ M @ transform
    diag_sqrt = M.diagonal(dim1=-2, dim2=-1).clamp(min=1e-6).sqrt()
    denom     = (diag_sqrt.unsqueeze(-1) * diag_sqrt.unsqueeze(-2)).clamp(min=1e-6)
    return M / denom


def compute_gravity_torques(obs: dict) -> torch.Tensor:
    """
    Analytic gravitational torque on each node (normalised).
    τ_g_i = g · Lᵢ · sin(θᵢ) · (distal_i − mᵢ/2)
    Returns (B, max_nodes): cart node = 0, joint nodes = normalised torque.
    """
    L, m, sin_th, _, _, rod_valid = _rod_tensors(obs)
    B      = L.shape[0]
    device = L.device

    distal = torch.flip(torch.cumsum(torch.flip(m, [1]), 1), [1])
    tau    = _G * L * sin_th * (distal - m / 2) * rod_valid.float()
    # A relative hinge rotates every downstream rod.
    tau = torch.flip(torch.cumsum(torch.flip(tau, [1]), 1), [1])
    tau = tau / _G_TORQUE_SCALE

    cart_zeros = torch.zeros(B, 1, device=device)
    return torch.cat([cart_zeros, tau], dim=1)   # (B, max_nodes)


def compute_hamiltonian(obs: dict) -> torch.Tensor:
    """
    Analytically compute normalised potential energy V_pot / H_scale.
    Returns (B,) — maximum when all links are upright; may be negative.
    """
    L, m, _, cos_th, _, rod_valid = _rod_tensors(obs)
    rv        = rod_valid.float()
    L_cos     = L * cos_th
    cum_L_cos = torch.cumsum(L_cos, dim=1) - L_cos
    h_com     = cum_L_cos + L * cos_th / 2
    V         = _G * (m * h_com * rv).sum(dim=1)
    return V / _H_SCALE
