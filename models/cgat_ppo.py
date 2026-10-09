# Legacy model class; current training uses models/cgat/ via --variant base.

"""
Coupled Graph Attention Transformer (CGAT) — actor-critic for PPO.

Architecture component (operating on the same observations as GNNTransformerPPO):

  Inertia-Coupled Graph Attention (ICGA)
  ───────────────────────────────────────
  The normalised inertia coupling M̃ᵢⱼ(q; L, m) is added as an analytic
  physics bias to each attention logit:

      logit_ij = Q·K/√d  +  β · M̃ᵢⱼ  +  w_e · e_ij

  M̃ᵢⱼ = Mᵢⱼ / √(Mᵢᵢ · Mⱼⱼ)  ∈ [-1, 1]

  is the normalised entry of the Lagrangian mass matrix. It encodes how
  strongly joints i and j are kinematically coupled — high when they share
  heavy distal mass, near zero when mechanically decoupled. β is a learned
  scalar initialised to 0 (degrades to standard transformer at β=0).

  The mass matrix is computed analytically from the existing node/edge
  features — no new information enters the observation.

# TODO: add w_H (PERC — potential energy residual critic) and RIM (bidirectional inertia message passing) later

Math reference (2-link, generalises to n-link):
  Mᵢⱼ(q) = Σₖ≥max(i,j) mₖ · Jᵢₖ(q) · Jⱼₖ(q) + δᵢⱼ Iᵢ
"""

import torch
import torch.nn as nn
from models.base_ppo import BasePPOPolicy

NODE_FEAT_DIM = 9
EDGE_FEAT_DIM = 2

# ── Denormalisation constants (must match graph/graph_builder.py) ──────────────
_LEN_MIN         = 0.3;  _LEN_RANGE       = 0.9   # L = norm * 0.9 + 0.3
_MASS_MIN        = 0.1;  _MASS_RANGE      = 1.9   # m = norm * 1.9 + 0.1
_CART_MASS_MIN   = 0.5;  _CART_MASS_RANGE = 2.5   # m_c = norm * 2.5 + 0.5
_ANG_VEL         = 10.0                            # θ̇  normalisation


# ═══════════════════════════════════════════════════════════════════════════════
# Physics helpers  (pure functions, no learned parameters)
# ═══════════════════════════════════════════════════════════════════════════════

from models.cgat._physics import compute_inertia_coupling


# ═══════════════════════════════════════════════════════════════════════════════
# 1. Inertia-Coupled Graph Attention (ICGA)
# ═══════════════════════════════════════════════════════════════════════════════

class ICGALayer(nn.Module):
    """
    Graph attention layer with an analytic physics bias on attention logits:

        logit_ij = Q_i · K_j / √d_head  +  β · M̃ᵢⱼ  +  w_e · e_ij
                                              ↑ physics (analytic)

    β is a single learned scalar shared across all heads (init=0 → degrades
    to standard graph transformer at the start of training).

    Numerically stable softmax via the log-sum-exp trick (identical to
    gnn_transformer_ppo.py to avoid NaN on overflow).
    """

    def __init__(self, hidden: int, n_heads: int):
        super().__init__()
        assert hidden % n_heads == 0
        self.n_heads = n_heads
        self.d_head  = hidden // n_heads

        self.W_q = nn.Linear(hidden, hidden, bias=False)
        self.W_k = nn.Linear(hidden, hidden, bias=False)
        self.W_v = nn.Linear(hidden, hidden, bias=False)
        self.W_o = nn.Linear(hidden, hidden)

        self.edge_bias    = nn.Linear(EDGE_FEAT_DIM, n_heads, bias=False)
        # β: physics-coupling scale — init=0 so model starts as standard transformer
        self.physics_beta = nn.Parameter(torch.zeros(1))

        self.norm1 = nn.LayerNorm(hidden)
        self.norm2 = nn.LayerNorm(hidden)
        self.ff    = nn.Sequential(
            nn.Linear(hidden, hidden * 2), nn.ReLU(),
            nn.Linear(hidden * 2, hidden),
        )

    def forward(self, h, edge_features, edge_index, n_edges, M_tilde):
        """
        h            : (B, max_nodes, hidden)
        edge_features: (B, max_edges, 2)
        edge_index   : (B, 2, max_edges)
        n_edges      : (B, 1)
        M_tilde      : (B, max_nodes, max_nodes)  analytic inertia coupling
        Returns      : (B, max_nodes, hidden)
        """
        B, max_nodes, hidden = h.shape
        max_edges = edge_index.shape[2]
        H, D = self.n_heads, self.d_head
        device = h.device

        src_idx = edge_index[:, 0, :]   # (B, max_edges)
        dst_idx = edge_index[:, 1, :]   # (B, max_edges)

        # Project nodes to Q, K, V
        Q = self.W_q(h).view(B, max_nodes, H, D)
        K = self.W_k(h).view(B, max_nodes, H, D)
        V = self.W_v(h).view(B, max_nodes, H, D)

        # Gather per-edge Q (dst), K (src), V (src)
        src_e = src_idx.unsqueeze(-1).unsqueeze(-1).expand(B, max_edges, H, D)
        dst_e = dst_idx.unsqueeze(-1).unsqueeze(-1).expand(B, max_edges, H, D)
        Q_e   = Q.gather(1, dst_e)    # (B, E, H, D)
        K_e   = K.gather(1, src_e)
        V_e   = V.gather(1, src_e)

        # Scaled dot-product logits
        logits = (Q_e * K_e).sum(-1) * (D ** -0.5)    # (B, E, H)

        # Learned edge-feature bias (existing technique)
        logits = logits + self.edge_bias(edge_features.float())  # (B, E, H)

        # ── Physics bias: β · M̃ᵢⱼ ──────────────────────────────────────────
        # Look up M̃[b, src, dst] for every edge (b, src→dst)
        # M_tilde: (B, max_nodes, max_nodes)
        # We need M_tilde[b, src_idx[b,e], dst_idx[b,e]] for each (b, e)
        src_flat = src_idx * max_nodes + dst_idx   # (B, E) — linear index into N×N
        M_flat   = M_tilde.view(B, max_nodes * max_nodes)
        M_edge   = M_flat.gather(1, src_flat)      # (B, E)  — M̃ per edge
        # Add as head-shared bias (broadcast across H).
        # tanh bounds β ∈ (-1, 1) — prevents attention from becoming hypersensitive
        # to M̃ values and destabilising when M̃ distribution shifts OOD.
        logits   = logits + torch.tanh(self.physics_beta) * M_edge.unsqueeze(-1)
        # ──────────────────────────────────────────────────────────────────────

        # Numerically stable scatter-softmax (log-sum-exp trick)
        edge_mask = (torch.arange(max_edges, device=device).unsqueeze(0)
                     < n_edges.squeeze(-1).unsqueeze(-1))            # (B, E)
        dst_h     = dst_idx.unsqueeze(-1).expand(B, max_edges, H)   # (B, E, H)

        logits    = logits.masked_fill(~edge_mask.unsqueeze(-1), float("-inf"))
        node_max  = torch.full((B, max_nodes, H), float("-inf"), device=device)
        node_max.scatter_reduce_(1, dst_h, logits, reduce="amax", include_self=True)
        node_max  = node_max.clamp(min=-1e9)
        max_edge  = node_max.gather(1, dst_h)

        logits_exp = (logits - max_edge).exp() * edge_mask.unsqueeze(-1).float()
        denom      = torch.zeros(B, max_nodes, H, device=device)
        denom.scatter_add_(1, dst_h, logits_exp)
        alpha      = logits_exp / denom.gather(1, dst_h).clamp(min=1e-6)  # (B, E, H)

        # Weighted value aggregation
        weighted = alpha.unsqueeze(-1) * V_e                         # (B, E, H, D)
        out      = torch.zeros(B, max_nodes, H, D, device=device)
        dst_hd   = dst_idx.unsqueeze(-1).unsqueeze(-1).expand(B, max_edges, H, D)
        out.scatter_add_(1, dst_hd, weighted)
        out      = self.W_o(out.reshape(B, max_nodes, hidden))

        h = self.norm1(h + out)
        h = self.norm2(h + self.ff(h))
        return h


# ═══════════════════════════════════════════════════════════════════════════════
# CGAT Encoder: node_embed → ICGA layers
# ═══════════════════════════════════════════════════════════════════════════════

class CGATEncoder(nn.Module):
    """
    node_embed  →  ICGA layers  →  masked mean pool

    Identical to GNNTransformerEncoder except each attention layer receives
    the analytic inertia coupling M̃ as an additional bias on the logits.
    """

    def __init__(self, hidden: int, n_icga_layers: int, n_heads: int):
        super().__init__()
        self.node_embed  = nn.Linear(NODE_FEAT_DIM, hidden)
        self.icga_layers = nn.ModuleList(
            [ICGALayer(hidden, n_heads) for _ in range(n_icga_layers)]
        )

    def forward(self, obs: dict, M_tilde: torch.Tensor) -> torch.Tensor:
        """Returns global graph embedding of shape (B, hidden)."""
        node_features = obs["node_features"].float()   # (B, max_nodes, 9)
        edge_index    = obs["edge_index"].long()        # (B, 2, max_edges)
        edge_features = obs["edge_features"].float()   # (B, max_edges, 2)
        n_nodes       = obs["n_nodes"].long()           # (B, 1)
        n_edges       = obs["n_edges"].long()           # (B, 1)

        B, max_nodes, _ = node_features.shape

        h = self.node_embed(node_features)              # (B, max_nodes, hidden)

        for layer in self.icga_layers:
            h = layer(h, edge_features, edge_index, n_edges, M_tilde)

        # Masked mean pool over real nodes
        node_mask = (torch.arange(max_nodes, device=h.device).unsqueeze(0)
                     < n_nodes.squeeze(-1).unsqueeze(-1))            # (B, max_nodes)
        h = h * node_mask.unsqueeze(-1).float()
        return h.sum(dim=1) / n_nodes.float()                        # (B, hidden)


# ═══════════════════════════════════════════════════════════════════════════════
# CGAT PPO Policy
# ═══════════════════════════════════════════════════════════════════════════════

class CGATPPOPolicy(BasePPOPolicy):
    """
    CGAT actor-critic — transformer + analytic inertia-coupling attention bias.

    Identical to GNNTransformerPPOPolicy except each attention layer adds
    β · M̃ᵢⱼ to the logits, where M̃ is the normalised Lagrangian mass matrix
    computed analytically from the observation every forward pass.
    """

    def __init__(self, hidden: int = 128, n_icga_layers: int = 2,
                 n_heads: int = 2, max_links: int = 4,
                 max_force: float = 20.0):
        super().__init__(hidden=hidden, max_force=max_force)
        self.encoder = CGATEncoder(hidden, n_icga_layers, n_heads)

    def encode(self, obs: dict) -> torch.Tensor:
        M_tilde = compute_inertia_coupling(obs)
        return self.encoder(obs, M_tilde)
