# -*- coding: utf-8 -*-
"""Stage I: AdaptiveSpatioTemporalStateConstructor (ASTSC), Section 3.2.

Implements
  * Eq. (1) sampling                -> phyworld/sampling/rtns_st.py
  * Eq. (2)-(4) facet encoding/gate -> Encoder_e_r_t_l, CrossAttention, Gate
  * Eq. (5)-(7) state tensor X      -> SparseStateTensor (sparse, Appendix A.2)
  * Eq. (9) rule gate table         -> GateTable (consumed by Stage II)
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from ..config import AblationConfig, DataConfig, ModelConfig
from ..rules.base import RuleBase


# ---------------------------------------------------------------------------
# sparse state tensor
# ---------------------------------------------------------------------------

@dataclass
class SparseStateTensor:
    """Sparse view of ``X in R^{|E_q| x |R_q| x |S_q| x |T_q| x C_in}``.

    Only the ``k`` activated cells are materialised (Appendix A.2):
      seq      : (P, C_in, T)  trajectories (e, r, varsigma) over time
      mask     : (P, 1, T)     which time buckets carry at least one fact
      traj_b/e/r/s : (P,)      batch / entity / relation / spatial-bucket ids
      z_agg    : (B, C_in)     Eq. (15) global context
    """

    seq: torch.Tensor
    mask: torch.Tensor
    traj_b: torch.Tensor
    traj_e: torch.Tensor
    traj_r: torch.Tensor
    traj_s: torch.Tensor
    z_agg: torch.Tensor
    fact_features: torch.Tensor
    fact_gamma: torch.Tensor
    fact_hl: torch.Tensor
    fact_loc: torch.Tensor
    fact_loc_valid: torch.Tensor
    tau: torch.Tensor
    s_bucket: torch.Tensor
    num_time_buckets: int
    num_spatial_buckets: int
    tau_q: int

    # ---- Table 1 aliases ------------------------------------------------
    @property
    def X(self) -> torch.Tensor:
        """Eq. (7): the feature payload of the five-order tensor X, per fact.

        ``X`` is never materialised densely; ``fact_features`` holds one row per
        *activated* cell ``(e, r, varsigma, tau)`` -- exactly the ``k`` live
        entries prescribed by Appendix A.2.
        """
        return self.fact_features

    @property
    def X_tilde(self) -> torch.Tensor:
        """Eq. (10): X~ = Gamma . X, elementwise on the activated cells."""
        return self.fact_features * self.fact_gamma.unsqueeze(-1)


# ---------------------------------------------------------------------------
# facet encoders (Encoder_e_r_t_l)
# ---------------------------------------------------------------------------

class SinusoidalTimeEncoding(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim

    def forward(self, dt: torch.Tensor) -> torch.Tensor:
        device = dt.device
        half = max(1, self.dim // 2)
        freqs = torch.exp(-math.log(10000.0) * torch.arange(half, device=device,
                                                             dtype=torch.float32) / half)
        x = dt.float().unsqueeze(-1) * freqs.unsqueeze(0)
        emb = torch.cat([torch.sin(x), torch.cos(x)], dim=-1)
        if emb.shape[-1] < self.dim:
            emb = F.pad(emb, (0, self.dim - emb.shape[-1]))
        return emb


class CrossAttention(nn.Module):
    """Eq. (2): H_bar = H + softmax(H W_Q (H W_K)^T / sqrt(d)) H W_V."""

    def __init__(self, dim: int, num_heads: int = 1):
        super().__init__()
        self.dim = dim
        self.num_heads = max(1, num_heads)
        self.Wq = nn.Linear(dim, dim, bias=False)
        self.Wk = nn.Linear(dim, dim, bias=False)
        self.Wv = nn.Linear(dim, dim, bias=False)

    def forward(self, H: torch.Tensor) -> torch.Tensor:
        B, M, d = H.shape
        q, k, v = self.Wq(H), self.Wk(H), self.Wv(H)
        attn = torch.softmax(q @ k.transpose(-2, -1) / math.sqrt(d), dim=-1)
        return H + attn @ v


class Gate(nn.Module):
    """Eq. (3): facet uncertainty u(m) = ||h_bar(m) - h(m)||, softmax gate a(m)."""

    def __init__(self, temperature: float = 0.1):
        super().__init__()
        self.temperature = float(temperature)

    def forward(self, H: torch.Tensor, H_bar: torch.Tensor) -> torch.Tensor:
        u = (H_bar - H).norm(dim=-1)                     # (N, 4)
        return torch.softmax(-u / max(self.temperature, 1e-6), dim=-1)


class Encoder_e_r_t_l(nn.Module):
    """Four-facet encoder producing ``c(e, r, t, l)`` (Eqs. 2-4)."""

    def __init__(self, num_entities: int, num_relations: int, embed_dim: int = 200,
                 c_in: int = 128, text_dim: int = 768, num_texts: int = 0,
                 gate_temperature: float = 0.1, attn_heads: int = 1,
                 use_text: bool = True, dropout: float = 0.1):
        super().__init__()
        self.d = int(embed_dim)
        self.c_in = int(c_in)
        self.use_text = bool(use_text) and num_texts > 0

        # h_e, h_r: learnable, initialised from pretrained static KG embeddings
        self.ent_emb = nn.Embedding(num_entities, self.d)
        self.rel_emb = nn.Embedding(num_relations, self.d)
        nn.init.xavier_uniform_(self.ent_emb.weight)
        nn.init.xavier_uniform_(self.rel_emb.weight)

        # h_t: sinusoidal encoding of (t - t_q) followed by a linear map
        self.time_enc = SinusoidalTimeEncoding(self.d)
        self.time_proj = nn.Linear(self.d, self.d)

        # h_l: dual-branch geo encoder (coordinates + location text)
        self.coord_mlp = nn.Sequential(
            nn.Linear(2, self.d), nn.ReLU(inplace=True), nn.Linear(self.d, self.d)
        )
        self.text_table = None
        self.text_proj = None
        if self.use_text:
            self.text_table = nn.Embedding(num_texts, text_dim)
            self.text_table.weight.requires_grad_(False)
            self.text_proj = nn.Linear(text_dim, self.d)
        self.h_empty = nn.Parameter(torch.randn(self.d) * 0.02)

        self.cross_attn = CrossAttention(self.d, attn_heads)
        self.gate = Gate(gate_temperature)
        self.fuse = nn.Sequential(
            nn.Linear(4 * self.d, 2 * self.d), nn.GELU(),
            nn.Dropout(dropout), nn.Linear(2 * self.d, self.c_in),
        )

    def encode_locations(self, lat: torch.Tensor, lon: torch.Tensor,
                         text_id: Optional[torch.Tensor]) -> torch.Tensor:
        """Dual-branch geo-encoder; unresolved location -> learnable h_empty."""
        B = lat.shape[0]
        device = lat.device
        has_coord = ~torch.isnan(lat)
        hl = torch.zeros(B, self.d, device=device, dtype=torch.float32)

        if bool(has_coord.any()):
            coords = torch.stack([torch.nan_to_num(lat), torch.nan_to_num(lon)], dim=-1)
            hl = hl + torch.where(has_coord.unsqueeze(-1),
                                  self.coord_mlp(coords.to(torch.float32)),
                                  torch.zeros_like(hl))

        has_text = torch.zeros_like(has_coord)
        if self.use_text and text_id is not None:
            has_text = text_id >= 0
            if bool(has_text.any()):
                tid = torch.where(has_text, text_id, torch.zeros_like(text_id))
                hl = hl + torch.where(has_text.unsqueeze(-1),
                                      self.text_proj(self.text_table(tid)),
                                      torch.zeros_like(hl))

        resolved = has_coord | has_text
        hl = torch.where(resolved.unsqueeze(-1), hl, self.h_empty.expand(B, self.d))
        # bounded hyperspherical embedding: ||h_l|| = 10, so that the L_geo target
        # xi * d_geo stays on the same scale as the embedding distance.
        # ``clamp_min`` keeps the gradient finite when a fact has no location at
        # all (otherwise 1/||h_l|| blows up at the origin).
        return 10.0 * hl / hl.norm(dim=-1, keepdim=True).clamp_min(1e-2)

    def forward(self, ent: torch.Tensor, rel: torch.Tensor, dt: torch.Tensor,
                lat: torch.Tensor, lon: torch.Tensor,
                text_id: Optional[torch.Tensor] = None,
                ablate_loc: bool = False, ablate_gate: bool = False):
        he = self.ent_emb(ent)
        hr = self.rel_emb(rel)
        ht = self.time_proj(self.time_enc(dt))
        hl = torch.zeros_like(he) if ablate_loc else self.encode_locations(lat, lon, text_id)

        H = torch.stack([he, hr, ht, hl], dim=1)          # (N, 4, d)
        H_bar = self.cross_attn(H)

        if ablate_gate:                                    # w/o Gate: fixed weights
            a = torch.full((H.shape[0], 4), 0.25, device=H.device, dtype=H.dtype)
        else:
            a = self.gate(H, H_bar)

        feat = (a.unsqueeze(-1) * H_bar).reshape(H.shape[0], 4 * self.d)
        return self.fuse(feat), hl# ---------------------------------------------------------------------------
# rule gate table (Eq. 9)
# ---------------------------------------------------------------------------

class GateTable:
    """Gamma[r, varsigma, tau] per head relation, cached (Eq. 9).

    ``Gamma`` depends only on the query *relation* (not on s_q / t_q), so it is
    precomputed once per relation and gathered per fact at training time.

        Gamma[r, varsigma, tau] = eps + (1 - eps) * max_rule max_{i: r_i = r}
                                  s(rule) * g_i(varsigma, tau)
        g_i = 1[|tau_q - tau| * Delta <= omega_i] * 1[d_bar_varsigma <= rho_i]
    """

    def __init__(self, rule_base: RuleBase, num_relations: int, num_spatial: int,
                 num_time: int, tau_q: int, bin_width: float,
                 ring_distance: np.ndarray, floor: float = 0.2,
                 ablate_rule_gate: bool = False, ablate_s_const: bool = False):
        self.num_relations = int(num_relations)
        self.S = int(num_spatial)
        self.T = int(num_time)
        self.tau_q = int(tau_q)
        self.bin_width = float(bin_width)
        rd = np.asarray(ring_distance, dtype=np.float32).reshape(-1)
        if rd.size < self.S:
            fill = float(rd.max()) if rd.size else 0.0
            rd = np.concatenate([rd, np.full(self.S - rd.size, fill, dtype=np.float32)])
        self.ring_distance = rd[: self.S]
        self.floor = float(floor)
        self.ablate_rule_gate = bool(ablate_rule_gate)
        self.ablate_s_const = bool(ablate_s_const)
        self.rule_base = rule_base
        self.table: Dict[int, np.ndarray] = {}

    def build(self, head_rel: int) -> np.ndarray:
        if head_rel in self.table:
            return self.table[head_rel]

        if self.ablate_rule_gate:                      # w/o RuleGate: Gamma == 1
            g = np.ones((self.num_relations, self.S, self.T), dtype=np.float32)
            self.table[head_rel] = g
            return g

        g = np.full((self.num_relations, self.S, self.T), self.floor, dtype=np.float32)
        taus = np.arange(self.T, dtype=np.float32)
        dt = np.abs(taus - self.tau_q) * self.bin_width            # (T,)

        for rule in self.rule_base.for_relation(head_rel):
            atoms = list(rule.atoms) + ([rule.branch] if rule.branch is not None else [])
            for atom in atoms:
                tm = (dt <= atom.window).astype(np.float32)
                if self.ablate_s_const:                            # w/o S.Const
                    sm = np.ones(self.S, dtype=np.float32)
                else:
                    sm = (self.ring_distance <= atom.radius).astype(np.float32)
                    if self.S > 0:
                        sm[-1] = 1.0                               # A_empty passes
                gi = rule.s_rule * np.outer(sm, tm).astype(np.float32)
                np.maximum(g[atom.rel], gi, out=g[atom.rel])

        g = self.floor + (1.0 - self.floor) * g
        self.table[head_rel] = g
        return g

    def gamma(self, head_rel: int) -> np.ndarray:
        return self.build(head_rel)


class RuleGating(GateTable):
    """Eq. (9)-(10): the rule-gating module of Stage II.

    It owns the gate tensor ``Gamma[r, varsigma, tau]`` (Eq. 9) and applies it to
    the state tensor (Eq. 10).  ``GateTable`` keeps the caching machinery; this
    class gives the module its paper-level name and the explicit ``X -> X~``
    transition.
    """

    def modulate(self, X: torch.Tensor, Gamma: torch.Tensor) -> torch.Tensor:
        """Eq. (10): ``X~ = Gamma . X`` for the activated cells of a batch."""
        g = Gamma.unsqueeze(-1) if Gamma.dim() == X.dim() - 1 else Gamma
        return X * g# ---------------------------------------------------------------------------
# Stage I module
# ---------------------------------------------------------------------------

class AdaptiveSpatioTemporalStateConstructor(nn.Module):
    """Stage I: heterogeneous facts -> query-centric spatio-temporal state tensor X."""

    def __init__(self, num_entities: int, num_relations: int, cfg: ModelConfig,
                 data_cfg: DataConfig, num_texts: int = 0, text_dim: int = 768,
                 ring_distance: Optional[np.ndarray] = None):
        super().__init__()
        self.cfg = cfg
        self.data_cfg = data_cfg
        d = cfg.facet_dim or cfg.embed_dim
        self.encoder = Encoder_e_r_t_l(
            num_entities=num_entities, num_relations=num_relations, embed_dim=d,
            c_in=cfg.c_in, text_dim=text_dim, num_texts=num_texts,
            gate_temperature=cfg.gate_temperature, attn_heads=cfg.attn_heads,
            use_text=cfg.use_text_encoder, dropout=cfg.dropout,
        )
        self.c_in = cfg.c_in
        # Static-embedding path: a single linear projection of the
        # [E_ent[e] || E_rel[r]] pair (the static KG embeddings pre-trained on the
        # training graph, Section 3.2).  It is added to Eq. (4)'s facet code as a
        # residual so that the pre-trained signal reaches Stage II at step 0.
        self.static_path = bool(getattr(cfg, "static_path", True))
        self.static_proj = (nn.Linear(2 * d, cfg.c_in)
                            if self.static_path else None)
        self.num_entities = int(num_entities)
        self.num_relations = int(num_relations)
        self.beta = 1.0
        self.beta_s = 1.0
        self.ring_distance = (np.asarray(ring_distance, dtype=np.float32)
                              if ring_distance is not None else np.zeros(1, dtype=np.float32))

        # Eq. (6) relation-dependent decay rates gamma_r and eta_r (softplus > 0)
        self.gamma_raw = nn.Parameter(torch.zeros(self.num_relations))
        self.eta_raw = nn.Parameter(torch.zeros(self.num_relations))
        self.register_buffer("ring_distance_t",
                             torch.as_tensor(self.ring_distance, dtype=torch.float32))

    # ------------------------------------------------------------------
    def init_embeddings(self, ent: torch.Tensor, rel: torch.Tensor) -> None:
        with torch.no_grad():
            n_e = min(self.encoder.ent_emb.weight.shape[0], ent.shape[0])
            n_r = min(self.encoder.rel_emb.weight.shape[0], rel.shape[0])
            self.encoder.ent_emb.weight[:n_e].copy_(ent[:n_e])
            self.encoder.rel_emb.weight[:n_r].copy_(rel[:n_r])

    def register_decay_priors(self, alpha: np.ndarray, zeta: np.ndarray) -> None:
        """Initialise gamma_r / eta_r from alpha_r = 1/mu_r and zeta_r = 1/rho_r."""
        def inv_softplus(x):
            return np.log(np.expm1(np.clip(x, 1e-4, 50.0)))

        with torch.no_grad():
            self.gamma_raw.copy_(torch.as_tensor(inv_softplus(alpha), dtype=torch.float32))
            self.eta_raw.copy_(torch.as_tensor(inv_softplus(zeta), dtype=torch.float32))

    # ------------------------------------------------------------------
    def forward(self, batch, gamma_table: Optional[GateTable] = None,
                ablations: Optional[AblationConfig] = None) -> SparseStateTensor:
        """Build X for a batch of queries (Eqs. 5-7)."""
        abl = ablations or AblationConfig()
        device = batch.src.device
        B = batch.src.shape[0]
        K = batch.neigh_ent.shape[1]
        m = int(self.data_cfg.window_past)
        n = int(self.data_cfg.future_window())
        T = int(self.data_cfg.time_buckets())
        tau_q = int(m // self.data_cfg.bin_width)

        ent = batch.neigh_ent.to(device)
        rel = batch.neigh_rel.to(device)
        ts = batch.neigh_ts.to(device)
        dt = ts - batch.ts.view(B, 1)
        valid = (ent >= 0) & (dt >= -m) & (dt <= n)
        b_idx, k_idx = torch.nonzero(valid, as_tuple=True)

        e_f = ent[b_idx, k_idx]
        r_f = rel[b_idx, k_idx]
        dt_f = dt[b_idx, k_idx]
        tau_f = torch.clamp(((dt_f + m) // self.data_cfg.bin_width).long(), 0, T - 1)

        lat_f = batch.neigh_lat.to(device)[b_idx, k_idx]
        lon_f = batch.neigh_lon.to(device)[b_idx, k_idx]
        dist_f = batch.neigh_dist.to(device)[b_idx, k_idx]

        # ---- spatial discretisation (Section 3.2) ----
        if abl.ablate_s_bucket:                        # w/o S.Buck.: single bucket
            S = 1
            s_f = torch.zeros_like(tau_f)
        else:
            S = int(self.data_cfg.spatial_rings) + 1   # rings + A_empty
            res = float(self.data_cfg.spatial_resolution_km)
            ring = torch.floor(torch.nan_to_num(dist_f / res, nan=float("inf")))
            ring = torch.clamp(ring, 0, self.data_cfg.spatial_rings - 1)
            s_f = torch.where(torch.isnan(dist_f), torch.full_like(ring, S - 1), ring).long()

        # ---- facet encoding (Eq. 2-4) ----
        fact_text = getattr(batch, "neigh_text", None)
        text_id = fact_text.to(device)[b_idx, k_idx] if fact_text is not None else None
        c_fact, hl = self.encoder(
            e_f, r_f, dt_f.float(), lat_f, lon_f, text_id=text_id,
            ablate_loc=abl.ablate_loc, ablate_gate=abl.ablate_gate,
        )
        if self.static_proj is not None:
            c_fact = c_fact + self.static_proj(torch.cat(
                [self.encoder.ent_emb(e_f), self.encoder.rel_emb(r_f)], dim=-1))

        # ---- slice weights sigma[tau, varsigma] (Eq. 6) ----
        counts = torch.zeros((B, T, S), device=device, dtype=torch.float32)
        counts.index_put_((b_idx, tau_f, s_f),
                          torch.ones_like(tau_f, dtype=torch.float32), accumulate=True)

        g_rate = F.softplus(self.gamma_raw)[batch.rel.to(device)]     # (B,)
        e_rate = F.softplus(self.eta_raw)[batch.rel.to(device)]       # (B,)

        taus = torch.arange(T, device=device, dtype=torch.float32)
        dist_tau = torch.abs(taus - tau_q).pow(self.beta)             # (T,)
        d_ring = self.ring_distance_t
        if d_ring.numel() < S:
            d_ring = F.pad(d_ring, (0, S - d_ring.numel()),
                           value=float(self.data_cfg.spatial_rings)
                           * float(self.data_cfg.spatial_resolution_km))
        dist_ring = d_ring[:S].pow(self.beta_s)                       # (S,)

        logw = (-g_rate.view(B, 1, 1) * dist_tau.view(1, T, 1)
                - e_rate.view(B, 1, 1) * dist_ring.view(1, 1, S))
        w = torch.exp(logw) / counts.clamp_min(1.0)
        w = w * (counts > 0).float()
        sigma = w / w.sum(dim=(1, 2), keepdim=True).clamp_min(1e-9)  # (B,T,S)
        w_f = sigma[b_idx, tau_f, s_f]                                # (N,)

        # ---- rule gate Gamma (Eq. 9-10) ----
        if gamma_table is not None:
            gamma_f = self._gather_gamma(gamma_table, batch.rel, b_idx, r_f, s_f, tau_f)
        else:
            gamma_f = torch.ones_like(w_f)
        # Eq. (10): X~ = Gamma . X, evaluated only on the activated cells.
        x_tilde = c_fact * gamma_f.unsqueeze(-1)
        feat_w = w_f.unsqueeze(-1)

        # ---- trajectory assembly (Eq. 7, sparse per Appendix A.2) ----
        # X is never materialised densely: trajectories are keyed by the *actual*
        # (e, r, varsigma) ids, so |P| <= |N_q^all| and cost stays linear in k.
        max_E = max(1, self.num_entities)
        max_R = max(1, self.num_relations)
        key = ((b_idx * max_E + e_f) * max_R + r_f) * S + s_f
        uniq, inv = torch.unique(key, sorted=False, return_inverse=True)
        P = int(uniq.numel())

        seq2d = torch.zeros((P * T, self.c_in), device=device, dtype=c_fact.dtype)
        flat = (inv * T + tau_f).to(torch.long)
        seq2d.scatter_add_(0, flat.view(-1, 1).expand(-1, self.c_in), x_tilde * feat_w)
        seq = seq2d.view(P, T, self.c_in).permute(0, 2, 1).contiguous()
        mask = torch.zeros((P * T,), device=device, dtype=torch.bool)
        mask.index_put_((flat,), torch.ones_like(flat, dtype=torch.bool), accumulate=True)
        mask = mask.view(P, 1, T)

        b_traj = (uniq // (max_E * max_R * S)).long()
        rem = uniq % (max_E * max_R * S)
        e_traj = (rem // (max_R * S)).long()
        r_traj = ((rem // S) % max_R).long()
        s_traj = (rem % S).long()

        # ---- global context z_q^Agg (Eq. 15) ----
        z_agg = torch.zeros((B, self.c_in), device=device, dtype=c_fact.dtype)
        z_agg.index_add_(0, b_idx, x_tilde * feat_w)
        z_agg = z_agg / torch.bincount(b_idx, minlength=B).to(
            dtype=c_fact.dtype, device=device).clamp_min(1.0).view(B, 1)

        return SparseStateTensor(
            seq=seq, mask=mask, traj_b=b_traj, traj_e=e_traj, traj_r=r_traj, traj_s=s_traj,
            z_agg=z_agg, fact_features=c_fact, fact_gamma=gamma_f,
            fact_hl=hl, fact_loc=torch.stack([lat_f, lon_f], dim=-1),
            fact_loc_valid=~torch.isnan(lat_f),
            tau=tau_f, s_bucket=s_f, num_time_buckets=T, num_spatial_buckets=S,
            tau_q=tau_q,
        )

    # ------------------------------------------------------------------
    def _gather_gamma(self, table: GateTable, rel_q: torch.Tensor,
                      b_idx: torch.Tensor, r_f: torch.Tensor,
                      s_f: torch.Tensor, tau_f: torch.Tensor) -> torch.Tensor:
        out = torch.ones_like(tau_f, dtype=torch.float32)
        rel_q_cpu = rel_q.detach().cpu().numpy()
        b_cpu = b_idx.detach().cpu().numpy()
        r_cpu = r_f.detach().cpu().numpy()
        s_cpu = s_f.detach().cpu().numpy()
        t_cpu = tau_f.detach().cpu().numpy()
        for q in np.unique(b_cpu).tolist():
            tab = table.build(int(rel_q_cpu[q]))
            sel = b_cpu == q
            out[sel] = torch.as_tensor(tab[r_cpu[sel], s_cpu[sel], t_cpu[sel]],
                                       device=out.device)
        return out