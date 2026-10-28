# -*- coding: utf-8 -*-
"""Stage II: RuleGuidedWorldKnowledgeEvolutionSimulator, Section 3.3.

Eq. (12) spatial ring aggregation       -> SpatialAggregation
Eq. (13) feature shuffling + pooling    -> blocks.StatePooling
Eq. (14) roll-forward simulation        -> RollForwardSimulation
Eq. (16) evolution consistency score    -> EvolutionConsistencyScorer
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..config import AblationConfig, DataConfig, ModelConfig
from .astsc import SparseStateTensor
from .blocks import StatePooling, TemporalConvolution


@dataclass
class SimulatorOutput:
    z_st: torch.Tensor                    # (B, c_out)   z_q^ST  (Eq. 13)
    g_hat: Optional[torch.Tensor]         # (B, |E|)     g_hat_q (Eq. 14), None if w/o Roll.
    rule_bonus: Optional[torch.Tensor]    # (B, |E|)     sum s(rule) * 1_rule(o)
    v_pairs: torch.Tensor                 # (M, c_out)   V[e,r,tau'_q,:]
    pair_ent: torch.Tensor                # (M,)
    pair_batch: torch.Tensor              # (M,)
    pair_key: Optional[torch.Tensor] = None
    gamma_mean: float = 1.0
    #: Eq. (15): the global context ``z_q^Agg`` (carried from Stage I).
    z_q_Agg: Optional[torch.Tensor] = None

    # ---- Table 1 aliases ------------------------------------------------
    @property
    def g_hat_q(self) -> Optional[torch.Tensor]:
        """Eq. (14): the rolled-forward activity ``g_hat_q[e]``."""
        return self.g_hat

    @property
    def z_q_ST(self) -> torch.Tensor:
        """Eq. (13): the feature-shuffled, mask-aware pooled ``z_q^{ST}``."""
        return self.z_st


class SpatialAggregation(nn.Module):
    """Eq. (12): V[e,r,:,:] = sum_varsigma psi_rq[varsigma] * U[e,r,varsigma,:,:]."""

    def __init__(self):
        super().__init__()

    @staticmethod
    def forward(u: torch.Tensor, traj_b: torch.Tensor, traj_e: torch.Tensor,
                traj_r: torch.Tensor, psi_f: torch.Tensor,
                num_entities: int, num_relations: int):
        key = (traj_b * num_entities + traj_e) * num_relations + traj_r
        uniq, inv = torch.unique(key, sorted=True, return_inverse=True)
        v = torch.zeros((uniq.numel(),) + tuple(u.shape[1:]), device=u.device, dtype=u.dtype)
        v.index_add_(0, inv, u * psi_f.view(-1, 1, 1))
        return v, uniq, inv


class RollForwardSimulation(nn.Module):
    """Eq. (14): g_hat_q[e] = MLP_roll([ sum_r phi_rq[r] V[e,r,tau'_q,:] || E_rel[rq] ]).

    ``phi_rq[r] ∝ exp(E_rel[r]^T W_phi E_rel[rq])`` is normalised over the
    relations that are actually present for entity ``e``; unseen entities fall
    back to the learned bias ``b_out``.
    """

    def __init__(self, c_out: int, rel_dim: int = 200, dropout: float = 0.1):
        super().__init__()
        self.W_phi = nn.Linear(rel_dim, rel_dim, bias=False)
        # V is a pooled spatio-temporal feature whose scale drifts with the
        # convolution stack; Eq. (14) feeds it straight into an MLP, so a
        # LayerNorm keeps the roll-forward logits identifiable.
        self.v_norm = nn.LayerNorm(c_out)
        self.mlp = nn.Sequential(
            nn.Linear(c_out + rel_dim, c_out), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(c_out, 1),
        )
        self.b_out = nn.Parameter(torch.zeros(1))

    def phi(self, rel_vec_pair: torch.Tensor, rel_vec_q: torch.Tensor,
            pair_key: torch.Tensor) -> torch.Tensor:
        """Normalised relation weights within every (batch, entity) group."""
        logits = (self.W_phi(rel_vec_pair) * rel_vec_q).sum(-1)      # (M,)
        uniq, inv = torch.unique(pair_key, sorted=True, return_inverse=True)
        mx = torch.full((uniq.numel(),), -1e30, device=logits.device, dtype=logits.dtype)
        mx.scatter_reduce_(0, inv, logits, reduce="amax", include_self=True)
        e = torch.exp(logits - mx[inv])
        den = torch.zeros_like(mx)
        den.index_add_(0, inv, e)
        return e / den[inv].clamp_min(1e-9)

    def forward(self, v_q: torch.Tensor, rel_vec_pair: torch.Tensor,
                rel_vec_q: torch.Tensor, pair_key: torch.Tensor,
                pair_batch: torch.Tensor, pair_ent: torch.Tensor,
                num_entities: int):
        phi = self.phi(rel_vec_pair, rel_vec_q[pair_batch], pair_key)  # (M,)
        uniq, inv = torch.unique(pair_key, sorted=True, return_inverse=True)
        v_ent = torch.zeros((uniq.numel(), v_q.shape[-1]), device=v_q.device, dtype=v_q.dtype)
        v_ent.index_add_(0, inv, v_q * phi.unsqueeze(-1))
        v_ent = self.v_norm(v_ent)

        b_ent = torch.zeros(uniq.numel(), dtype=torch.long, device=v_q.device)
        e_ent = torch.zeros(uniq.numel(), dtype=torch.long, device=v_q.device)
        b_ent.scatter_(0, inv, pair_batch)
        e_ent.scatter_(0, inv, pair_ent)

        h = torch.cat([v_ent, rel_vec_q[b_ent]], dim=-1)
        g = self.mlp(h).squeeze(-1)                                   # (U,)
        dense = self.b_out.expand(pair_batch.max().item() + 1 if pair_batch.numel() else 0,
                                  num_entities).clone()
        dense[b_ent, e_ent] = g
        return dense


class RuleGuidedWorldKnowledgeEvolutionSimulator(nn.Module):
    """Stage II: gated state tensor -> evolutionary representation + roll-forward."""

    def __init__(self, num_entities: int, num_relations: int, cfg: ModelConfig,
                 data_cfg: DataConfig):
        super().__init__()
        self.cfg = cfg
        self.data_cfg = data_cfg
        self.num_entities = int(num_entities)
        self.num_relations = int(num_relations)
        self.rel_emb = nn.Embedding(num_relations, cfg.embed_dim)
        nn.init.xavier_uniform_(self.rel_emb.weight)

        self.temporal = TemporalConvolution(
            c_in=cfg.c_in, c_out=cfg.c_out, kernel_size=cfg.kernel_size,
            dilations=cfg.dilations, num_layers=cfg.num_conv_layers,
            dropout=cfg.dropout, rel_dim=cfg.embed_dim,
        )
        self.spatial = SpatialAggregation()
        self.pooling = StatePooling(cfg.c_out, cfg.shuffle_groups)
        self.psi_mlp = nn.Sequential(
            nn.Linear(cfg.embed_dim, max(32, cfg.embed_dim // 2)), nn.GELU(),
            nn.Linear(max(32, cfg.embed_dim // 2), data_cfg.spatial_rings + 2),
        )
        self.roll = RollForwardSimulation(cfg.c_out, cfg.embed_dim, cfg.dropout)
        # Eq. (16) mixing weights (learnable, initialised from the config)
        self.roll_scale = nn.Parameter(torch.tensor(float(cfg.lambda_roll)))
        self.rule_scale = nn.Parameter(torch.tensor(float(cfg.lambda_rule)))

    # ------------------------------------------------------------------
    def forward(self, state: SparseStateTensor, batch,
                rule_scores: Optional[List[Dict[int, float]]] = None,
                ablations: Optional[AblationConfig] = None) -> SimulatorOutput:
        abl = ablations or AblationConfig()
        device = state.seq.device
        B = int(batch.src.shape[0])

        # w/o Evol.: Stage II degenerates to mean-pooled Stage-I features
        if abl.ablate_evol:
            return SimulatorOutput(z_st=state.z_agg, g_hat=None, rule_bonus=None,
                                   v_pairs=state.z_agg, pair_ent=torch.zeros(0, dtype=torch.long, device=device),
                                   pair_batch=torch.zeros(0, dtype=torch.long, device=device),
                                   pair_key=torch.zeros(0, dtype=torch.long, device=device),
                                   gamma_mean=float(state.fact_gamma.mean()),
                                   z_q_Agg=state.z_agg)

        rel_vec = self.rel_emb(state.traj_r)
        U = self._temporal_chunked(state, rel_vec, ablate=abl.ablate_st_conv)

        # ---- Eq. (12): spatial ring aggregation ----
        psi = torch.softmax(self.psi_mlp(self.rel_emb(batch.rel.to(device))), dim=-1)
        psi = psi[:, : state.num_spatial_buckets]
        psi_f = psi[state.traj_b, state.traj_s]
        V, pair_key, inv = self.spatial(
            U, state.traj_b, state.traj_e, state.traj_r, psi_f,
            self.num_entities, self.num_relations,
        )
        pair_b = (pair_key // (self.num_entities * self.num_relations)).long()
        pair_e = ((pair_key // self.num_relations) % self.num_entities).long()
        pair_r = (pair_key % self.num_relations).long()

        # ---- Eq. (13): feature shuffle + mask-aware average pooling ----
        mask_pair = torch.zeros((pair_key.numel(), state.num_time_buckets),
                                device=device, dtype=torch.float32)
        mask_pair.index_add_(0, inv, state.mask.squeeze(1).float())
        pooled = self.pooling(V, (mask_pair > 0).unsqueeze(1))         # (M, c_out)
        z_st = torch.zeros((B, self.cfg.c_out), device=device, dtype=pooled.dtype)
        z_st.index_add_(0, pair_b, pooled)
        z_st = z_st / torch.bincount(pair_b, minlength=B).to(
            dtype=pooled.dtype, device=device).clamp_min(1.0).view(B, 1)

        # ---- Eq. (14): roll-forward simulation ----
        g_hat = None
        if not abl.ablate_roll:
            v_q = V[:, :, state.tau_q]
            # Eq. (14): raw logit of query-time activity, supervised by the BCE
            # objective of Eq. (21) and added to Eq. (16) with weight lambda_roll.
            g_hat = self.roll(
                v_q, rel_vec_pair=self.rel_emb(pair_r), rel_vec_q=self.rel_emb(batch.rel.to(device)),
                pair_key=pair_key, pair_batch=pair_b, pair_ent=pair_e,
                num_entities=self.num_entities,
            )
            if g_hat.shape[0] < B:                       # pad empty rows in a batch
                g_hat = F.pad(g_hat, (0, 0, 0, B - g_hat.shape[0]))

        # ---- Eq. (16) rule term: sum_rules s(rule) * 1_rule(o) ----
        rule_bonus = None
        if rule_scores is not None and not abl.ablate_rule_term:
            rule_bonus = torch.zeros((B, self.num_entities), device=device, dtype=torch.float32)
            for b, scores in enumerate(rule_scores):
                if not scores:
                    continue
                idx = torch.as_tensor(list(scores.keys()), dtype=torch.long, device=device)
                val = torch.as_tensor(list(scores.values()), dtype=torch.float32, device=device)
                rule_bonus[b].index_add_(0, idx, val)

        return SimulatorOutput(
            z_st=z_st, g_hat=g_hat, rule_bonus=rule_bonus,
            v_pairs=V[:, :, state.tau_q], pair_ent=pair_e, pair_batch=pair_b,
            pair_key=pair_key, gamma_mean=float(state.fact_gamma.mean()),
            z_q_Agg=state.z_agg,
        )

    # ------------------------------------------------------------------
    def _temporal_chunked(self, state: SparseStateTensor, rel_vec: torch.Tensor,
                          ablate: bool = False) -> torch.Tensor:
        """Temporal convolution in trajectory chunks (memory guard, exact)."""
        P = state.seq.shape[0]
        chunk = max(1, int(getattr(self.cfg, "traj_chunk", 512)))
        if P <= chunk:
            return self.temporal(state.seq, state.mask, rel_vec, ablate=ablate)
        outs = []
        for i in range(0, P, chunk):
            sl = slice(i, min(i + chunk, P))
            outs.append(self.temporal(state.seq[sl], state.mask[sl], rel_vec[sl],
                                      ablate=ablate))
        return torch.cat(outs, dim=0)