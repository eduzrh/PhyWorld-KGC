# -*- coding: utf-8 -*-
"""Training objectives, Eqs. (20)-(22).

    L_kgc  = CE over all entities + lambda2 * (||h_s||^2 + ||h_r||^2 + ||h_o||^2)
    L_wm   = BCE on the masked query-bucket cells of X~            (Eq. 21)
    L_geo  = mean (||h_li - h_lj|| - xi * d_geo(l_i, l_j))^2       (Eq. 22)
    L      = L_kgc + mu * L_wm + nu * L_geo
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..config import ModelConfig
from ..utils import haversine_km


@dataclass
class LossOutput:
    total: torch.Tensor
    l_kgc: torch.Tensor
    l_wm: torch.Tensor
    l_geo: torch.Tensor

    @property
    def L_total(self) -> torch.Tensor:
        """Section 3.5: L_total = L_kgc + mu * L_wm + nu * L_geo."""
        return self.total

    @property
    def L_kgc(self) -> torch.Tensor:
        """Eq. (20)."""
        return self.l_kgc

    @property
    def L_wm(self) -> torch.Tensor:
        """Eq. (21)."""
        return self.l_wm

    @property
    def L_geo(self) -> torch.Tensor:
        """Eq. (22)."""
        return self.l_geo


class WorldModelHead(nn.Module):
    """Reconstructs the masked query-bucket cell V[s_q, r_q, tau_q] (Eq. 21)."""

    def __init__(self, c_out: int, embed_dim: int):
        super().__init__()
        self.proj = nn.Sequential(nn.Linear(c_out, c_out), nn.GELU(),
                                  nn.Linear(c_out, embed_dim))

    def forward(self, cell: torch.Tensor, ent_emb: nn.Embedding,
                gold: torch.Tensor) -> torch.Tensor:
        h = self.proj(cell)                                   # (B, d)
        logits = h @ ent_emb.weight.t()                       # (B, |E|)
        return F.cross_entropy(logits, gold)


class PhyWorldLoss(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.cfg = cfg
        self.wm_head = WorldModelHead(cfg.c_out, cfg.embed_dim)

    # ------------------------------------------------------------------
    def kgc(self, scores: torch.Tensor, gold: torch.Tensor,
            emb_parts: Optional[list] = None,
            h_reg: Optional[list] = None) -> torch.Tensor:
        """Eq. (20): CE + lambda_2 * (||h_s||^2 + ||h_r||^2 + ||h_o||^2).

        ``h_reg`` carries the *batch-level* representations, which is what the
        manuscript regularises; ``emb_parts`` is kept for callers that only have
        embedding tables available.
        """
        loss = F.cross_entropy(scores, gold)
        if self.cfg.l2_lambda > 0:
            if h_reg:
                reg = sum(h.pow(2).sum(dim=-1).mean() for h in h_reg if h is not None)
                loss = loss + self.cfg.l2_lambda * reg
            elif emb_parts:
                reg = sum((p ** 2).sum() for p in emb_parts if p is not None)
                loss = loss + self.cfg.l2_lambda * reg * 1e-3
        return loss

    # ------------------------------------------------------------------
    def wm(self, g_hat: Optional[torch.Tensor], gold: torch.Tensor,
           cell: Optional[torch.Tensor] = None,
           ent_emb: Optional[nn.Embedding] = None) -> torch.Tensor:
        """Eq. (21): BCE on the masked query-bucket (entity, query) pairs.

        ``g_hat[b, e]`` is the logit of entity ``e`` being active with the query
        subject at the query bucket.  The positive pair is the true object; the
        negative pair is drawn uniformly from the entity vocabulary.
        """
        if g_hat is None:
            if cell is None or ent_emb is None:
                return torch.zeros((), device=gold.device)
            return self.wm_head(cell, ent_emb, gold)
        n_ent = g_hat.shape[1]
        device = g_hat.device
        # BCE is scale-free: a logit of 50 already has ~0 loss, so nothing stops
        # g_hat from drifting.  Clamping keeps the gradient of the rolling term
        # bounded (it is the only unbounded additive term of Eq. (16)).
        g_hat = g_hat.clamp(-20.0, 20.0)
        pos = g_hat.gather(1, gold.view(-1, 1)).squeeze(1)
        neg_idx = torch.randint(0, n_ent, (gold.numel(),), device=device)
        neg_idx = torch.where(neg_idx == gold, (neg_idx + 1) % n_ent, neg_idx)
        neg = g_hat.gather(1, neg_idx.view(-1, 1)).squeeze(1)
        ones = torch.ones_like(pos)
        zeros = torch.zeros_like(neg)
        return 0.5 * (F.binary_cross_entropy_with_logits(pos, ones)
                      + F.binary_cross_entropy_with_logits(neg, zeros))

    # ------------------------------------------------------------------
    def geo(self, h_l: torch.Tensor, loc: torch.Tensor,
            valid: torch.Tensor) -> torch.Tensor:
        """Eq. (22): pull the location embedding norm towards xi * great-circle distance."""
        idx = torch.nonzero(valid).squeeze(-1)
        if idx.numel() < 4:
            return torch.zeros((), device=h_l.device)
        # cap the number of pairs for tractability (documented sampling)
        n = min(int(idx.numel()), 512)
        sel = idx[torch.randperm(idx.numel(), device=h_l.device)[:n]]
        a = sel[0::2]
        b = sel[1::2]
        m = min(a.numel(), b.numel())
        if m == 0:
            return torch.zeros((), device=h_l.device)
        a, b = a[:m], b[:m]
        d_pred = (h_l[a] - h_l[b]).norm(dim=-1)
        la, lb = loc[a].detach().cpu().numpy(), loc[b].detach().cpu().numpy()
        d_true = torch.as_tensor(
            [haversine_km(float(x1), float(y1), float(x2), float(y2))
             for (x1, y1), (x2, y2) in zip(la, lb)],
            dtype=torch.float32, device=h_l.device)
        return ((d_pred - self.cfg.geo_scale_xi * d_true) ** 2).mean()

    # ------------------------------------------------------------------
    def forward(self, scores: torch.Tensor, gold: torch.Tensor,
                wm_cell: Optional[torch.Tensor] = None,
                g_hat: Optional[torch.Tensor] = None,
                ent_emb: Optional[nn.Embedding] = None,
                h_l: Optional[torch.Tensor] = None,
                loc: Optional[torch.Tensor] = None,
                loc_valid: Optional[torch.Tensor] = None,
                emb_parts: Optional[list] = None,
                h_reg: Optional[list] = None,
                ablate_wm: bool = False, ablate_geo: bool = False) -> LossOutput:
        l_kgc = self.kgc(scores, gold, emb_parts, h_reg=h_reg)
        zero = torch.zeros((), device=scores.device)
        l_wm = zero if (ablate_wm or (g_hat is None and wm_cell is None)) \
            else self.wm(g_hat, gold, wm_cell, ent_emb)
        l_geo = zero if (ablate_geo or h_l is None or loc is None) \
            else self.geo(h_l, loc, loc_valid)
        total = l_kgc + self.cfg.mu * l_wm + self.cfg.nu * l_geo
        return LossOutput(total=total, l_kgc=l_kgc, l_wm=l_wm, l_geo=l_geo)