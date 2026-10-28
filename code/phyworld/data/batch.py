# -*- coding: utf-8 -*-
"""Batched query tensors fed to PhyWorld-KGC."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch

from ..sampling.rtns_st import SampledNeighborhood, SpatioTemporalNeighborSampler


@dataclass
class QueryBatch:
    """One mini-batch of queries plus their sampled neighbourhoods N_q (Eq. 1)."""

    src: torch.Tensor                 # (B,) subject entity
    rel: torch.Tensor                 # (B,) query relation (may be an inverse rel.)
    ts: torch.Tensor                  # (B,) query timestamp
    loc: torch.Tensor                 # (B, 2) query location (NaN if missing)
    gold: Optional[torch.Tensor]      # (B,) answer entity (None at test time)
    neigh_ent: torch.Tensor           # (B, K)
    neigh_rel: torch.Tensor           # (B, K)
    neigh_ts: torch.Tensor            # (B, K)
    neigh_lat: torch.Tensor           # (B, K)
    neigh_lon: torch.Tensor           # (B, K)
    neigh_dist: torch.Tensor          # (B, K)
    neigh_text: Optional[torch.Tensor] = None   # (B, K) text-location ids, -1 = none
    src_text: Optional[torch.Tensor] = None     # (B,)
    coocc: Optional[torch.Tensor] = None        # (B,) bool
    #: Table 1: N_q^all (ragged list of neighbour fact ids) and N_q (sampled ids).
    N_q_all: Optional[list] = None
    N_q: Optional[torch.Tensor] = None
    setting: str = "interpolation"
    meta: Dict = field(default_factory=dict)

    def to(self, device) -> "QueryBatch":
        for name in ("src", "rel", "ts", "loc", "gold", "neigh_ent", "neigh_rel",
                     "neigh_ts", "neigh_lat", "neigh_lon", "neigh_dist",
                     "neigh_text", "src_text", "coocc"):
            v = getattr(self, name, None)
            if torch.is_tensor(v):
                setattr(self, name, v.to(device))
        return self

    def __len__(self) -> int:
        return int(self.src.shape[0])


class QueryBatcher:
    """Turns ``(s, r, t)`` queries into :class:`QueryBatch` tensors."""

    def __init__(self, ds, index, sampler: SpatioTemporalNeighborSampler,
                 fact_text_id: Optional[np.ndarray] = None,
                 cooccurring: Optional[Sequence[int]] = None,
                 grounder=None):
        self.ds = ds
        self.index = index
        self.sampler = sampler
        self.fact_text_id = fact_text_id
        self.cooccurring = set(int(r) for r in (cooccurring or []))
        self.grounder = grounder

    # ------------------------------------------------------------------
    def build(self, queries: Sequence[Tuple[int, int, float]],
              gold: Optional[Sequence[int]] = None,
              setting: str = "interpolation") -> QueryBatch:
        src = np.asarray([q[0] for q in queries], dtype=np.int64)
        rel = np.asarray([q[1] for q in queries], dtype=np.int64)
        ts = np.asarray([int(q[2]) for q in queries], dtype=np.int64)
        loc = np.asarray([self.index.query_location(int(q[0]), int(q[2]))
                          for q in queries], dtype=np.float64).reshape(-1, 2)

        nb = self.sampler.sample(src, rel, ts, loc)
        B, K = nb.ent.shape

        text = np.full((B, K), -1, dtype=np.int64)
        if self.fact_text_id is not None and nb.fidx is not None:
            ok = nb.fidx >= 0
            text[ok] = self.fact_text_id[nb.fidx[ok]]

        src_text = np.asarray([self.index.ent_text_id[int(s)] for s in src],
                              dtype=np.int64)

        return QueryBatch(
            src=torch.as_tensor(src),
            rel=torch.as_tensor(rel),
            ts=torch.as_tensor(ts, dtype=torch.float32),
            loc=torch.as_tensor(loc, dtype=torch.float32),
            gold=(torch.as_tensor(np.asarray(gold, dtype=np.int64)) if gold is not None else None),
            neigh_ent=torch.as_tensor(nb.ent),
            neigh_rel=torch.as_tensor(nb.rel),
            neigh_ts=torch.as_tensor(nb.ts, dtype=torch.float32),
            neigh_lat=torch.as_tensor(nb.lat, dtype=torch.float32),
            neigh_lon=torch.as_tensor(nb.lon, dtype=torch.float32),
            neigh_dist=torch.as_tensor(nb.dist, dtype=torch.float32),
            neigh_text=torch.as_tensor(text),
            src_text=torch.as_tensor(src_text),
            coocc=torch.as_tensor(np.asarray([r in self.cooccurring for r in rel])),
            # Table 1: N_q^all (all neighbours of s_q, ragged) and N_q (sampled).
            N_q_all=getattr(nb, "N_q_all", None),
            N_q=torch.as_tensor(nb.fidx),
            setting=setting,
        )