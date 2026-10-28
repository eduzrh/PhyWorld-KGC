# -*- coding: utf-8 -*-
"""Relation-adaptive spatio-temporal neighbour sampling (Eq. 1).

Symbols (Table 1):
  N_q^all : all neighbour facts of s_q
  N_q     : sampled neighbourhood (STNG), |N_q| = k
  psi_q   : spatio-temporal decay,  alpha_r |t_q - t|^beta + zeta_r d^(l_q, l)^beta_s
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple

import numpy as np

from ..data.stkg import STKGDataset, STKGIndex
from ..utils import haversine_km


@dataclass
class SampledNeighborhood:
    """The STNG N_q for a batch of queries.

    ent  : (B, K) neighbour entity        (``-1`` = padding)
    rel  : (B, K) relation
    ts   : (B, K) timestamp
    lat  : (B, K) latitude   (NaN if missing)
    lon  : (B, K) longitude
    dist : (B, K) spatial distance d^(l_q, l) in km (NaN if either side missing)
    """

    ent: np.ndarray
    rel: np.ndarray
    ts: np.ndarray
    lat: np.ndarray
    lon: np.ndarray
    dist: np.ndarray
    fidx: Optional[np.ndarray] = None      # index into the source fact table
    #: Table 1: N_q^all -- every neighbour fact of s_q before sampling (ragged).
    N_q_all: Optional[list] = None
    #: Table 1: N_q -- the sampled neighbourhood, as fact indices (-1 = padding).
    N_q: Optional[np.ndarray] = None

    def valid_mask(self) -> np.ndarray:
        return self.ent >= 0


class SpatioTemporalNeighborSampler:
    """Eq. (1): p((e,r,t,l) | q) ∝ exp(-psi_q(t, l))."""

    def __init__(
        self,
        ds: STKGDataset,
        index: STKGIndex,
        k: int = 50,
        beta: float = 1.0,
        beta_s: float = 1.0,
        sampling: str = "st_rtns",
        deterministic: bool = False,
        seed: int = 42,
        time_mode: str = "auto",
        ablate_s_sample: bool = False,
    ):
        self.ds = ds
        self.index = index
        self.k = int(k)
        self.beta = float(beta)
        self.beta_s = float(beta_s)
        self.sampling = sampling
        self.deterministic = bool(deterministic)
        self.rng = np.random.RandomState(seed)
        self.ablate_s_sample = bool(ablate_s_sample)

        self.alpha = ds.alpha_prior()                 # 1/mu_r  (base + inverse)
        rho = ds.relation_rho
        self.zeta = np.concatenate([1.0 / np.maximum(rho, 1e-3)] * 2)  # 1/rho_r
        self.rho = np.concatenate([rho, rho])

        if time_mode == "auto":
            time_mode = "past" if ds.setting == "extrapolation" else "all"
        self.time_mode = time_mode

    # ------------------------------------------------------------------
    def _query_location(self, s_q: int, t_q: int) -> Tuple[float, float]:
        """l_q fallback: nearest available location of s_q around t_q."""
        best, best_dt = None, None
        for (o, r, t, idx) in self.index.adj_out[s_q]:
            lat, lon = self.index.fact_location(idx)
            if np.isnan(lat):
                continue
            dt = abs(t - t_q)
            if best_dt is None or dt < best_dt:
                best_dt, best = dt, (lat, lon)
        if best is None:
            return np.nan, np.nan
        return best

    def _candidates(self, s_q: int, t_q: int):
        neigh = self.index.adj_out[s_q]
        if not neigh:
            return None
        ent = np.fromiter((x[0] for x in neigh), dtype=np.int64, count=len(neigh))
        rel = np.fromiter((x[1] for x in neigh), dtype=np.int64, count=len(neigh))
        ts = np.fromiter((x[2] for x in neigh), dtype=np.int64, count=len(neigh))
        idx = np.fromiter((x[3] for x in neigh), dtype=np.int64, count=len(neigh))
        if self.time_mode == "past":
            keep = ts < t_q
        elif self.time_mode == "past_inclusive":
            keep = ts <= t_q
        else:
            keep = np.ones_like(ts, dtype=bool)
        if not keep.any():
            return None
        return ent[keep], rel[keep], ts[keep], idx[keep]

    def sample(self, src: Sequence[int], rel_q: Sequence[int], ts_q: Sequence[int],
               loc_q: Optional[Sequence[Sequence[float]]] = None) -> SampledNeighborhood:
        B = len(src)
        K = self.k
        ent = np.full((B, K), -1, dtype=np.int64)
        rel = np.full((B, K), -1, dtype=np.int64)
        ts = np.zeros((B, K), dtype=np.int64)
        lat = np.full((B, K), np.nan, dtype=np.float64)
        lon = np.full((B, K), np.nan, dtype=np.float64)
        dist = np.full((B, K), np.nan, dtype=np.float64)
        fidx = np.full((B, K), -1, dtype=np.int64)
        all_idx: list = []                     # N_q^all per query

        for b in range(B):
            s_q, r_q, t_q = int(src[b]), int(rel_q[b]), int(ts_q[b])
            cand = self._candidates(s_q, t_q)
            if cand is None:
                all_idx.append(np.zeros(0, dtype=np.int64))
                continue
            c_ent, c_rel, c_ts, c_idx = cand
            all_idx.append(c_idx)              # N_q^all

            # query location
            if loc_q is not None and not np.isnan(loc_q[b][0]):
                q_lat, q_lon = float(loc_q[b][0]), float(loc_q[b][1])
            else:
                q_lat, q_lon = self._query_location(s_q, t_q)

            locs = self.ds.train.loc_latlon[c_idx]
            d = np.full(len(c_ent), np.nan, dtype=np.float64)
            if not np.isnan(q_lat):
                ok = ~np.isnan(locs).any(axis=1)
                if ok.any():
                    d[ok] = haversine_km(q_lat, q_lon, locs[ok, 0], locs[ok, 1])
            d_eff = np.where(np.isnan(d), self.rho[r_q], d)   # missing -> rho_{r_q}

            dt = np.abs(c_ts.astype(np.float64) - float(t_q))
            psi = self.alpha[r_q] * np.power(dt, self.beta)
            if not self.ablate_s_sample:
                psi = psi + self.zeta[r_q] * np.power(d_eff, self.beta_s)

            if self.sampling in ("st_rtns", "rtns"):
                logits = -psi
                logits -= logits.max()
                w = np.exp(logits)
            elif self.sampling == "recent":
                w = np.zeros_like(psi)
                w[np.argsort(-c_ts.astype(np.float64))[:K]] = 1.0
            else:  # uniform
                w = np.ones_like(psi)

            take = min(K, len(c_ent))
            if self.deterministic:
                sel = np.argsort(-w)[:take]
            else:
                # Gumbel top-k: exact weighted sampling without replacement in
                # O(n + k log k), much faster than rng.choice(p=...) for large n
                w = np.maximum(w, 1e-30)
                g = self.rng.gumbel(size=w.shape[0])
                keys = np.log(w) + g
                if take < keys.shape[0]:
                    sel = np.argpartition(-keys, take - 1)[:take]
                else:
                    sel = np.argsort(-keys)

            ent[b, :take] = c_ent[sel]
            rel[b, :take] = c_rel[sel]
            ts[b, :take] = c_ts[sel]
            lat[b, :take] = locs[sel, 0]
            lon[b, :take] = locs[sel, 1]
            dist[b, :take] = d[sel]
            fidx[b, :take] = c_idx[sel]

        return SampledNeighborhood(ent=ent, rel=rel, ts=ts, lat=lat, lon=lon,
                                   dist=dist, fidx=fidx,
                                   N_q_all=all_idx, N_q=fidx)