# -*- coding: utf-8 -*-
"""Candidate rule generation by spatio-temporal random walks (Section 3.3).

Starting from each training fact ``(s, r_h, o, t, l)`` we walk backwards in the
temporally non-increasing direction for at most ``n_max`` steps and stop when the
walk returns to ``s``.  The resulting cyclic path defines a chain rule; empirical
90% quantiles of the time differences / spatial distances become ``omega_i`` /
``rho_i``.  Branch rules append one extra walk step from ``X``.

Dense graphs
------------
On benchmarks such as PhyGDELT (485 entities, 2.5M training facts) the in-degree
of a single entity reaches tens of thousands, so materialising the admissible
edge list as Python tuples dominates the run time.  The walk therefore filters a
per-entity ``int64`` array with a boolean mask, which keeps the original edge
order (and hence the RNG stream) while making each step a single numpy call.
"""

from __future__ import annotations

from collections import defaultdict
from typing import Dict, List, Optional, Sequence, Tuple, Union

import numpy as np

from ..data.stkg import STKGDataset, STKGIndex
from ..utils import haversine_km
from .base import Atom, Rule, RuleBase

#: A walk candidate list is either the legacy list of tuples or an ``(n, 4)``
#: array of ``(entity, relation, timestamp, fact_index)`` rows.
Candidates = Union[Sequence[Tuple[int, int, int, int]], np.ndarray]


class _SignatureStats:
    """Accumulates the empirical distributions of omega_i / rho_i per signature."""

    def __init__(self, head_rel: int, rels: Tuple[int, ...], branch_rel: Optional[int],
                 max_samples: int = 4000):
        self.head_rel = head_rel
        self.rels = rels
        self.branch_rel = branch_rel
        self.windows: List[List[float]] = [[] for _ in rels]
        self.radii: List[List[float]] = [[] for _ in rels]
        self.branch_windows: List[float] = []
        self.branch_radii: List[float] = []
        self.count = 0
        self.max_samples = max_samples

    def add(self, windows: Sequence[float], radii: Sequence[Optional[float]],
            b_window: Optional[float] = None, b_radius: Optional[float] = None) -> None:
        self.count += 1
        if self.count > self.max_samples:
            return
        for i, w in enumerate(windows):
            self.windows[i].append(float(w))
            if radii[i] is not None:
                self.radii[i].append(float(radii[i]))
        if b_window is not None:
            self.branch_windows.append(float(b_window))
        if b_radius is not None:
            self.branch_radii.append(float(b_radius))

    def build(self, q: float, default_radius: float) -> Rule:
        atoms = []
        for i, rel in enumerate(self.rels):
            w = float(np.quantile(self.windows[i], q)) if self.windows[i] else 1.0
            r = float(np.quantile(self.radii[i], q)) if self.radii[i] else default_radius
            atoms.append(Atom(rel=int(rel), window=max(w, 0.0), radius=max(r, 1.0)))
        branch = None
        if self.branch_rel is not None:
            bw = float(np.quantile(self.branch_windows, q)) if self.branch_windows else 1.0
            br = float(np.quantile(self.branch_radii, q)) if self.branch_radii else default_radius
            branch = Atom(rel=int(self.branch_rel), window=max(bw, 0.0), radius=max(br, 1.0))
        return Rule(head_rel=self.head_rel, atoms=atoms, branch=branch)


class SpatioTemporalRuleMiner:
    """Bottom-up candidate generation (walk part of Section 3.3)."""

    def __init__(self, ds: STKGDataset, index: STKGIndex, max_length: int = 3,
                 walks_per_fact: int = 10, quantile: float = 0.9,
                 use_branch: bool = True, seed: int = 42,
                 fact_sample: int = 20000, beta: float = 1.0, min_length: int = 2):
        self.ds = ds
        self.index = index
        self.max_length = int(max_length)
        self.min_length = int(min_length)
        self.walks = int(walks_per_fact)
        self.quantile = float(quantile)
        self.use_branch = bool(use_branch)
        self.rng = np.random.RandomState(seed)
        self.fact_sample = int(fact_sample)
        self.beta = float(beta)

        self.alpha = ds.alpha_prior()
        rho = ds.relation_rho
        self.zeta = np.concatenate([1.0 / np.maximum(rho, 1e-3)] * 2)
        self.rho = np.concatenate([rho, rho])

        # Lazily materialised adjacency rows, keyed by entity.
        self._adj_in_np: Dict[int, np.ndarray] = {}
        self._adj_out_np: Dict[int, np.ndarray] = {}

    # ------------------------------------------------------------------
    @staticmethod
    def _as_rows(edges: Sequence[Tuple[int, int, int, int]]) -> np.ndarray:
        if not edges:
            return np.zeros((0, 4), dtype=np.int64)
        return np.asarray(edges, dtype=np.int64)

    def _incoming(self, ent: int, cur_t: int) -> np.ndarray:
        """Edges of ``ent`` whose timestamp is <= ``cur_t`` (original order)."""
        rows = self._adj_in_np.get(ent)
        if rows is None:
            rows = self._as_rows(self.index.adj_in[ent])
            self._adj_in_np[ent] = rows
        if rows.shape[0] == 0:
            return rows
        return rows[rows[:, 2] <= cur_t]

    def _outgoing(self, ent: int, t: int) -> np.ndarray:
        rows = self._adj_out_np.get(ent)
        if rows is None:
            rows = self._as_rows(self.index.adj_out[ent])
            self._adj_out_np[ent] = rows
        if rows.shape[0] == 0:
            return rows
        return rows[rows[:, 2] <= t]

    # ------------------------------------------------------------------
    def _sample_edge(self, candidates: Candidates,
                     t_head: int, lat: float, lon: float) -> Tuple[int, int, int, int]:
        if isinstance(candidates, np.ndarray):
            ent = candidates[:, 0]
            rel = candidates[:, 1]
            ts = candidates[:, 2]
            idx = candidates[:, 3]
        else:
            n = len(candidates)
            ent = np.fromiter((c[0] for c in candidates), dtype=np.int64, count=n)
            rel = np.fromiter((c[1] for c in candidates), dtype=np.int64, count=n)
            ts = np.fromiter((c[2] for c in candidates), dtype=np.int64, count=n)
            idx = np.fromiter((c[3] for c in candidates), dtype=np.int64, count=n)

        d = np.full(ent.shape[0], np.nan, dtype=np.float64)
        if not np.isnan(lat):
            locs = self.ds.train.loc_latlon[idx]
            ok = ~np.isnan(locs).any(axis=1)
            if ok.any():
                d[ok] = haversine_km(lat, lon, locs[ok, 0], locs[ok, 1])
        d_eff = np.where(np.isnan(d), self.rho[rel], d)

        dt = np.abs(ts.astype(np.float64) - float(t_head))
        psi = self.alpha[rel] * np.power(dt, self.beta) + self.zeta[rel] * np.power(d_eff, self.beta)
        w = np.exp(-(psi - psi.min()))
        w = w / max(w.sum(), 1e-12)
        j = int(self.rng.choice(ent.shape[0], p=w))
        return int(ent[j]), int(rel[j]), int(ts[j]), int(idx[j])

    # ------------------------------------------------------------------
    def mine(self) -> RuleBase:
        f = self.ds.train.facts
        if f.shape[0] > self.fact_sample:
            sel = np.sort(self.rng.choice(f.shape[0], size=self.fact_sample, replace=False))
        else:
            sel = np.arange(f.shape[0])

        stats: Dict[tuple, _SignatureStats] = {}
        default_radius = float(np.median(self.rho))

        for fi in sel.tolist():
            s, rh, o, t = (int(x) for x in f[fi])
            hlat, hlon = self.index.fact_location(fi)
            has_head_loc = not (np.isnan(hlat) or np.isnan(hlon))

            for _ in range(self.walks):
                cur, cur_t = o, t
                # reversed walk: entries are (from_ent, rel, to_ent, ts, fact_idx)
                rev: List[Tuple[int, int, int, int, int]] = []
                ok = False
                for _step in range(self.max_length):
                    cands = self._incoming(cur, cur_t)
                    if cands.shape[0] == 0:
                        break
                    nxt, r, tt, idx = self._sample_edge(cands, t, hlat, hlon)
                    rev.append((nxt, r, cur, tt, idx))
                    cur, cur_t = nxt, tt
                    if cur == s:
                        ok = True
                        break
                if not ok or len(rev) < self.min_length:
                    continue

                rev.reverse()                     # now follows X -> ... -> Y
                rels = tuple(r for (_, r, _, _, _) in rev)
                windows, radii = [], []
                for (_, _, _, tt, idx) in rev:
                    if self.ds.setting == "extrapolation":
                        windows.append(float(max(0, t - tt)))
                    else:
                        windows.append(float(abs(t - tt)))
                    lat_i, lon_i = self.index.fact_location(idx)
                    if has_head_loc and not np.isnan(lat_i):
                        radii.append(float(haversine_km(hlat, hlon, lat_i, lon_i)))
                    else:
                        radii.append(None)

                key = (rh, rels, None)
                stats.setdefault(key, _SignatureStats(rh, rels, None)).add(windows, radii)

                if self.use_branch:
                    b = self._branch_step(s, t, hlat, hlon, rh)
                    if b is not None:
                        b_rel, b_w, b_r = b
                        bkey = (rh, rels, b_rel)
                        stats.setdefault(bkey, _SignatureStats(rh, rels, b_rel)).add(
                            windows, radii, b_w, b_r)

        rules = [st.build(self.quantile, default_radius) for st in stats.values() if st.count >= 2]
        return RuleBase(rules)

    # ------------------------------------------------------------------
    def _branch_step(self, s: int, t: int, hlat: float, hlon: float,
                     head_rel: int) -> Optional[Tuple[int, float, float]]:
        """One extra walk step from X landing in the spatial bucket of Y."""
        bucket = float(max(self.rho[head_rel], 1.0))
        rows = self._outgoing(s, t)
        if rows.shape[0] == 0 or np.isnan(hlat):
            return None
        idx = rows[:, 3]
        locs = self.ds.train.loc_latlon[idx]
        ok = ~np.isnan(locs).any(axis=1)
        if not ok.any():
            return None
        rows, locs = rows[ok], locs[ok]
        d = haversine_km(hlat, hlon, locs[:, 0], locs[:, 1])
        keep = d <= bucket
        if not keep.any():
            return None
        rows, d = rows[keep], d[keep]
        if self.ds.setting == "extrapolation":
            w = (t - rows[:, 2]).astype(np.float64)
        else:
            w = np.abs(t - rows[:, 2]).astype(np.float64)
        j = int(self.rng.randint(rows.shape[0]))
        return int(rows[j, 1]), float(w[j]), float(d[j])