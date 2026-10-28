# -*- coding: utf-8 -*-
"""Vectorised rule grounding on the training graph (Appendix A.1, Step 7).

The grounding engine implements the adjacency-hashing scheme described in the
manuscript: facts are stored per ``(entity, relation)`` sorted by time, and a rule
body is matched atom by atom with constant-time time-window / spatial-radius
checks.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from ..utils import haversine_km
from .base import Rule


class CSRIndex:
    """Per-relation CSR adjacency sorted by (subject, time)."""

    def __init__(self, facts: np.ndarray, latlon: np.ndarray, num_entities: int,
                 num_relations_base: int, augment_inverse: bool = True):
        self.num_entities = int(num_entities)
        self.num_rel_base = int(num_relations_base)
        self.num_rel = 2 * self.num_rel_base if augment_inverse else self.num_rel_base

        n = facts.shape[0]
        subj = np.concatenate([facts[:, 0], facts[:, 2]]) if augment_inverse else facts[:, 0]
        obj = np.concatenate([facts[:, 2], facts[:, 0]]) if augment_inverse else facts[:, 2]
        rel = np.concatenate([facts[:, 1], facts[:, 1] + self.num_rel_base]) if augment_inverse else facts[:, 1]
        ts = np.concatenate([facts[:, 3], facts[:, 3]]) if augment_inverse else facts[:, 3]
        ll = np.concatenate([latlon, latlon], axis=0) if augment_inverse else latlon

        self.rel_obj: List[np.ndarray] = []
        self.rel_ts: List[np.ndarray] = []
        self.rel_lat: List[np.ndarray] = []
        self.rel_lon: List[np.ndarray] = []
        self.rel_ptr: List[np.ndarray] = []

        order = np.argsort(rel * (self.num_entities + 1) + subj, kind="stable")
        rel_s, subj_s, obj_s, ts_s = rel[order], subj[order], obj[order], ts[order]
        lat_s, lon_s = ll[order, 0], ll[order, 1]

        # sort by (rel, subj, ts)
        order2 = np.lexsort((ts_s, subj_s, rel_s))
        rel_s, subj_s, obj_s, ts_s = rel_s[order2], subj_s[order2], obj_s[order2], ts_s[order2]
        lat_s, lon_s = lat_s[order2], lon_s[order2]

        for r in range(self.num_rel):
            lo = np.searchsorted(rel_s, r, side="left")
            hi = np.searchsorted(rel_s, r, side="right")
            s_r = subj_s[lo:hi]
            # ptr over entities
            ptr = np.zeros(self.num_entities + 1, dtype=np.int64)
            np.add.at(ptr, s_r + 1, 1)
            ptr = np.cumsum(ptr)
            self.rel_ptr.append(ptr)
            self.rel_obj.append(obj_s[lo:hi].astype(np.int64))
            self.rel_ts.append(ts_s[lo:hi].astype(np.int64))
            self.rel_lat.append(lat_s[lo:hi].astype(np.float32))
            self.rel_lon.append(lon_s[lo:hi].astype(np.float32))

    # ------------------------------------------------------------------
    def expand(self, rel: int, ent: np.ndarray, t_lo: np.ndarray, t_hi: np.ndarray,
               q_lat: np.ndarray, q_lon: np.ndarray, radius: Optional[float],
               max_rows: int = 2_000_000) -> Tuple[np.ndarray, np.ndarray, np.ndarray,
                                                   np.ndarray, np.ndarray]:
        """One body-atom expansion.

        Returns ``(parent, obj, ts, lat, lon)`` where ``parent`` indexes the input
        ``ent`` array.  Facts with a missing location always satisfy the spatial
        constraint (Section 3.3).
        """
        rel = int(rel)
        ptr = self.rel_ptr[rel]
        N = ent.shape[0]
        if N == 0:
            empty = np.zeros(0, dtype=np.int64)
            return empty, empty, empty, empty.astype(np.float64), empty.astype(np.float64)

        lo = ptr[ent]
        hi = ptr[ent + 1]
        counts = (hi - lo).astype(np.int64)
        total = int(counts.sum())
        if total == 0:
            empty = np.zeros(0, dtype=np.int64)
            return empty, empty, empty, empty.astype(np.float64), empty.astype(np.float64)

        if total > max_rows:
            # Dense hubs (PhyICEWS-0515 / PhyGDELT) can expand to >10^8 rows in
            # one call, blowing past RAM before the post-hoc truncation below.
            # Sample each row's adjacency slice *before* materialising it, so the
            # working set stays at max_rows.  The sample is deterministic and
            # evenly spaced, hence reproducible.
            per = max(1, int(max_rows) // max(N, 1))
            parents, idxs = [], []
            for i in range(N):
                a, b = int(lo[i]), int(hi[i])
                if b <= a:
                    continue
                cnt = b - a
                sel = (a + np.linspace(0, cnt - 1, per).astype(np.int64)
                       if cnt > per else np.arange(a, b, dtype=np.int64))
                idxs.append(sel)
                parents.append(np.full(sel.size, i, dtype=np.int64))
            if not idxs:
                empty = np.zeros(0, dtype=np.int64)
                return empty, empty, empty, empty.astype(np.float64), empty.astype(np.float64)
            parent = np.concatenate(parents)
            idx = np.concatenate(idxs)
            total = int(idx.size)
        else:
            parent = np.repeat(np.arange(N, dtype=np.int64), counts)
            starts = np.repeat(lo, counts)
            offs = np.arange(total, dtype=np.int64) - np.repeat(np.cumsum(counts) - counts, counts)
            idx = starts + offs

        ts_all = self.rel_ts[rel][idx]
        keep = (ts_all >= t_lo[parent]) & (ts_all <= t_hi[parent])

        if radius is not None and np.isfinite(radius):
            lat_all = self.rel_lat[rel][idx]
            lon_all = self.rel_lon[rel][idx]
            qla = q_lat[parent]
            qlo = q_lon[parent]
            ok_loc = ~np.isnan(lat_all) & ~np.isnan(qla)
            d = np.zeros_like(lat_all, dtype=np.float64)
            if ok_loc.any():
                d[ok_loc] = haversine_km(qla[ok_loc], qlo[ok_loc], lat_all[ok_loc], lon_all[ok_loc])
            keep &= (~ok_loc) | (d <= radius)
        else:
            lat_all = self.rel_lat[rel][idx]
            lon_all = self.rel_lon[rel][idx]

        parent, idx = parent[keep], idx[keep]
        if parent.shape[0] > max_rows:
            parent = parent[:max_rows]
            idx = idx[:max_rows]
        obj = self.rel_obj[rel][idx]
        ts = self.rel_ts[rel][idx]
        lat = self.rel_lat[rel][idx].astype(np.float64)
        lon = self.rel_lon[rel][idx].astype(np.float64)
        return parent, obj, ts, lat, lon


class RuleGrounder:
    """Grounds rules for (a) quality estimation and (b) Eq. (16) support."""

    def __init__(self, index: CSRIndex, setting: str = "interpolation",
                 ablate_s_const: bool = False, rule_base=None, loc_fn=None):
        self.index = index
        self.setting = setting
        self.ablate_s_const = bool(ablate_s_const)
        self.rule_base = rule_base
        self.loc_fn = loc_fn

    # ------------------------------------------------------------------
    def _window_bounds(self, head_t: np.ndarray, window: float) -> Tuple[np.ndarray, np.ndarray]:
        if self.setting == "extrapolation":
            return head_t.astype(np.float64) - window, head_t.astype(np.float64)
        return head_t.astype(np.float64) - window, head_t.astype(np.float64) + window

    def ground(self, rule: Rule, x: np.ndarray, t: np.ndarray,
               lat: np.ndarray, lon: np.ndarray,
               max_paths: int = 4_000_000) -> Dict[str, np.ndarray]:
        """Ground the body of ``rule`` for a batch of (X, T, L) rows.

        Returns arrays ``(parent, y)`` describing all grounded body pairs.  When the
        branch atom is present, only rows where the branch holds are kept.
        """
        N = x.shape[0]
        parent = np.arange(N, dtype=np.int64)
        ent = x.astype(np.int64)
        t_arr = t.astype(np.float64)
        lat_arr = lat.astype(np.float64)
        lon_arr = lon.astype(np.float64)
        head_t = t.astype(np.float64)
        head_lat = lat.astype(np.float64)
        head_lon = lon.astype(np.float64)

        # ``origin`` maps the current (intermediate) row to the original input row
        origin = np.arange(N, dtype=np.int64)
        for atom in rule.atoms:
            lo, hi = self._window_bounds(head_t[origin], atom.window)
            radius = None if self.ablate_s_const else atom.radius
            pos, ent, t_arr, lat_arr, lon_arr = self.index.expand(
                atom.rel, ent, lo, hi, head_lat[origin], head_lon[origin], radius,
                max_rows=max_paths,
            )
            origin = origin[pos]
            if origin.shape[0] == 0:
                return {"parent": origin, "y": ent}

        if rule.branch is not None:
            lo, hi = self._window_bounds(head_t, rule.branch.window)
            radius = None if self.ablate_s_const else rule.branch.radius
            b_parent, _, _, _, _ = self.index.expand(
                rule.branch.rel, x.astype(np.int64), lo, hi, head_lat, head_lon, radius,
                max_rows=max_paths,
            )
            mask = np.isin(origin, np.unique(b_parent))
            origin, ent = origin[mask], ent[mask]

        return {"parent": origin, "y": ent}

    # ------------------------------------------------------------------
    def ground_batch(self, queries: Sequence[Tuple[int, int, float]]) -> List[Dict[int, float]]:
        """Per-query ``{candidate: sum s(rule)}`` for a list of ``(s, r, t)``."""
        out: List[Dict[int, float]] = []
        for (s_q, r_q, t_q) in queries:
            lat_q, lon_q = (self.loc_fn(int(s_q), float(t_q)) if self.loc_fn
                            else (float("nan"), float("nan")))
            rules = self.rule_base.for_relation(int(r_q)) if self.rule_base is not None else []
            out.append(self.ground_query(rules, int(s_q), int(t_q), lat_q, lon_q))
        return out

    # ------------------------------------------------------------------
    def ground_query(self, rules: Sequence[Rule], s_q: int, t_q: int,
                     lat_q: float, lon_q: float) -> Dict[int, float]:
        """Rule support for one query: ``{candidate: sum of s(rule)}``.

        This is the ``sum_r s(rule) * I_rule(o)`` term of Eq. (16).
        """
        scores: Dict[int, float] = {}
        if not rules:
            return scores
        x = np.asarray([s_q], dtype=np.int64)
        t = np.asarray([t_q], dtype=np.float64)
        lat = np.asarray([lat_q], dtype=np.float64)
        lon = np.asarray([lon_q], dtype=np.float64)
        for rule in rules:
            out = self.ground(rule, x, t, lat, lon, max_paths=200_000)
            ys = np.unique(out["y"])
            for y in ys.tolist():
                scores[int(y)] = scores.get(int(y), 0.0) + float(rule.quality)
        return scores