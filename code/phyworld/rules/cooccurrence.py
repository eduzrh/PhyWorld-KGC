# -*- coding: utf-8 -*-
"""Co-occurrence-type relation annotation R_co (Section 3.3, last paragraph)."""

from __future__ import annotations

from typing import Dict, List, Tuple

import numpy as np

from ..data.stkg import STKGDataset
from ..utils import haversine_km


def annotate_cooccurrence(ds: STKGDataset, d_co: float = 50.0,
                          pi_co: float = 0.8) -> Tuple[List[int], Dict[int, Dict[str, float]]]:
    """Relations whose subject and object share the same place when the fact occurs.

    ``d_co`` is the distance threshold (km) and ``pi_co`` the required fraction of
    facts below it.  The entity location is the median of the locations of its
    training facts (the dataset records one location per fact).
    """
    f = ds.train.facts
    ll = ds.train.loc_latlon
    n_ent = ds.num_entities

    # entity location = median of the fact locations it participates in
    ent_lat = np.full(n_ent, np.nan)
    ent_lon = np.full(n_ent, np.nan)
    ok = ~np.isnan(ll).any(axis=1)
    # median location per entity, computed over both subject and object roles
    subj_e, obj_e = f[ok, 0], f[ok, 2]
    lat, lon = ll[ok, 0], ll[ok, 1]
    for role, ents in (("s", subj_e), ("o", obj_e)):
        order = np.argsort(ents, kind="stable")
        ents_s, lat_s, lon_s = ents[order], lat[order], lon[order]
        uniq, starts = np.unique(ents_s, return_index=True)
        ends = np.append(starts[1:], len(ents_s))
        for u, a, b in zip(uniq, starts, ends):
            if np.isnan(ent_lat[u]):
                ent_lat[u] = float(np.median(lat_s[a:b]))
                ent_lon[u] = float(np.median(lon_s[a:b]))

    n_rel = ds.num_relations_base
    result: List[int] = []
    stats: Dict[int, Dict[str, float]] = {}
    for r in range(n_rel):
        mask = (f[:, 1] == r) & ok
        if mask.sum() < 5:
            continue
        idx = np.nonzero(mask)[0]
        s_lat = ent_lat[f[idx, 0]]
        dist = haversine_km(s_lat, ent_lon[f[idx, 0]], ll[idx, 0], ll[idx, 1])
        dist = dist[np.isfinite(dist)]
        if dist.size < 5:
            continue
        med = float(np.median(dist))
        frac = float(np.mean(dist <= d_co))
        stats[r] = {"median_km": med, "fraction": frac, "n": int(dist.size)}
        if med < d_co and frac >= pi_co:
            result.append(r)
    return result, stats


def ablate_cooccurrence(ds: STKGDataset, r_co: List[int]) -> List[int]:
    """Variant ``w/o Co-occ.``: velocity checks apply to every relation."""
    return list(range(ds.num_relations_base))