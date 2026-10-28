# -*- coding: utf-8 -*-
"""Rule quality: support, coverage, confidence, PCA confidence and s(rule).

Implements Section 3.3 "Rule quality and filtering":

    cov  = supp / #{(X,Y) : head holds}
    conf = supp / #{(X,Y') : body holds}
    pca  = supp / #{(X,Y') : body holds and X has a head fact}
    s(rule) = (cov * conf * pca) ^ (1/3)

with ``supp`` the number of distinct (X, Y) pairs for which body *and* head hold.

Grounding is chunked over head facts (``chunk_size``) and the PCA denominator is
resolved with sorted-array lookups instead of a Python loop.  Both changes keep
the peak memory of a large benchmark (PhyICEWS18-Ext: 426k facts, 252 relations)
below a few hundred MiB -- the naive 3,000-row grounding exceeded 7 GB and was
OOM-killed on the release machine.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from ..data.stkg import STKGDataset
from .base import Rule, RuleBase
from .grounding import RuleGrounder


_KNOWN_CACHE: Dict[int, Dict[Tuple[int, int], set]] = {}


def _known_objects(ds) -> Dict[Tuple[int, int], set]:
    """``(s, r) -> {o}`` over the training graph, cached per dataset object."""
    key = id(ds)
    if key not in _KNOWN_CACHE:
        d: Dict[Tuple[int, int], set] = {}
        for s_, r_, o_, _t in ds.train.facts:
            d.setdefault((int(s_), int(r_)), set()).add(int(o_))
        _KNOWN_CACHE[key] = d
    return _KNOWN_CACHE[key]


def _compact(chunks: List[np.ndarray], limit: int = 2_000_000) -> np.ndarray:
    """Merge unique pair arrays, bounding the intermediate to ``limit`` rows."""
    if not chunks:
        return np.zeros(0, dtype=np.int64)
    merged = chunks[0] if len(chunks) == 1 else np.concatenate(chunks)
    if merged.size > limit:
        merged = np.unique(merged)
    return merged


def estimate_rule_quality(rule: Rule, ds: STKGDataset, grounder: RuleGrounder,
                          sample_size: int = 800, seed: int = 42,
                          use_confidence_only: bool = False,
                          chunk_size: int = 64) -> Rule:
    """Estimate the four standard measures on (a sample of) the training graph."""
    train = ds.train
    facts = train.facts
    mask = facts[:, 1] == rule.head_rel
    idx = np.nonzero(mask)[0]
    if idx.size == 0:
        rule.support, rule.coverage, rule.confidence, rule.pca, rule.quality = 0, 0, 0, 0, 0
        return rule

    if idx.size > sample_size:
        rng = np.random.RandomState(seed + rule.head_rel + 7 * rule.length)
        idx = np.sort(rng.choice(idx, size=sample_size, replace=False))

    head = facts[idx]                        # (N,4) s, r, o, t
    loc = train.loc_latlon[idx]
    x = head[:, 0].astype(np.int64)
    y = head[:, 2].astype(np.int64)
    t = head[:, 3].astype(np.float64)
    lat = loc[:, 0].astype(np.float64)
    lon = loc[:, 1].astype(np.float64)
    n_head = int(x.size)

    n_supp = 0
    pair_chunks: List[np.ndarray] = []
    step = max(1, int(chunk_size))
    for start in range(0, n_head, step):
        sl = slice(start, min(start + step, n_head))
        out = grounder.ground(rule, x[sl], t[sl], lat[sl], lon[sl],
                              max_paths=200_000)
        parent, gy = out["parent"], out["y"]
        if parent.size == 0:
            continue
        # support: head object appears among the grounded Y of the same head fact
        matched = np.zeros(x[sl].size, dtype=bool)
        hit = gy == y[sl][parent]
        matched[parent[hit]] = True
        n_supp += int(matched.sum())
        # ``parent`` indexes the current chunk; shift it back to the sample row.
        pairs = (start + parent).astype(np.int64) * (ds.num_entities + 1) \
            + gy.astype(np.int64)
        pair_chunks.append(np.unique(pairs))
        if sum(c.size for c in pair_chunks) > 2_000_000:
            pair_chunks = [_compact(pair_chunks)]

    xy = _compact(pair_chunks)
    if xy.size:
        xy = np.unique(xy)
    n_body = int(xy.size)

    # PCA denominator: body groundings (X, Y) whose head fact (X, r, Y) is in the KB
    if xy.size > 200_000:
        rng = np.random.RandomState(seed)
        xy = xy[rng.choice(xy.size, size=200_000, replace=False)]
    px, py = xy // (ds.num_entities + 1), xy % (ds.num_entities + 1)

    known = _known_objects(ds)
    known_vals: List[np.ndarray] = []
    # ``xy`` encodes (sample row, candidate); row ``a`` maps to entity ``x[a]``.
    for a in np.unique(px) if xy.size else np.zeros(0, dtype=np.int64):
        objs = known.get((int(x[a]), int(rule.head_rel)))
        if objs:
            known_vals.append(int(a) * (ds.num_entities + 1)
                              + np.fromiter(objs, dtype=np.int64, count=len(objs)))
    if known_vals and xy.size:
        known_arr = np.sort(np.concatenate(known_vals))
        pos = np.clip(np.searchsorted(known_arr, xy), 0, known_arr.size - 1)
        pca_den = int((known_arr[pos] == xy).sum())
    else:
        pca_den = 0

    cov = n_supp / max(1, n_head)
    conf = n_supp / max(1, n_body)
    pca = n_supp / max(1, pca_den)

    rule.support = n_supp
    rule.coverage = float(cov)
    rule.confidence = float(conf)
    rule.pca = float(pca)
    if use_confidence_only:
        rule.quality = float(conf)      # ablation "w/o conf"
    else:
        rule.quality = float((max(cov, 1e-12) * max(conf, 1e-12) * max(pca, 1e-12)) ** (1.0 / 3.0))
    return rule


def filter_and_rank(rules: Sequence[Rule], min_support: int = 5, min_pca: float = 0.1,
                    max_per_relation: int = 40) -> RuleBase:
    """Retain rules with ``pca >= c_min`` and ``supp >= n_min``; keep the best ones."""
    kept: List[Rule] = []
    for r in rules:
        if r.support >= min_support and r.pca >= min_pca and r.quality > 0:
            kept.append(r)
    kept.sort(key=lambda r: -r.quality)
    per: Dict[int, int] = {}
    out = []
    for r in kept:
        c = per.get(r.head_rel, 0)
        if c >= max_per_relation:
            continue
        per[r.head_rel] = c + 1
        out.append(r)
    return RuleBase(out)


def quality_summary(rules: RuleBase) -> Dict[str, float]:
    if len(rules) == 0:
        return {"num_rules": 0}
    sup = np.array([r.support for r in rules], dtype=np.float64)
    q = np.array([r.quality for r in rules], dtype=np.float64)
    cov = np.array([r.coverage for r in rules], dtype=np.float64)
    conf = np.array([r.confidence for r in rules], dtype=np.float64)
    pca = np.array([r.pca for r in rules], dtype=np.float64)
    return {
        "num_rules": len(rules),
        "mean_support": float(sup.mean()),
        "median_support": float(np.median(sup)),
        "mean_coverage": float(cov.mean()),
        "mean_confidence": float(conf.mean()),
        "mean_pca": float(pca.mean()),
        "mean_quality": float(q.mean()),
        "max_quality": float(q.max()),
    }