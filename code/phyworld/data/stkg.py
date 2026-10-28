# -*- coding: utf-8 -*-
"""STKG data structures: 5-tuples (s, r, o, t, l), adjacency, filters."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from ..utils import Geocoder, haversine_km, load_json, parse_location
from .layouts import normalise_layout

Fact = Tuple[int, int, int, int]          # (s, r, o, t)
Fact5 = Tuple[int, int, int, int, int]   # (s, r, o, t, location_id)


@dataclass
class Split:
    """One split of an STKG.

    ``facts``      : (N, 4) int64 array [s, r, o, t] (base relations only)
    ``loc_latlon`` : (N, 2) float64, NaN when the location is missing
    ``loc_text``   : list[str] or None, the raw administrative text (if any)
    """

    facts: np.ndarray
    loc_latlon: np.ndarray
    loc_text: List[Optional[str]] = field(default_factory=list)

    def __len__(self) -> int:
        return int(self.facts.shape[0])


@dataclass
class STKGDataset:
    name: str
    num_entities: int
    num_relations_base: int
    num_timestamps: int
    train: Split
    valid: Split
    test: Split
    setting: str = "interpolation"
    #: (num_rel_base,) median subject-object distance per relation (rho_r)
    relation_rho: Optional[np.ndarray] = None
    #: (num_rel_base,) median time gap per relation (mu_r)
    relation_mu: Optional[np.ndarray] = None

    # ------------------------------------------------------------------
    @property
    def num_relations(self) -> int:
        """Relations after inverse augmentation."""
        return 2 * self.num_relations_base

    def inv(self, r: int) -> int:
        return r + self.num_relations_base if r < self.num_relations_base else r - self.num_relations_base

    def alpha_prior(self) -> np.ndarray:
        """alpha_r = 1 / mu_r for base and inverse relations."""
        mu = np.where(self.relation_mu <= 0, 1.0, self.relation_mu).astype(np.float32)
        a = 1.0 / np.maximum(mu, 1e-6)
        a = np.clip(a, 1e-4, 100.0)
        return np.concatenate([a, a], axis=0)


# ----------------------------------------------------------------------
# loading
# ----------------------------------------------------------------------

def _split_paths(root: str, split: str) -> Tuple[str, Optional[str]]:
    """Locate ``<split>.txt`` and the aligned ``locations/<split>.txt``."""
    cands = [
        os.path.join(root, f"{split}.txt"),
        os.path.join(root, f"{os.path.basename(root)}_{split}.txt"),
    ]
    fact_path = next((p for p in cands if os.path.exists(p)), None)
    if fact_path is None:
        raise FileNotFoundError(f"No {split}.txt under {root}")
    loc_path = os.path.join(root, "locations", f"{split}.txt")
    if not os.path.exists(loc_path):
        loc_path = None
    return fact_path, loc_path


def _read_split(root: str, split: str, geocoder: Geocoder, time_mode: str = "raw") -> Split:
    fact_path, loc_path = _split_paths(root, split)
    with open(fact_path, "r", encoding="utf-8") as f:
        first = f.readline().split()
    if first and not (first[0].lstrip("-").isdigit() and first[1].lstrip("-").isdigit()):
        # layout B: names + date (+ optional inline location)
        from .layouts import normalise_layout, read_split_named
        facts, latlon, texts = read_split_named(fact_path, loc_path, geocoder)
        return Split(facts=facts, loc_latlon=latlon, loc_text=texts)

    rows: List[Tuple[int, int, int, int]] = []
    with open(fact_path, "r", encoding="utf-8") as f:
        for line in f:
            parts = line.split()
            if len(parts) < 4:
                continue
            rows.append((int(parts[0]), int(parts[1]), int(parts[2]), int(parts[3])))
    facts = np.asarray(rows, dtype=np.int64) if rows else np.zeros((0, 4), dtype=np.int64)

    latlon = np.full((len(rows), 2), np.nan, dtype=np.float64)
    texts: List[Optional[str]] = [None] * len(rows)
    if loc_path is not None:
        with open(loc_path, "r", encoding="utf-8") as f:
            loc_lines = [ln.rstrip("\n") for ln in f]
        n = min(len(loc_lines), len(rows))
        for i in range(n):
            lat, lon, txt = parse_location(loc_lines[i])
            if lat is None and txt is not None:
                # offline geocoding of administrative text (Section 3.2)
                ll = geocoder.resolve(txt)
                if ll is not None:
                    lat, lon = ll
                else:
                    texts[i] = txt      # unresolved -> treated as missing
                    continue
            if lat is not None:
                latlon[i, 0] = lat
                latlon[i, 1] = lon
    return Split(facts=facts, loc_latlon=latlon, loc_text=texts)


def load_stkg(data_root: str, dataset: str, geocoder: Optional[Geocoder] = None,
              setting: str = "interpolation", layout: str = "auto") -> STKGDataset:
    """Load one benchmark.

    ``data_root`` must contain ``<dataset>/train.txt`` (optionally
    ``locations/train.txt``), ``stat.txt`` and ``entities.txt`` / ``relations.txt``.
    """
    layout = normalise_layout(layout)
    root = os.path.join(data_root, dataset)
    # ``ids/<dataset_lower>`` shares the integer id space of the exported
    # pretrained tables (``entity_init`` / ``relation_init``).  Using the native
    # name layout would silently re-index entities and destroy that alignment, so
    # ``layout="auto"`` prefers the exported layout whenever it exists.
    ids_root = os.path.join(data_root, "ids", dataset.lower())
    if layout in ("auto", "ids") and os.path.isdir(ids_root):
        root = ids_root
    elif not os.path.isdir(root):
        alt = os.path.join(data_root, dataset.lower())
        if os.path.isdir(alt):
            root = alt
        else:
            raise FileNotFoundError(root)
    geocoder = geocoder or Geocoder()

    stat_path = os.path.join(root, "stat.txt")
    n_ent = n_rel = n_ts = 0
    if os.path.exists(stat_path):
        vals = open(stat_path).read().split()
        n_ent, n_rel = int(vals[0]), int(vals[1])
        n_ts = int(vals[2]) if len(vals) > 2 else 0

    train = _read_split(root, "train", geocoder)
    valid = _read_split(root, "valid", geocoder)
    test = _read_split(root, "test", geocoder)

    all_f = np.concatenate([train.facts, valid.facts, test.facts], axis=0)
    if n_ent <= 0:
        n_ent = int(all_f[:, [0, 2]].max()) + 1
    # ``entities.txt`` is the entity space of the benchmark, and Table 2 counts
    # it: some datasets keep genuine source entities in the vocabulary that no
    # shipped fact mentions any more (see datasets/README.md).
    ent_path = os.path.join(root, "entities.txt")
    if os.path.exists(ent_path):
        with open(ent_path, "r", encoding="utf-8") as fh:
            n_vocab = sum(1 for line in fh if line.strip())
        n_ent = max(n_ent, n_vocab)
    if n_rel <= 0:
        n_rel = int(all_f[:, 1].max()) + 1

    ds = STKGDataset(
        name=dataset, num_entities=n_ent, num_relations_base=n_rel,
        num_timestamps=n_ts, train=train, valid=valid, test=test, setting=setting,
    )
    ds.relation_rho, ds.relation_mu = compute_relation_scales(ds)
    return ds


def compute_relation_scales(ds: STKGDataset) -> Tuple[np.ndarray, np.ndarray]:
    """Per-relation spatial scale rho_r and temporal scale mu_r (Section 3.2).

    rho_r : median great-circle distance between subject and object locations
            over training facts of r with both locations available.
    mu_r  : median time interval between consecutive occurrences of r.
    """
    n_rel = ds.num_relations_base
    rho = np.full(n_rel, np.nan, dtype=np.float64)
    mu = np.zeros(n_rel, dtype=np.float64)
    f = ds.train.facts
    ll = ds.train.loc_latlon
    for r in range(n_rel):
        mask = f[:, 1] == r
        if not mask.any():
            continue
        times = np.sort(f[mask, 3])
        if times.size >= 2:
            mu[r] = float(np.median(np.diff(times)))
        sub = ll[mask]
        ok = ~np.isnan(sub).any(axis=1)
        # subject/object locations both needed: fall back to the fact location
        if ok.sum() >= 2:
            sel = sub[ok]
            d = haversine_km(sel[:-1, 0], sel[:-1, 1], sel[1:, 0], sel[1:, 1])
            d = d[np.isfinite(d)]
            if d.size:
                rho[r] = float(np.median(d))
    # global fallbacks
    fin = rho[np.isfinite(rho)]
    global_rho = float(np.median(fin)) if fin.size else 1000.0
    rho = np.where(np.isfinite(rho), rho, global_rho)
    rho = np.clip(rho, 1.0, 20000.0)
    mu = np.where(mu > 0, mu, np.median(mu[mu > 0]) if (mu > 0).any() else 1.0)
    return rho.astype(np.float32), mu.astype(np.float32)


# ----------------------------------------------------------------------
# graph indices
# ----------------------------------------------------------------------

class STKGIndex:
    """Adjacency structures used by sampling, tensorisation and rule grounding.

    ``adj_out[e]``  : list of (object, relation, time, fact_index)
    ``adj_in[e]``   : list of (subject, relation, time, fact_index)
    Both are sorted by time (ascending).
    """

    def __init__(self, ds: STKGDataset, augment_inverse: bool = True):
        self.ds = ds
        self.augment_inverse = augment_inverse
        n_ent = ds.num_entities
        n_base = ds.num_relations_base

        f = ds.train.facts
        latlon = ds.train.loc_latlon
        out: List[List[Tuple[int, int, int, int]]] = [[] for _ in range(n_ent)]
        inc: List[List[Tuple[int, int, int, int]]] = [[] for _ in range(n_ent)]
        for i, (s, r, o, t) in enumerate(f):
            out[s].append((o, r, t, i))
            inc[o].append((s, r, t, i))
            if augment_inverse:
                out[o].append((s, r + n_base, t, i))
                inc[s].append((o, r + n_base, t, i))
        for e in range(n_ent):
            if len(out[e]) > 1:
                out[e].sort(key=lambda x: x[2])
            if len(inc[e]) > 1:
                inc[e].sort(key=lambda x: x[2])
        self.adj_out = out
        self.adj_in = inc

        # filter dicts for time-aware filtered ranking
        # ---- text-location vocabulary (frozen encoder input) ----
        self.text_vocab: List[str] = []
        self.fact_text_id = np.full(len(f), -1, dtype=np.int64)
        texts = ds.train.loc_text or []
        if texts:
            vocab: Dict[str, int] = {}
            for i, tx in enumerate(texts):
                if not tx:
                    continue
                key = str(tx).strip().lower()
                if not key:
                    continue
                j = vocab.get(key)
                if j is None:
                    j = len(self.text_vocab)
                    vocab[key] = j
                    self.text_vocab.append(key)
                self.fact_text_id[i] = j
        self.ent_text_id = np.full(n_ent, -1, dtype=np.int64)
        for i, (s0, r0, o0, t0) in enumerate(f):
            if self.fact_text_id[i] >= 0 and self.ent_text_id[s0] < 0:
                self.ent_text_id[s0] = self.fact_text_id[i]

        # filter dicts for time-aware filtered ranking
        self.sr2o: Dict[Tuple[int, int], set] = {}
        self.srt2o: Dict[Tuple[int, int, int], set] = {}
        for split in (ds.train, ds.valid, ds.test):
            ff = split.facts
            for s, r, o, t in ff:
                self.sr2o.setdefault((s, r), set()).add(o)
                self.srt2o.setdefault((s, r, t), set()).add(o)
                if augment_inverse:
                    self.sr2o.setdefault((o, r + n_base), set()).add(s)
                    self.srt2o.setdefault((o, r + n_base, t), set()).add(s)

    # ------------------------------------------------------------------
    def query_location(self, ent: int, t: float) -> Tuple[float, float]:
        """Location of ``ent`` at time ``t`` (last fact at or before t, else first)."""
        hist = self.adj_out[ent]
        best = None
        first = None
        for (o, r, tt, idx) in hist:
            la, lo = self.fact_location(idx)
            if np.isnan(la):
                continue
            if first is None:
                first = (la, lo)
            if tt <= t:
                best = (la, lo)
        if best is not None:
            return best
        if first is not None:
            return first
        return float("nan"), float("nan")

    def entity_history(self):
        """Subject-side history used by the velocity check of Eq. (18)."""
        from ..models.reasoner import EntityHistory
        f = self.ds.train.facts
        ll = self.ds.train.loc_latlon
        subj = np.concatenate([f[:, 0], f[:, 2]])
        ts = np.concatenate([f[:, 3], f[:, 3]]).astype(np.float64)
        lat = np.concatenate([ll[:, 0], ll[:, 0]])
        lon = np.concatenate([ll[:, 1], ll[:, 1]])
        return EntityHistory.from_arrays(self.ds.num_entities, subj, ts, lat, lon)

    def fact_location(self, idx: int) -> Tuple[float, float]:
        ll = self.ds.train.loc_latlon[idx]
        return float(ll[0]), float(ll[1])

    def gather_between(self, entities: Sequence[int], t_center: int, window: int,
                       cap: int = 2000) -> np.ndarray:
        """Facts among ``entities`` with ``|t - t_center| <= window`` (induced subgraph)."""
        ents = np.asarray(sorted(set(int(e) for e in entities)), dtype=np.int64)
        if ents.size == 0:
            return np.zeros((0, 4), dtype=np.int64)
        collected: List[Tuple[int, int, int, int]] = []
        for e in ents.tolist():
            for (o, r, t, idx) in self.adj_out[e]:
                if abs(t - t_center) <= window:
                    collected.append((e, r, o, t))
        if not collected:
            return np.zeros((0, 4), dtype=np.int64)
        arr = np.asarray(collected, dtype=np.int64)
        if arr.shape[0] > cap:
            # keep the temporally closest facts
            order = np.argsort(np.abs(arr[:, 3] - t_center))[:cap]
            arr = arr[np.sort(order)]
        return arr