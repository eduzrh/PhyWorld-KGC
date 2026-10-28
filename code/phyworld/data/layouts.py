# -*- coding: utf-8 -*-
"""Tolerant readers for the two on-disk layouts shipped with the benchmark.

Layout A (``ids/<dataset>``): ``train.txt`` holds ``s r o t`` integer ids and
locations live in ``locations/<split>.txt`` (one entry per line).

Layout B (``<Dataset>``): ``train.txt`` holds tab-separated *names* and a date,
optionally with an inline location column::

    South Korea \t Criticize or denounce \t North Korea \t 2014-05-13 \t (37.5,126.9)
"""

from __future__ import annotations

import datetime as _dt
import os
from typing import Dict, List, Optional, Tuple

import numpy as np

from ..utils import Geocoder, parse_location

LAYOUT_CHOICES = ("auto", "native", "ids")


def normalise_layout(layout: str) -> str:
    """Lower-case and strip ``layout`` so CLI input matches :data:`LAYOUT_CHOICES`."""
    return (layout or "auto").strip().lower()


def _is_int(token: str) -> bool:
    try:
        int(token)
        return True
    except ValueError:
        return False


def _read_vocab(path: str) -> Dict[str, int]:
    vocab: Dict[str, int] = {}
    if os.path.exists(path):
        with open(path, "r", encoding="utf-8") as f:
            for i, line in enumerate(f):
                name = line.strip()
                if name and name not in vocab:
                    vocab[name] = i
    return vocab


def _parse_time(token: str, cache: Dict[str, int]) -> int:
    if _is_int(token):
        return int(token)
    if token not in cache:
        try:
            cache[token] = _dt.date.fromisoformat(token[:10]).toordinal()
        except ValueError:
            cache[token] = len(cache)
    return cache[token]


def read_split_named(fact_path: str, loc_path: Optional[str],
                     geocoder: Geocoder) -> Tuple[np.ndarray, np.ndarray, List[Optional[str]]]:
    """Parse layout B into the same ``(facts, latlon, texts)`` triple."""
    root = os.path.dirname(fact_path)
    ent_vocab = _read_vocab(os.path.join(root, "entities.txt"))
    rel_vocab = _read_vocab(os.path.join(root, "relations.txt"))
    tcache: Dict[str, int] = {}

    rows: List[Tuple[int, int, int, int]] = []
    inline_loc: List[Optional[str]] = []
    with open(fact_path, "r", encoding="utf-8") as f:
        for line in f:
            parts = line.rstrip("\n").split("\t")
            if len(parts) < 4:
                parts = line.split()
            if len(parts) < 4:
                continue
            s_name, r_name, o_name, t_raw = parts[0], parts[1], parts[2], parts[3]
            s = ent_vocab.setdefault(s_name, len(ent_vocab))
            o = ent_vocab.setdefault(o_name, len(ent_vocab))
            r = rel_vocab.setdefault(r_name, len(rel_vocab))
            rows.append((s, r, o, _parse_time(t_raw, tcache)))
            inline_loc.append(parts[4] if len(parts) > 4 else None)

    facts = np.asarray(rows, dtype=np.int64) if rows else np.zeros((0, 4), dtype=np.int64)
    raw_loc: List[Optional[str]] = inline_loc
    if loc_path is not None and os.path.exists(loc_path):
        with open(loc_path, "r", encoding="utf-8") as f:
            file_loc = [ln.rstrip("\n") for ln in f]
        if len(file_loc) >= len(raw_loc):
            raw_loc = file_loc

    latlon = np.full((len(rows), 2), np.nan, dtype=np.float64)
    texts: List[Optional[str]] = [None] * len(rows)
    for i, token in enumerate(raw_loc):
        if i >= len(rows) or not token:
            continue
        lat, lon, txt = parse_location(token)
        if lat is None and txt is not None:
            ll = geocoder.resolve(txt)
            if ll is not None:
                lat, lon = ll
            else:
                texts[i] = txt
                continue
        if lat is not None:
            latlon[i, 0] = lat
            latlon[i, 1] = lon
    return facts, latlon, texts