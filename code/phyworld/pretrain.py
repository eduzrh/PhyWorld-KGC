# -*- coding: utf-8 -*-
"""Static KG embedding pre-training (the ``E_fix`` initialisation of Section 3.2).

The paper initialises the entity and relation embeddings of Stage I "from static
KG embeddings pre-trained on the training graph".  This module implements that
step: a neighbour-aware DistMult is fitted on the *training* split only (no
timestamps, no locations, no validation leakage) and the resulting tables become
``entity_init`` / ``relation_init``:

    h_e = LN( E[e] + mean_{(e,r,n) in N(e)} ( R[r] * E[n] ) ),      (init)
    score(s, r, o) = < h_s * R[r], h_o > .

The exported relation table covers the augmented relation space of the model
(``r`` and its inverse ``r + |R|``), so a checkpoint produced here can be passed
straight to ``phyworld.train --init-emb`` and to
``PhyWorldKGC.load_pretrained``.

Neighbour samples are drawn once per epoch into a fixed ``(E, k)`` table, which
keeps the aggregation a handful of tensor ops per batch even on PhyGDELT
(2.5M training facts).
"""

from __future__ import annotations

from typing import Dict, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


# ----------------------------------------------------------------------
def build_neighbour_table(subjects: np.ndarray, objects: np.ndarray,
                          relations: np.ndarray, num_entities: int, k: int,
                          rng: np.random.RandomState):
    """Sample up to ``k`` outgoing neighbours per entity.

    Returns ``(neigh, rel, has)`` where ``neigh`` / ``rel`` are ``(E, k)`` int64
    arrays padded with ``-1`` and ``has`` is ``(E,)``.  ``subjects`` must be
    sorted so that the rows of one subject are contiguous.
    """
    k = max(1, int(k))
    neigh = np.full((num_entities, k), -1, dtype=np.int64)
    rel = np.full((num_entities, k), -1, dtype=np.int64)
    indptr = np.searchsorted(subjects, np.arange(num_entities + 1), side="left")
    for e in range(num_entities):
        lo, hi = int(indptr[e]), int(indptr[e + 1])
        if hi <= lo:
            continue
        if hi - lo <= k:
            sel = np.arange(hi - lo)
        else:
            sel = rng.choice(hi - lo, size=k, replace=False)
        m = sel.size
        neigh[e, :m] = objects[lo:hi][sel]
        rel[e, :m] = relations[lo:hi][sel]
    return neigh, rel


def augmented_triples(facts: np.ndarray, num_relations_base: int):
    """``(s, r, o)`` plus the inverse copy ``(o, r + |R|, s)`` used by the model."""
    s = facts[:, 0].astype(np.int64)
    r = facts[:, 1].astype(np.int64)
    o = facts[:, 2].astype(np.int64)
    src = np.concatenate([s, o])
    rel = np.concatenate([r, r + int(num_relations_base)])
    dst = np.concatenate([o, s])
    order = np.argsort(src, kind="stable")
    return src[order], rel[order], dst[order]


# ----------------------------------------------------------------------
class NeighbourAwareDistMult(nn.Module):
    """The static scorer of the initialisation step."""

    def __init__(self, num_entities: int, num_relations: int, embed_dim: int,
                 neigh: np.ndarray, rel: np.ndarray, dropout: float = 0.1):
        super().__init__()
        self.num_entities = int(num_entities)
        self.num_relations = int(num_relations)
        self.embed_dim = int(embed_dim)
        self.E = nn.Embedding(self.num_entities, self.embed_dim)
        self.R = nn.Embedding(self.num_relations, self.embed_dim)
        nn.init.xavier_uniform_(self.E.weight)
        nn.init.xavier_uniform_(self.R.weight)
        self.ln = nn.LayerNorm(self.embed_dim)
        self.drop = nn.Dropout(dropout)
        self.register_buffer("neigh", torch.as_tensor(neigh, dtype=torch.long))
        self.register_buffer("rel", torch.as_tensor(rel, dtype=torch.long))

    def set_neighbours(self, neigh: np.ndarray, rel: np.ndarray) -> None:
        self.neigh = torch.as_tensor(neigh, dtype=torch.long, device=self.neigh.device)
        self.rel = torch.as_tensor(rel, dtype=torch.long, device=self.rel.device)

    def aggregate(self, ent_ids: torch.Tensor) -> torch.Tensor:
        base = self.E(ent_ids)
        nb = self.neigh[ent_ids]
        nr = self.rel[ent_ids]
        has = nb >= 0
        msg = self.E(nb.clamp_min(0)) * self.R(nr.clamp_min(0))
        msg = msg * has.unsqueeze(-1).to(msg.dtype)
        cnt = has.sum(dim=1, keepdim=True).clamp_min(1).to(msg.dtype)
        return self.ln(base + self.drop(msg.sum(dim=1) / cnt))

    def score(self, s: torch.Tensor, r: torch.Tensor, o: torch.Tensor) -> torch.Tensor:
        return (self.aggregate(s) * self.R(r) * self.aggregate(o)).sum(dim=-1)


# ----------------------------------------------------------------------
def pretrain_static_embeddings(facts: np.ndarray, num_entities: int,
                               num_relations_base: int, embed_dim: int = 200,
                               epochs: int = 20, batch_size: int = 2048,
                               lr: float = 1e-3, weight_decay: float = 1e-6,
                               num_neighbours: int = 20,
                               num_neighbours_export: int = 50,
                               neg_ratio: int = 10, seed: int = 42,
                               device: str = "cpu", resample_every: int = 1,
                               logger: Optional[object] = None) -> Dict[str, object]:
    """Fit the static scorer and return ``entity_init`` / ``relation_init``."""
    torch.manual_seed(seed)
    rng = np.random.RandomState(seed)
    facts = np.asarray(facts, dtype=np.int64)
    src, rel, dst = augmented_triples(facts, num_relations_base)
    num_relations = 2 * int(num_relations_base)

    neigh, nrel = build_neighbour_table(src, dst, rel, num_entities, num_neighbours, rng)
    model = NeighbourAwareDistMult(num_entities, num_relations, embed_dim,
                                   neigh, nrel).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    bce = nn.BCEWithLogitsLoss()

    s_all = torch.as_tensor(src, dtype=torch.long, device=device)
    r_all = torch.as_tensor(rel, dtype=torch.long, device=device)
    o_all = torch.as_tensor(dst, dtype=torch.long, device=device)
    n = int(s_all.numel())
    history = []

    for epoch in range(1, int(epochs) + 1):
        if epoch > 1 and resample_every and (epoch - 1) % resample_every == 0:
            neigh, nrel = build_neighbour_table(src, dst, rel, num_entities,
                                                num_neighbours, rng)
            model.set_neighbours(neigh, nrel)
        model.train()
        perm = torch.as_tensor(rng.permutation(n), dtype=torch.long, device=device)
        total, steps = 0.0, 0
        for st in range(0, n, int(batch_size)):
            idx = perm[st:st + int(batch_size)]
            s, r, o = s_all[idx], r_all[idx], o_all[idx]
            pos = model.score(s, r, o)
            b = int(s.numel())
            neg_o = torch.randint(0, num_entities, (b * int(neg_ratio),), device=device)
            neg = model.score(s.repeat_interleave(neg_ratio),
                              r.repeat_interleave(neg_ratio), neg_o)
            loss = bce(pos, torch.ones_like(pos)) + bce(neg, torch.zeros_like(neg))
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            opt.step()
            total += float(loss.detach())
            steps += 1
        epoch_loss = total / max(1, steps)
        history.append(epoch_loss)
        if logger is not None:
            logger.info("[pretrain] epoch=%d loss=%.4f", epoch, epoch_loss)
        else:
            print(f"[pretrain] epoch={epoch} loss={epoch_loss:.4f}", flush=True)

    # Export: neighbour-aware features for every vocabulary entry, R for every
    # augmented relation.  Entities without a training mention keep LN(E[e]).
    model.eval()
    model.set_neighbours(*build_neighbour_table(src, dst, rel, num_entities,
                                                num_neighbours_export, rng))
    with torch.no_grad():
        entity_init = torch.zeros((num_entities, embed_dim), dtype=torch.float32)
        for st in range(0, num_entities, 512):
            ed = min(num_entities, st + 512)
            ids = torch.arange(st, ed, dtype=torch.long, device=device)
            entity_init[st:ed] = model.aggregate(ids).detach().cpu().float()
        relation_init = model.R.weight.detach().cpu().float()

    return {
        "entity_init": entity_init,
        "relation_init": relation_init,
        "meta": {
            "num_entities": int(num_entities),
            "num_relations": int(num_relations),
            "num_relations_base": int(num_relations_base),
            "embed_dim": int(embed_dim),
            "score_func": "distmult",
            "neighbour_aware": True,
            "epochs": int(epochs),
            "num_neighbours": int(num_neighbours),
            "seed": int(seed),
            "history": history,
        },
    }