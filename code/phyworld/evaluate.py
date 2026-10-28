# -*- coding: utf-8 -*-
"""Time-aware filtered evaluation: MRR, Hits@1/3/10 (Section 4.1)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch

from .data.batch import QueryBatcher
from .models.astsc import GateTable
from .models.reasoner import EntityHistory


@dataclass
class EvalResult:
    mrr: float
    h1: float
    h3: float
    h10: float
    num_queries: int
    direction: str

    def as_dict(self) -> Dict[str, float]:
        return {"MRR": self.mrr, "H@1": self.h1, "H@3": self.h3,
                "H@10": self.h10, "n": self.num_queries}


def _rank_of(scores: torch.Tensor, gold: int, filtered: Sequence[int]) -> int:
    s = scores.clone()
    if filtered:
        s[torch.as_tensor(list(filtered), dtype=torch.long, device=s.device)] = -1e30
    return int((s > s[gold]).sum().item()) + 1


@torch.no_grad()
def evaluate(model, ds, index, batcher: QueryBatcher, split: str,
             batch_size: int = 32, device: str = "cpu",
             gamma_table: Optional[GateTable] = None,
             history: Optional[EntityHistory] = None,
             ablations=None, use_reasoner: bool = True,
             max_queries: Optional[int] = None,
             direction: str = "tail") -> EvalResult:
    """Evaluate tail prediction (``direction='tail'``) or head prediction.

    Head queries are answered through the inverse relation ``r + |R|/2`` so that
    a single scoring head is used for both directions (Section 4.1).
    """
    model.eval()
    data = getattr(ds, split)
    facts = data.facts
    n_base = ds.num_relations_base
    rows = facts if direction == "tail" else np.stack(
        [facts[:, 2], facts[:, 1] + n_base, facts[:, 0], facts[:, 3]], axis=1)
    if max_queries is not None and len(rows) > max_queries:
        rng = np.random.RandomState(42)
        rows = rows[rng.choice(len(rows), size=max_queries, replace=False)]

    ranks: List[int] = []
    for start in range(0, len(rows), batch_size):
        chunk = rows[start:start + batch_size]
        queries = [(int(s), int(r), int(t)) for s, r, o, t in chunk]
        gold = [int(o) for s, r, o, t in chunk]
        batch = batcher.build(queries, gold=gold, setting=split).to(device)

        # rule grounding scores for Stage II rule term (Eq. 16)
        rule_scores = None
        if getattr(model.cfg, "use_rule_term", True) and getattr(batcher, "grounder", None):
            rule_scores = batcher.grounder.ground_batch(queries)

        out = model(batch, gamma_table=gamma_table, rule_scores=rule_scores,
                    history=history, ablations=ablations, use_reasoner=use_reasoner)
        scores = out.scores.detach().cpu()
        for i, (s, r, o, t) in enumerate(chunk):
            filt = index.srt2o.get((int(s), int(r), int(t)), set()) - {int(o)}
            ranks.append(_rank_of(scores[i], int(o), filt))

    ranks_arr = np.asarray(ranks, dtype=np.float64)
    return EvalResult(
        mrr=float(np.mean(1.0 / ranks_arr)),
        h1=float(np.mean(ranks_arr <= 1)),
        h3=float(np.mean(ranks_arr <= 3)),
        h10=float(np.mean(ranks_arr <= 10)),
        num_queries=int(ranks_arr.size),
        direction=direction,
    )


def evaluate_both(model, ds, index, batcher, split: str, **kwargs) -> Dict[str, EvalResult]:
    return {
        "tail": evaluate(model, ds, index, batcher, split, direction="tail", **kwargs),
        "head": evaluate(model, ds, index, batcher, split, direction="head", **kwargs),
    }