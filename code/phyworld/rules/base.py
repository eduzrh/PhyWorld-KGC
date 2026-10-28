# -*- coding: utf-8 -*-
"""Rule representation: chain rules with time windows, spatial radii and an
optional branch atom (Eq. 8 of the manuscript)."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence

import numpy as np

from ..utils import load_json, save_json


@dataclass
class Atom:
    """A body atom r_i(Z_{i-1}, Z_i, T_i, L_i) with window omega_i and radius rho_i."""

    rel: int
    window: float = 0.0          # omega_i (time units, inclusive)
    radius: float = 0.0          # rho_i (km)

    def to_dict(self) -> Dict:
        return {"rel": int(self.rel), "window": float(self.window), "radius": float(self.radius)}

    @classmethod
    def from_dict(cls, d: Dict) -> "Atom":
        return cls(rel=int(d["rel"]), window=float(d["window"]), radius=float(d["radius"]))


@dataclass
class Rule:
    """A spatio-temporal logical rule.

    ``rh(X, Y, T_{n+1}, L) <- r_1(X,Z_1,T_1,L_1) ... r_n(Z_{n-1},Y,T_n,L_n)
       [ ^ r'(X,Z',T',L') ]  ^ windows ^ radii``
    """

    head_rel: int
    atoms: List[Atom]
    branch: Optional[Atom] = None
    # ---- quality measures (Section 3.3 "Rule quality and filtering") ----
    support: int = 0            # supp
    coverage: float = 0.0       # cov
    confidence: float = 0.0     # conf
    pca: float = 0.0            # pca
    quality: float = 0.0        # s(rule) = (cov * conf * pca) ^ (1/3)
    source: str = "walk"        # walk | llm
    hits: int = 0

    @property
    def length(self) -> int:
        return len(self.atoms)

    @property
    def s_rule(self) -> float:
        """Table 1: s(rule), the fused quality used in Eq. (9) and Eq. (16)."""
        return float(self.quality)

    def signature(self) -> tuple:
        return (int(self.head_rel), tuple(int(a.rel) for a in self.atoms),
                -1 if self.branch is None else int(self.branch.rel))

    def to_dict(self) -> Dict:
        return {
            "head_rel": int(self.head_rel),
            "atoms": [a.to_dict() for a in self.atoms],
            "branch": None if self.branch is None else self.branch.to_dict(),
            "support": int(self.support),
            "coverage": float(self.coverage),
            "confidence": float(self.confidence),
            "pca": float(self.pca),
            "quality": float(self.quality),
            "source": self.source,
        }

    @classmethod
    def from_dict(cls, d: Dict) -> "Rule":
        return cls(
            head_rel=int(d["head_rel"]),
            atoms=[Atom.from_dict(a) for a in d["atoms"]],
            branch=None if d.get("branch") is None else Atom.from_dict(d["branch"]),
            support=int(d.get("support", 0)),
            coverage=float(d.get("coverage", 0.0)),
            confidence=float(d.get("confidence", 0.0)),
            pca=float(d.get("pca", 0.0)),
            quality=float(d.get("quality", 0.0)),
            source=d.get("source", "walk"),
        )

    def to_text(self) -> str:
        head = f"r{self.head_rel}(X,Y)"
        body = ", ".join(
            f"r{a.rel}({'X' if i == 0 else f'Z{i}'},"
            f"{'Y' if i == len(self.atoms) - 1 else f'Z{i + 1}'}|w={a.window:g}|rho={a.radius:g})"
            for i, a in enumerate(self.atoms)
        )
        if self.branch is not None:
            body += f", r{self.branch.rel}(X,Z'|w={self.branch.window:g}|rho={self.branch.radius:g})"
        return f"{head} <- {body}"


class RuleBase:
    """Omega: the rule base, indexed by head relation (Omega_r)."""

    def __init__(self, rules: Optional[Sequence[Rule]] = None):
        self.rules: List[Rule] = list(rules or [])
        self.by_head: Dict[int, List[Rule]] = {}
        self.reindex()

    def reindex(self) -> None:
        self.by_head = {}
        for r in self.rules:
            self.by_head.setdefault(r.head_rel, []).append(r)
        for k in self.by_head:
            self.by_head[k].sort(key=lambda r: -r.quality)

    def add(self, rule: Rule) -> None:
        self.rules.append(rule)
        self.by_head.setdefault(rule.head_rel, []).append(rule)

    def for_relation(self, rel: int) -> List[Rule]:
        return self.by_head.get(int(rel), [])

    def filter(self, *, use_branch: bool = True, use_llm: bool = True,
               min_pca: float = 0.0, min_support: int = 0) -> "RuleBase":
        out = []
        for r in self.rules:
            if not use_branch and r.branch is not None:
                continue
            if not use_llm and r.source == "llm":
                continue
            if r.pca < min_pca or r.support < min_support:
                continue
            out.append(r)
        return RuleBase(out)

    def __len__(self) -> int:
        return len(self.rules)

    def __iter__(self):
        return iter(self.rules)

    # ---- IO ----
    def save(self, path: str) -> None:
        save_json([r.to_dict() for r in self.rules], path)

    @classmethod
    def load(cls, path: str) -> "RuleBase":
        return cls([Rule.from_dict(d) for d in load_json(path)])

    def stats(self) -> Dict[str, float]:
        if not self.rules:
            return {"num_rules": 0}
        return {
            "num_rules": len(self.rules),
            "num_branch": sum(1 for r in self.rules if r.branch is not None),
            "num_llm": sum(1 for r in self.rules if r.source == "llm"),
            "mean_quality": float(np.mean([r.quality for r in self.rules])),
            "mean_length": float(np.mean([r.length for r in self.rules])),
        }