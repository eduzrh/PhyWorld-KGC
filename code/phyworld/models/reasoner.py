# -*- coding: utf-8 -*-
"""Stage III: SpatioTemporalConstraintReasoner, Section 3.4 + Algorithm 1.

Eq. (17) entropy trigger     -> entropy / should_trigger
Eq. (18) velocity check      -> velocity_plausibility
Algorithm 1 meta evaluation  -> MetaEvaluation
Eq. (19) lambda fusion       -> fuse
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from ..config import AblationConfig, ReasonerConfig
from ..rules.llm import LLMJudge  # noqa: F401  (optional OpenAI backend)
from ..utils import haversine_km


@dataclass
class ReasonerOutput:
    scores: torch.Tensor            # (B, |E|) or (B, K) fused scores
    triggered: torch.Tensor         # (B,) bool
    entropy: torch.Tensor           # (B,)
    r_agent: Optional[torch.Tensor]  # (B, K)
    v_impl: Optional[torch.Tensor]   # (B, K)
    v_max: float
    conflicts: List[List[Tuple[int, int]]] = field(default_factory=list)
    rounds: int = 1
    # ---- Section 3.4 / Table 1 -----------------------------------------
    #: C(q): the Top-K candidate set produced by Sim_evo (Eq. 16).
    C_q: Optional[torch.Tensor] = None
    #: P(o|q): Eq. (19) probability over C(q).
    P_o_q: Optional[torch.Tensor] = None
    #: o_hat_q = argmax_{o in C(q)} P(o|q).
    o_hat_q: Optional[torch.Tensor] = None

    @property
    def R_agent(self) -> Optional[torch.Tensor]:
        """Algorithm 1: the meta-evaluation score R_agent(o)."""
        return self.r_agent

    @property
    def H_q(self) -> torch.Tensor:
        """Eq. (17): the candidate-set entropy H(q) used for the delta test."""
        return self.entropy

    def finalize(self, cand_idx: torch.Tensor) -> "ReasonerOutput":
        """Fill C(q), P(o|q) and o_hat_q from the fused scores."""
        self.C_q = cand_idx
        cand = self.scores.gather(1, cand_idx)
        prob = torch.softmax(cand, dim=-1)
        self.P_o_q = prob
        self.o_hat_q = cand_idx.gather(1, prob.argmax(dim=-1, keepdim=True)).squeeze(1)
        return self


class EntityHistory:
    """Most recent known location of every entity (used by Eq. 18)."""

    def __init__(self, num_entities: int, ent_times: Sequence[np.ndarray],
                 ent_lat: Sequence[np.ndarray], ent_lon: Sequence[np.ndarray]):
        self.num_entities = num_entities
        self.times = ent_times
        self.lat = ent_lat
        self.lon = ent_lon

    @classmethod
    def from_arrays(cls, num_entities: int, subj: np.ndarray, ts: np.ndarray,
                    lat: np.ndarray, lon: np.ndarray) -> "EntityHistory":
        order = np.lexsort((ts, subj))
        subj, ts, lat, lon = subj[order], ts[order], lat[order], lon[order]
        bounds = np.searchsorted(subj, np.arange(num_entities + 1))
        times, lats, lons = [], [], []
        for e in range(num_entities):
            s, t = bounds[e], bounds[e + 1]
            times.append(ts[s:t])
            lats.append(lat[s:t])
            lons.append(lon[s:t])
        return cls(num_entities, times, lats, lons)

    def last_before(self, ent: int, t: float) -> Tuple[float, float, float]:
        ts = self.times[ent]
        if ts.size == 0:
            return np.nan, np.nan, np.nan
        i = int(np.searchsorted(ts, t, side="right")) - 1
        if i < 0:
            return np.nan, np.nan, np.nan
        return float(self.lat[ent][i]), float(self.lon[ent][i]), float(ts[i])


class MetaEvaluation:
    """Algorithm 1: cooperative meta-evaluation that produces R_agent(o).

    The loop is ``N_meta`` rounds of judge calls over the candidate card; the
    rounds are what the ``w/o Meta`` ablation collapses to a single pass.  The
    heavy lifting lives in the judge backend (``LLMJudge`` offline / OpenAI);
    this class owns the iteration and the velocity/conflict conditioning so the
    Stage-III module can be written exactly as the paper pseudocode.
    """

    def __init__(self, judge):
        self.judge = judge
        #: number of judge calls issued so far (Table 8 accounting)
        self.calls = 0

    def run(self, payload, n_candidates: int, rounds: int,
            ablate_meta: bool = False, velocity_ok=None,
            v_impl=None, v_max: float = 900.0):
        """Return ``R_agent`` for one query (Algorithm 1, lines 4-11)."""
        r = int(1 if ablate_meta else max(1, rounds))
        self.calls += r
        return self.judge.score(payload, n_candidates=n_candidates, rounds=r,
                                ablate_meta=ablate_meta,
                                velocity_ok=velocity_ok, v_impl=v_impl,
                                v_max=v_max)


class SpatioTemporalConstraintReasoner(nn.Module):
    """Stage III: entropy-triggered, constraint-checked, meta-evaluated reranking."""

    def __init__(self, cfg: ReasonerConfig, num_entities: int):
        super().__init__()
        self.cfg = cfg
        self.num_entities = int(num_entities)
        self.judge = LLMJudge(cfg)
        #: Algorithm 1 -- the meta-evaluation loop that yields R_agent.
        self.meta_eval = MetaEvaluation(self.judge)
        # fusion weights (lambda of Eq. 19 is a scalar; the rest are judge features)
        self.lam = float(cfg.fusion_weight)

    # ------------------------------------------------------------------
    # Eq. (17)
    # ------------------------------------------------------------------
    @staticmethod
    def entropy(scores: torch.Tensor, temperature: float = 1.0) -> torch.Tensor:
        p = torch.softmax(scores / max(temperature, 1e-6), dim=-1)
        logp = torch.log(p.clamp_min(1e-12))
        return -(p * logp).sum(-1)

    def should_trigger(self, scores: torch.Tensor,
                       ablate_entropy: bool = False) -> torch.Tensor:
        if ablate_entropy:                       # w/o Entr.: always cascade
            return torch.ones(scores.shape[0], dtype=torch.bool, device=scores.device)
        return self.entropy(scores, self.cfg.temperature) > self.cfg.entropy_threshold

    # ------------------------------------------------------------------
    # Eq. (18)
    # ------------------------------------------------------------------
    def velocity_plausibility(self, q_lat: float, q_lon: float, q_t: float,
                              cand_loc: torch.Tensor, cand_t: torch.Tensor,
                              eps_v: float) -> torch.Tensor:
        """v_impl = d_geo(l_q, l_o) / (|t_q - t_o| + eps_v); violated if > v_max."""
        d = torch.as_tensor(
            [haversine_km(q_lat, q_lon, float(x), float(y))
             for x, y in cand_loc.detach().cpu().tolist()], dtype=torch.float32,
            device=cand_loc.device,
        )
        denom = (q_t - cand_t).abs() + eps_v
        return d / denom.clamp_min(1e-6)

    # ------------------------------------------------------------------
    def forward(self, sim: torch.Tensor, batch, cand_idx: torch.Tensor,
                    cand_scores: torch.Tensor, history: Optional[EntityHistory] = None,
                    ablations: Optional[AblationConfig] = None) -> ReasonerOutput:
        """Rerank the Top-K candidates by Eq. (19).

        All geometry is computed once per batch with numpy; only the fused scores
        are written back into the (B, |E|) tensor, so Stage III costs
        ``O(B * K^2)`` instead of touching the full entity set.
        """
        abl = ablations or AblationConfig()
        B = int(sim.shape[0])
        device = sim.device
        # Eq. (17): H(q) = -sum_o P(o|q) log P(o|q), where the distribution is
        # normalised over the candidate set C(q) (not over the full vocabulary).
        H_q = self.entropy(cand_scores, self.cfg.temperature)
        ent = H_q
        fused = sim.clone()

        if abl.ablate_reasoner:
            return ReasonerOutput(scores=sim,
                                  triggered=torch.zeros(B, dtype=torch.bool, device=device),
                                  entropy=ent, r_agent=None, v_impl=None,
                                  v_max=float(self.cfg.v_max_kmh))

        trig = self.should_trigger(cand_scores, ablate_entropy=abl.ablate_entropy)
        # Table 8: budget the (expensive) agent calls.  When call_ratio < 1 the
        # budget is spent on the highest-entropy queries -- exactly those whose
        # candidate set is most ambiguous -- and every other triggered query
        # keeps the Stage-II ranking.
        ratio = float(getattr(self.cfg, "call_ratio", 1.0))
        if 0.0 < ratio < 1.0:
            idx = torch.nonzero(trig, as_tuple=False).squeeze(1)
            budget = max(1, int(round(ratio * int(idx.numel()))))
            if idx.numel() > budget:
                keep = torch.zeros_like(trig)
                order = torch.argsort(H_q[idx], descending=True)[:budget]
                keep[idx[order]] = True
                trig = keep
        K = int(cand_idx.shape[1])
        r_agent = torch.zeros_like(cand_scores)
        v_all = torch.zeros_like(cand_scores)
        conflicts: List[List[Tuple[int, int]]] = [[] for _ in range(B)]
        eps_v = float(getattr(self.cfg, "eps_v", 0.5))

        if not bool(trig.any()):
            return ReasonerOutput(scores=fused, triggered=trig, entropy=ent,
                                  r_agent=r_agent, v_impl=v_all,
                                  v_max=float(self.cfg.v_max_kmh), rounds=1)

        cand_np = cand_idx.detach().cpu().numpy()
        loc_np = batch.loc.detach().cpu().numpy()
        ts_np = batch.ts.detach().cpu().numpy()
        src_np = batch.src.detach().cpu().numpy()
        rel_np = batch.rel.detach().cpu().numpy()
        # Algorithm 1, line 3: the velocity check only applies to relations in
        # R_co (annotated during rule mining, Section 3.3).
        coocc_np = (batch.coocc.detach().cpu().numpy()
                    if getattr(batch, "coocc", None) is not None
                    else np.ones(B, dtype=bool))

        for b in range(B):
            if not bool(trig[b]):
                continue
            q_lat, q_lon, q_t = float(loc_np[b, 0]), float(loc_np[b, 1]), float(ts_np[b])

            cand_lat = np.full(K, np.nan)
            cand_lon = np.full(K, np.nan)
            cand_t = np.full(K, q_t)
            facts: List[str] = ["unknown"] * K
            if history is not None and not np.isnan(q_lat):
                for j, o in enumerate(cand_np[b].tolist()):
                    la, lo, tt = history.last_before(int(o), q_t)
                    if not np.isnan(la):
                        cand_lat[j], cand_lon[j], cand_t[j] = la, lo, tt
                        facts[j] = f"({la:.4f}, {lo:.4f}) @ t={tt:.1f}"

            # Eq. (18): v_impl = d_geo(l_q, l_o) / (|t_q - t_o| + eps_v)
            is_co = bool(coocc_np[b])
            v_impl = np.full(K, np.nan)
            if is_co and not np.isnan(q_lat):
                ok_loc = ~np.isnan(cand_lat)
                if ok_loc.any():
                    v_impl[ok_loc] = (haversine_km(q_lat, q_lon,
                                                   cand_lat[ok_loc], cand_lon[ok_loc])
                                      / (np.abs(q_t - cand_t[ok_loc]) + eps_v))
            ok = ((~np.isnan(v_impl)) & (v_impl <= self.cfg.v_max_kmh)
                  if is_co else np.ones(K, dtype=bool))

            # pairwise conflicts F (vectorised K x K)
            conf_b: List[Tuple[int, int]] = []
            idx_ok = np.nonzero(~np.isnan(cand_lat))[0]
            if idx_ok.size > 1:
                la = cand_lat[idx_ok][:, None]
                lo = cand_lon[idx_ok][:, None]
                dmat = haversine_km(la, lo, la.T, lo.T)
                dt = np.abs(cand_t[idx_ok][:, None] - cand_t[idx_ok][None, :]) + eps_v
                bad = (dmat / dt) > self.cfg.v_max_kmh
                ii, jj = np.triu_indices(idx_ok.size, k=1)
                sel = bad[ii, jj]
                for a, c in zip(ii[sel].tolist(), jj[sel].tolist()):
                    conf_b.append((int(idx_ok[a]), int(idx_ok[c])))

            payload = self._judge_payload(
                src=int(src_np[b]), rel=int(rel_np[b]), t=q_t,
                q_loc=(q_lat, q_lon), cand=cand_np[b].tolist(), facts=facts,
                v_impl=v_impl, conflicts=conf_b, coocc=is_co,
            )
            # NaN marks "velocity not evaluated" (non-co-occurrence relation or
            # missing location); the surrogate maps it to ratio 0.
            v_t = torch.as_tensor(np.nan_to_num(v_impl, nan=0.0), dtype=torch.float32)
            rounds = 1 if abl.ablate_meta else max(1, int(self.cfg.n_meta))
            r_b = self.meta_eval.run(
                payload, n_candidates=K, rounds=rounds,
                ablate_meta=abl.ablate_meta,
                velocity_ok=torch.as_tensor(ok), v_impl=v_t,
                v_max=self.cfg.v_max_kmh).to(device)
            r_agent[b] = r_b
            v_all[b] = v_t.to(device)
            conflicts[b] = conf_b
            fused[b, cand_idx[b]] = (self.lam * cand_scores[b] + (1.0 - self.lam) * r_b
                                     - (1.0 - self.lam) * 5.0
                                     * torch.as_tensor(~ok, device=device).float())

        return ReasonerOutput(scores=fused, triggered=trig, entropy=ent,
                              r_agent=r_agent, v_impl=v_all,
                              v_max=float(self.cfg.v_max_kmh),
                              conflicts=conflicts, rounds=1)

        # ------------------------------------------------------------------
    def _judge_payload(self, src: int, rel: int, t: float, q_loc,
                       cand: List[int], facts: List[str],
                       v_impl: np.ndarray, conflicts: List[Tuple[int, int]],
                       coocc: bool = False) -> Dict:
        """[Query]/[Rule grounding evidence]/[Candidates]/[Pairwise conflicts F] card."""
        query = {
            "subject": int(src), "relation": int(rel), "coocc": bool(coocc),
            "time": float(t),
            "location": ("unknown" if np.isnan(q_loc[0])
                         else f"({q_loc[0]:.4f}, {q_loc[1]:.4f})"),
            "rule_evidence": [],
        }
        candidates = [
            {"candidate": int(e), "fact": facts[i],
             "v_impl": float(v_impl[i]), "v_max": float(self.cfg.v_max_kmh)}
            for i, e in enumerate(cand)
        ]
        return {"query": query, "candidates": candidates, "conflicts": conflicts}
# ------------------------------------------------------------------
