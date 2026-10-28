# -*- coding: utf-8 -*-
"""End-to-end PhyWorld-KGC model: Stage I -> Stage II -> Stage III.

Eq. (16)  Sim_evo(o) = (h_sq * h_rq)^T h_o + lambda_roll * g_hat_q[o]
                        + lambda_rule * sum_rules s(rule) * 1_rule(o)
Eq. (19)  P_hat(o|q) ∝ exp(lambda * Sim_evo + (1 - lambda) * R_agent)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..config import AblationConfig, ExperimentConfig
from .astsc import AdaptiveSpatioTemporalStateConstructor, GateTable, SparseStateTensor
from .reasoner import EntityHistory, ReasonerOutput, SpatioTemporalConstraintReasoner
from .simulator import RuleGuidedWorldKnowledgeEvolutionSimulator, SimulatorOutput


@dataclass
class PhyWorldOutput:
    scores: torch.Tensor            # (B, |E|) final scores used for ranking
    sim_scores: torch.Tensor        # (B, |E|) Eq. (16), before Stage III
    h_sq: torch.Tensor
    state: SparseStateTensor
    sim: SimulatorOutput
    reasoner: Optional[ReasonerOutput] = None
    wm_cell: Optional[torch.Tensor] = None
    #: C(q): Top-K candidate set of Eq. (16) (None before Stage III).
    C_q: Optional[torch.Tensor] = None

    # ---- Table 1 aliases ------------------------------------------------
    @property
    def Sim_evo(self) -> torch.Tensor:
        """Eq. (16): the evolution-consistency scores before Stage III."""
        return self.sim_scores

    @property
    def P_o_q(self):
        """Eq. (19): P(o|q) over C(q), or None if Stage III did not run."""
        return None if self.reasoner is None else self.reasoner.P_o_q

    @property
    def o_hat_q(self):
        """o_hat_q = argmax_{o in C(q)} P(o|q)."""
        return None if self.reasoner is None else self.reasoner.o_hat_q


class PhyWorldKGC(nn.Module):
    def __init__(self, num_entities: int, num_relations: int, cfg: ExperimentConfig,
                 num_texts: int = 0, text_dim: int = 768,
                 ring_distance=None):
        super().__init__()
        self.cfg = cfg
        self.num_entities = int(num_entities)
        self.num_relations = int(num_relations)
        self.astsc = AdaptiveSpatioTemporalStateConstructor(
            num_entities, num_relations, cfg.model, cfg.data,
            num_texts=num_texts, text_dim=text_dim, ring_distance=ring_distance,
        )
        self.simulator = RuleGuidedWorldKnowledgeEvolutionSimulator(
            num_entities, num_relations, cfg.model, cfg.data,
        )
        d = cfg.model.embed_dim
        self.fuse = nn.Linear(cfg.model.c_out + cfg.model.c_in + d, d)
        self.fuse_drop = nn.Dropout(cfg.model.dropout)
        #: keeps the query representation at a fixed scale so that the bilinear
        #: score of Eq. (16) neither vanishes nor explodes during training
        self.fuse_norm = nn.LayerNorm(d)
        #: frozen copy of the pretrained entity table (``E_fix`` of Section 3.2)
        self.ent_static: Optional[nn.Embedding] = None
        self.reasoner = SpatioTemporalConstraintReasoner(cfg.reasoner, num_entities)

    # ------------------------------------------------------------------
    def load_pretrained(self, path: str, verbose: bool = True) -> None:
        """Load the static KG embeddings of Section 3.2 into the embedding tables.

        The file is the output of ``scripts/pretrain_embeddings.py``
        (``phyworld.pretrain``): neighbour-aware DistMult embeddings pre-trained
        on the training graph, stored as ``entity_init`` / ``relation_init``.
        ``entity_init`` initialises ``E_ent`` and ``E_fix``; ``relation_init``
        holds ``2 x |R|`` rows — the forward relations used here, followed by the
        inverse relations that exist only inside pre-training.
        """
        ckpt = torch.load(path, map_location="cpu", weights_only=False)
        ent = ckpt.get("entity_init")
        rel = ckpt.get("relation_init")
        with torch.no_grad():
            if ent is not None:
                n = min(ent.shape[0], self.ent_emb.weight.shape[0])
                self.ent_emb.weight[:n].copy_(ent[:n].to(self.ent_emb.weight.dtype))
                if self.cfg.model.static_entity_mode == "pretrained_frozen":
                    table = torch.zeros_like(self.ent_emb.weight)
                    table[:n] = ent[:n].to(self.ent_emb.weight.dtype)
                    self.ent_static = nn.Embedding.from_pretrained(table, freeze=True)
            if rel is not None:
                n = min(rel.shape[0], self.rel_emb.weight.shape[0])
                self.rel_emb.weight[:n].copy_(rel[:n].to(self.rel_emb.weight.dtype))
        if verbose:
            print(f"[init] loaded pretrained embeddings from {path}")
        if ent is not None and int(ent.shape[0]) != self.num_entities:
            msg = (f"[init] WARNING: checkpoint has {ent.shape[0]} entities but the "
                   f"dataset has {self.num_entities}; the id spaces do not match and "
                   f"the pretrained rows will be misaligned. Load the exported "
                   f"ids/<dataset> layout instead.")
            print(msg)

    # ------------------------------------------------------------------
    @property
    def ent_emb(self) -> nn.Embedding:
        return self.astsc.encoder.ent_emb

    @property
    def rel_emb(self) -> nn.Embedding:
        return self.astsc.encoder.rel_emb

    # ------------------------------------------------------------------
    def encode(self, batch, gamma_table: Optional[GateTable] = None,
               rule_scores: Optional[List[Dict[int, float]]] = None,
               ablations: Optional[AblationConfig] = None):
        abl = ablations or AblationConfig()
        state = self.astsc(batch, gamma_table=gamma_table, ablations=abl)
        sim = self.simulator(state, batch, rule_scores=rule_scores, ablations=abl)

        z_st = state.z_agg if abl.ablate_astsc else sim.z_st
        fused = self.fuse(self.fuse_drop(torch.cat(
            [z_st, state.z_agg,
             (self.ent_static if self.ent_static is not None
              else self.ent_emb)(batch.src.to(z_st.device))], dim=-1)))
        # Eq. (15): h_sq = tanh(W [z_ST || z_Agg || E_fix]).  tanh keeps the
        # sign/mean structure of the DistMult embedding space, which a LayerNorm
        # would remove (its output is zero-mean, breaking the score geometry).
        h_sq = (torch.tanh(fused) if self.cfg.model.fuse_act == "tanh"
                else self.fuse_norm(fused))
        h_rq = self.rel_emb(batch.rel.to(z_st.device))

        # ---- Eq. (16) ----
        query = h_sq * h_rq
        scores = query @ self.ent_emb.weight.t()
        if sim.g_hat is not None and not abl.ablate_roll:
            # Eq. (16) adds lambda_roll * g_hat.  g_hat is supervised as a logit by
            # Eq. (21), so it is unbounded; tanh keeps the additive term inside the
            # scale of the bilinear term without changing its ranking direction.
            scores = scores + self.simulator.roll_scale * torch.tanh(sim.g_hat)
        if sim.rule_bonus is not None and not abl.ablate_rule_term:
            scores = scores + self.simulator.rule_scale * sim.rule_bonus
        # query-bucket cell V[s_q, r_q, tau'_q]  (masked during training, Eq. 21)
        wm_cell = None
        if sim.pair_key is not None and sim.pair_key.numel() > 0:
            B = int(batch.src.shape[0])
            qk = (torch.arange(B, device=scores.device) * self.num_entities
                  + batch.src.to(scores.device)) * self.num_relations \
                + batch.rel.to(scores.device)
            pos = torch.searchsorted(sim.pair_key, qk)
            pos = pos.clamp(max=sim.pair_key.numel() - 1)
            hit = sim.pair_key[pos] == qk
            wm_cell = torch.zeros((B, sim.v_pairs.shape[-1]), device=scores.device,
                                  dtype=sim.v_pairs.dtype)
            wm_cell[hit] = sim.v_pairs[pos[hit]]

        return PhyWorldOutput(scores=scores, sim_scores=scores, h_sq=h_sq,
                              state=state, sim=sim, wm_cell=wm_cell)

    # ------------------------------------------------------------------
    def forward(self, batch, gamma_table: Optional[GateTable] = None,
                rule_scores: Optional[List[Dict[int, float]]] = None,
                history: Optional[EntityHistory] = None,
                ablations: Optional[AblationConfig] = None,
                use_reasoner: bool = True) -> PhyWorldOutput:
        out = self.encode(batch, gamma_table, rule_scores, ablations)
        abl = ablations or AblationConfig()
        if not use_reasoner:
            return out

        k = min(int(self.cfg.reasoner.top_k), out.scores.shape[-1])
        cand_scores, cand_idx = torch.topk(out.scores, k=k, dim=-1)
        res = self.reasoner(out.scores, batch, cand_idx, cand_scores,
                            history=history, ablations=abl)
        res.finalize(cand_idx)          # C(q), P(o|q), o_hat_q
        out.scores = res.scores
        out.reasoner = res
        out.C_q = cand_idx
        return out