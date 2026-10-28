# -*- coding: utf-8 -*-
"""Configuration objects for PhyWorld-KGC.

Every field carries the symbol used in the VLDBJ manuscript so that the code and
the paper stay aligned (Table 1 / Section 3 / Figure 7 / Figure 9).
"""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass, field
from typing import List, Optional, Sequence


# ---------------------------------------------------------------------------
# Ablation registry (Figure 7 of the manuscript: 19 variants)
# ---------------------------------------------------------------------------

#: Group A - Stage I (adaptive spatio-temporal state constructor)
ABLATION_GROUP_A = ["w/o ASTSC", "w/o Loc.", "w/o Gate", "w/o S.Buck.", "w/o S.Sample"]

#: Group B.1 - Stage II architecture & evolution
ABLATION_GROUP_B1 = ["w/o Evol.", "w/o ST-Conv.", "w/o Roll.", "w/o L_geo"]

#: Group B.2 - Stage II logical rules
ABLATION_GROUP_B2 = [
    "w/o RuleGate",
    "w/o RuleTerm",
    "w/o S.Const",
    "w/o conf",
    "w/o Branch",
    "w/o LLM-Rule",
]

#: Group C - Stage III constraint reasoner
ABLATION_GROUP_C = ["w/o Reasoner", "w/o Co-occ.", "w/o Meta", "w/o Entr."]

ABLATIONS_ALL = ABLATION_GROUP_A + ABLATION_GROUP_B1 + ABLATION_GROUP_B2 + ABLATION_GROUP_C

#: mapping variant -> internal flag
ABLATION_FLAGS = {
    "w/o ASTSC": "ablate_astsc",
    "w/o Loc.": "ablate_loc",
    "w/o Gate": "ablate_gate",
    "w/o S.Buck.": "ablate_s_bucket",
    "w/o S.Sample": "ablate_s_sample",
    "w/o Evol.": "ablate_evol",
    "w/o ST-Conv.": "ablate_st_conv",
    "w/o Roll.": "ablate_roll",
    "w/o L_geo": "ablate_l_geo",
    "w/o RuleGate": "ablate_rule_gate",
    "w/o RuleTerm": "ablate_rule_term",
    "w/o S.Const": "ablate_s_const",
    "w/o conf": "ablate_conf",
    "w/o Branch": "ablate_branch",
    "w/o LLM-Rule": "ablate_llm_rule",
    "w/o Reasoner": "ablate_reasoner",
    "w/o Co-occ.": "ablate_coocc",
    "w/o Meta": "ablate_meta",
    "w/o Entr.": "ablate_entropy",
}


@dataclass
class AblationConfig:
    """Boolean switches for the 19 variants of Figure 7.

    ``active`` keeps the *paper* name (e.g. ``"w/o Gate"``) so that a run can be
    launched with ``--ablation "w/o Gate"`` and the log/result file carries the
    exact figure label.
    """

    active: List[str] = field(default_factory=list)

    # Group A -- Stage I
    ablate_astsc: bool = False      # w/o ASTSC : replace Stage I by mean-pooled facts
    ablate_loc: bool = False        # w/o Loc.
    ablate_gate: bool = False       # w/o Gate : fixed-weight concatenation
    ablate_s_bucket: bool = False   # w/o S.Buck.
    ablate_s_sample: bool = False   # w/o S.Sample

    # Group B.1 -- Stage II architecture
    ablate_evol: bool = False       # w/o Evol.
    ablate_st_conv: bool = False    # w/o ST-Conv.
    ablate_roll: bool = False       # w/o Roll.
    ablate_l_geo: bool = False      # w/o L_geo

    # Group B.2 -- Stage II rules
    ablate_rule_gate: bool = False  # w/o RuleGate
    ablate_rule_term: bool = False  # w/o RuleTerm
    ablate_s_const: bool = False    # w/o S.Const
    ablate_conf: bool = False       # w/o conf  (quality = confidence only)
    ablate_branch: bool = False     # w/o Branch
    ablate_llm_rule: bool = False   # w/o LLM-Rule

    # Group C -- Stage III
    ablate_reasoner: bool = False   # w/o Reasoner
    ablate_coocc: bool = False      # w/o Co-occ.
    ablate_meta: bool = False       # w/o Meta
    ablate_entropy: bool = False    # w/o Entr.

    def apply(self, names: Sequence[str]) -> "AblationConfig":
        for name in names:
            key = ABLATION_FLAGS.get(name)
            if key is None:
                raise ValueError(
                    f"Unknown ablation {name!r}. Valid names: {ABLATIONS_ALL}"
                )
            setattr(self, key, True)
        self.active = list(names)
        return self

    @classmethod
    def from_names(cls, names: Sequence[str]) -> "AblationConfig":
        return cls().apply(names)

    def label(self) -> str:
        return "+".join(self.active) if self.active else "full"


@dataclass
class DataConfig:
    """Dataset / neighbourhood configuration."""

    data_dir: str = "data"
    rules_dir: str = "rules"              # cached rule base (<dataset>.json)
    dataset: str = "PhyICEWS14"
    setting: str = "interpolation"          # interpolation | extrapolation
    #: on-disk layout: "auto" prefers ids/<dataset> when it exists
    layout: str = "auto"
    time_granularity: int = 1               # one unit = one day (ICEWS/GDELT)
    #: k in Section 3.2 -- number of sampled facts
    num_neighbors: int = 50
    #: m / n -- past and future half-windows (in raw time units)
    window_past: int = 50
    window_future: Optional[int] = None     # None -> same as window_past
    #: Delta -- width of a time bucket
    bin_width: int = 1
    #: |T_q| -- number of time buckets (derived)
    num_time_buckets: Optional[int] = None
    #: |S_q| - 1 -- number of spatial rings (Figure 9 default = 4)
    spatial_rings: int = 4
    #: resolution (km) of the innermost geographic grid cell
    spatial_resolution_km: float = 200.0
    #: cap on the induced subgraph used for rule grounding
    max_induced_facts: int = 2000
    #: scope of rule grounding: "stng" (sampled facts) | "induced" (paper, Eq. 16)
    rule_ground_scope: str = "induced"
    #: number of text-location embeddings kept in the cache
    text_encoder: str = "auto"              # auto | bert | hash
    geocoder: str = "builtin"               # builtin | nominatim | none
    seed: int = 42

    def future_window(self) -> int:
        return self.window_past if self.window_future is None else int(self.window_future)

    def time_buckets(self) -> int:
        if self.num_time_buckets is not None:
            return int(self.num_time_buckets)
        import math

        return int(math.ceil((self.window_past + self.future_window() + 1) / self.bin_width))


@dataclass
class ModelConfig:
    """Neural architecture of Stage I and Stage II.

    Symbols: d (facet dim), C_in, C_out, K_t, D = {d'_1..d'_N}, L_c.
    """

    embed_dim: int = 200                # d
    score_func: str = "distmult"        # distmult | complex | bique
    static_entity_mode: str = "pretrained_frozen"
    #: activation of Eq. (15): "tanh" (paper) or "layernorm"
    fuse_act: str = "tanh"
    init_emb: Optional[str] = None       # static KG init, see scripts/pretrain_embeddings.py

    # ---- Stage I: Encoder_e_r_t_l ----
    facet_dim: Optional[int] = None     # d for facets; None -> embed_dim
    attn_heads: int = 1
    gate_temperature: float = 0.1       # kappa_u in Eq. (3)
    c_in: int = 128                     # C_in, dimension of c(e,r,t,l)
    use_text_encoder: bool = True
    text_dim: int = 768

    # ---- Stage II: tensor layout ----
    max_entities: int = 60              # slot budget |E_q|
    max_relations: int = 60             # slot budget |R_q|
    kernel_size: int = 3                # K_t
    dilations: Sequence[int] = (1, 2, 4)  # D
    num_conv_layers: int = 2            # L_c
    c_out: int = 128                    # C_out
    shuffle_groups: int = 4             # feature-shuffling groups in Eq. (13)
    traj_chunk: int = 512               # trajectories per temporal-conv chunk
    gate_hidden_dim: Optional[int] = None
    dropout: float = 0.2

    # ---- Stage II: rule gating / scoring ----
    gate_floor: float = 0.2             # epsilon in Eq. (9)
    lambda_roll: float = 1.0            # lambda_roll in Eq. (16)
    lambda_rule: float = 1.0            # lambda_rule in Eq. (16)
    rule_topk: int = 50                 # K in C(q)

    # ---- training objective (Section 3.5) ----
    l2_lambda: float = 1e-6             # lambda_2 in Eq. (20)
    mu: float = 0.5                     # mu  : weight of L_wm
    nu: float = 0.1                     # nu  : weight of L_geo
    geo_scale_xi: float = 1e-3          # xi  in Eq. (22)
    wm_batch_frac: float = 0.5          # fraction of the batch used for L_wm


@dataclass
class RuleConfig:
    """Offline rule mining (Section 3.3 left part, Appendix A.3)."""

    max_length: int = 3                 # n_max
    walks_per_fact: int = 20            # W
    max_rules_per_relation: int = 40
    quantile: float = 0.90              # empirical quantile for omega/rho
    min_support: int = 5                # n_min
    pca_threshold: float = 0.1          # c_min (Figure 9)
    quality_sample: int = 3000          # subjects sampled for quality estimation
    use_branch: bool = True
    use_llm: bool = True
    llm_topk: int = 20                  # top-20 chain candidates rewritten for the LLM
    llm_max_new: int = 20
    #: co-occurrence annotation
    coocc_distance_km: float = 50.0     # d_co
    coocc_fraction: float = 0.8         # pi_co
    seed: int = 42


@dataclass
class ReasonerConfig:
    """Stage III (Section 3.4)."""

    enabled: bool = True
    entropy_threshold: float = 0.8      # delta (Figure 9)
    temperature: float = 1.0            # tau_e in Eq. (17)
    fusion_weight: float = 0.8          # lambda in Eq. (19)
    n_meta: int = 2                     # N_meta in Algorithm 1
    top_k: int = 10                     # |C(q)|
    #: velocity bound used by Eq. (18)
    v_max_kmh: float = 900.0            # v_max
    eps_v: float = 0.5                  # eps_v: half a time granularity
    #: LLM backend: "offline" (deterministic symbolic judge) or "openai"
    backend: str = "offline"
    model: str = "gpt-5.4-mini"
    api_base: Optional[str] = None
    api_key_env: str = "OPENAI_API_KEY"
    temperature_llm: float = 0.0
    max_retries: int = 2
    cache_path: Optional[str] = None
    #: fraction of triggered calls actually issued (Table 8 stress test)
    call_ratio: float = 1.0


@dataclass
class TrainConfig:
    epochs: int = 30
    batch_size: int = 32
    lr: float = 1e-3
    weight_decay: float = 1e-6
    grad_clip: float = 1.0
    eval_every: int = 1
    num_workers: int = 0
    device: str = "cuda"
    use_amp: bool = False
    seed: int = 42
    save_dir: str = "checkpoints"
    run_name: str = "phyworld"
    log_dir: str = "results"
    patience: int = 8
    #: debug knob: subsample the training facts (None = use all)
    max_train_facts: Optional[int] = None


@dataclass
class ExperimentConfig:
    data: DataConfig = field(default_factory=DataConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    rules: RuleConfig = field(default_factory=RuleConfig)
    reasoner: ReasonerConfig = field(default_factory=ReasonerConfig)
    train: TrainConfig = field(default_factory=TrainConfig)
    ablation: AblationConfig = field(default_factory=AblationConfig)

    def to_dict(self):
        return dataclasses.asdict(self)