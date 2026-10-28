"""PhyWorld-KGC: A Logical Rule-Guided World Knowledge Model for STKGC.

Reference: Zhao et al., "PhyWorld-KGC: A Logical Rule-Guided World Knowledge
Model for Spatio-Temporal Knowledge Graph Completion", The VLDB Journal.

Module map (paper section -> code):
  Section 3.2  AdaptiveSpatioTemporalStateConstructor  -> phyworld/models/astsc.py
  Section 3.3  RuleGuidedWorldKnowledgeEvolutionSimulator -> phyworld/models/simulator.py
  Section 3.4  SpatioTemporalConstraintReasoner        -> phyworld/models/reasoner.py
  Section 3.5  L_total = L_kgc + mu * L_wm + nu * L_geo -> phyworld/models/losses.py
  Appendix A   complexity / sparse storage             -> see README + astsc docstrings
"""

__version__ = "1.1.1"

from .config import (  # noqa: F401
    AblationConfig,
    DataConfig,
    ModelConfig,
    ReasonerConfig,
    RuleConfig,
    TrainConfig,
    ExperimentConfig,
)

__all__ = [
    "AblationConfig",
    "DataConfig",
    "ModelConfig",
    "ReasonerConfig",
    "RuleConfig",
    "TrainConfig",
    "ExperimentConfig",
]