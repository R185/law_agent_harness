"""Decision layer: coverage recommendation and draft fact check.

Importing this package does not open a socket and does not read an API key.
Outcome functions are exported by the story that implements them.
"""

from decision.config import DecisionConfig, FailureReason, LayerStatus, ModelMode
from decision.coverage import (
    Coverage,
    CoverageRequest,
    Decision,
    DialogueRole,
    DialogueTurn,
    Gap,
    MissingField,
    PremiseError,
    ReasonCode,
    Recommendation,
    evaluate,
)
from decision.factcheck import (
    Action,
    Claim,
    ClaimKind,
    FactCheckInput,
    FactCheckVerdict,
    FactFinding,
    FindingCode,
    Severity,
    fact_check,
)
from decision.questioner import Clarification

__all__ = [
    "Action",
    "Claim",
    "ClaimKind",
    "Clarification",
    "Coverage",
    "CoverageRequest",
    "Decision",
    "DecisionConfig",
    "DialogueRole",
    "DialogueTurn",
    "evaluate",
    "fact_check",
    "FactCheckInput",
    "FactCheckVerdict",
    "FactFinding",
    "FindingCode",
    "FailureReason",
    "Gap",
    "LayerStatus",
    "MissingField",
    "ModelMode",
    "PremiseError",
    "ReasonCode",
    "Recommendation",
    "Severity",
]
