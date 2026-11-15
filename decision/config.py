"""Decision-layer configuration and failure vocabulary.

``evaluate`` and ``fact_check`` turn an unusable relevance threshold into
``layer_status=failed`` and ``failure_reason=config_invalid``. This module
stores the raw threshold and reports it. It does not raise for that case
and it does not read an API key.
"""

from __future__ import annotations

import math
import os
from enum import Enum
from numbers import Number
from typing import Any

from pydantic import BaseModel, ConfigDict, Field


class ModelMode(str, Enum):
    """Whether the optional classifier runs."""

    off = "off"
    on = "on"


class LayerStatus(str, Enum):
    """Outcome of reading the contract and, when enabled, the classifier."""

    ok = "ok"
    degraded = "degraded"
    failed = "failed"


class FailureReason(str, Enum):
    """Why ``layer_status`` is ``degraded`` or ``failed``."""

    contract_invalid = "contract_invalid"
    config_invalid = "config_invalid"
    model_timeout = "model_timeout"
    model_network = "model_network"
    model_invalid_json = "model_invalid_json"
    model_schema = "model_schema"
    model_unconfigured = "model_unconfigured"


def _model_mode_from_env() -> ModelMode:
    """Default ``model_mode`` from ``DECISION_MODEL_MODE``.

    Only the strings ``on`` and ``off`` are accepted. An unset variable and
    every other value mean ``off``. Used as a ``default_factory``, so an
    explicit constructor argument does not read the environment.
    """
    raw = os.environ.get("DECISION_MODEL_MODE")
    if raw == "on":
        return ModelMode.on
    if raw == "off":
        return ModelMode.off
    return ModelMode.off


def _relevance_threshold_invalid(value: Any) -> bool:
    """True unless ``value`` is a finite real number on the closed interval 0..1.

    ``bool`` is invalid even though it is an ``int``. Strings are stored raw
    and are not parsed. ``Decimal`` is a ``Number`` but not a ``Real``.
    NaN and infinities are invalid. This helper does not raise.
    """
    try:
        # bool subclasses int and compares equal to 0 or 1. numpy.bool is a
        # separate type whose name is also "bool".
        if isinstance(value, bool) or type(value).__name__ == "bool":
            return True
        # Do not coerce numeric strings such as "0.5".
        if isinstance(value, (str, bytes, bytearray, complex)):
            return True
        if isinstance(value, float):
            return bool((not math.isfinite(value)) or value < 0.0 or value > 1.0)
        # int, Decimal, Fraction, and other real numbers. Decimal NaN raises.
        if isinstance(value, Number):
            return not bool(0 <= value <= 1)
    except Exception:
        return True
    return True


class DecisionConfig(BaseModel):
    """Starting settings for coverage and draft check.

    ``min_relevance_semantic`` and ``min_relevance_lexical`` are sufficiency
    thresholds on the closed interval from 0 to 1. There is no threshold
    field for ``order_only``. ``max_clarify_margin`` is not a sufficiency
    threshold: a top-two margin strictly below it can open the first
    clarification case, and a margin equal to it or above it does not.

    Out-of-range, non-numeric, boolean, NaN, and infinite relevance
    thresholds are stored as given. ``thresholds_invalid`` reports them.
    ``model_mode`` is taken from ``DECISION_MODEL_MODE`` only when the caller
    omits it.
    """

    model_config = ConfigDict(extra="ignore")

    min_relevance_semantic: Any = 0.35
    min_relevance_lexical: Any = 0.20
    max_clarify_margin: float = 0.05
    model_mode: ModelMode = Field(default_factory=_model_mode_from_env)
    model_timeout_seconds: int = 20
    check_rules: bool = False

    def thresholds_invalid(self) -> bool:
        """Return True when either relevance threshold cannot be used.

        Does not raise. Handles map a True result to ``layer_status=failed``
        and ``failure_reason=config_invalid``.
        """
        return bool(
            _relevance_threshold_invalid(self.min_relevance_semantic)
            or _relevance_threshold_invalid(self.min_relevance_lexical)
        )


__all__ = [
    "DecisionConfig",
    "FailureReason",
    "LayerStatus",
    "ModelMode",
]
