"""Coverage policy loaded from the package, not from the process cwd."""

from __future__ import annotations

import json
from pathlib import Path

_POLICY_PATH = Path(__file__).resolve().parent / "coverage_policy.json"


def required_clauses(
    question_kind: str | None,
    policy: dict[str, list[str]] | None,
) -> list[str]:
    """Clause ids the orchestrator required for this question kind.

    ``policy is None`` reads the package file. A mapping replaces that file
    for one call. An unknown kind yields an empty list, and the layer does
    not guess the kind from the question text.
    """

    if policy is None:
        data = json.loads(_POLICY_PATH.read_text(encoding="utf-8"))
    else:
        data = policy
    if not question_kind or not isinstance(data, dict):
        return []
    values = data.get(question_kind)
    if not isinstance(values, list):
        return []
    return [str(item) for item in values]


def confirmed_clauses(retrieval, required: list[str]) -> tuple[list[str], list[str]]:
    """Split ``required`` into ids the retrieval confirmed and ids it did not."""

    found: set[str] = set()
    for document in retrieval.documents:
        found.update(document.matched_clauses)
    for item in retrieval.evidence:
        found.update(item.clause_ids)
    confirmed = [clause_id for clause_id in required if clause_id in found]
    missing = [clause_id for clause_id in required if clause_id not in found]
    return confirmed, missing
