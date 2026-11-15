"""Optional classifier. Imported only when the model mode is on.

Reading this module does not open a socket. The key is read from the
process at call time and is not written into logs, exceptions, or results.
"""

from __future__ import annotations

import json
import logging
import os
from pathlib import Path

from decision.config import FailureReason, LayerStatus

_LOGGER = logging.getLogger("decision.llm")
_SCHEMA_PATH = Path(__file__).resolve().parent / "prompts" / "classifier_schema.json"
_KEY_NAMES = ("DEEPSEEK_API_KEY", "LLM_API_KEY", "OPENAI_API_KEY")
_BASE_NAMES = ("DEEPSEEK_BASE_URL", "LLM_BASE_URL", "OPENAI_BASE_URL")
_MODEL_NAMES = ("DEEPSEEK_MODEL", "LLM_MODEL")
_DEFAULT_BASE = "https://api.deepseek.com"
_DEFAULT_MODEL = "deepseek-flash"

_TOP = {
    "kind",
    "suggested_decision",
    "reason_codes",
    "missing_fields",
    "candidate_doc_ids",
    "premise_errors",
    "suggested_action",
    "findings",
}
_DECISIONS = {
    "allow_answer",
    "reject_insufficient_kb",
    "reject_partial_coverage",
    "clarify",
}
_REASONS = {
    "topic_mismatch",
    "insufficient_relevance",
    "low_confidence",
    "sufficiency_scale_unavailable",
    "ambiguous_entity",
    "missing_identifier",
    "named_not_found",
    "partial_clause_coverage",
    "premise_error",
    "ambiguous_edition",
    "date_conflict",
    "edition_missing",
    "missing_facts",
    "reference_error",
}
_ACTIONS = {"accept", "rewrite", "refuse"}
_MISSING = {"contract", "counterparty", "period"}
_CODES = {
    "fabricated_citation",
    "citation_not_in_context",
    "citation_check_missing",
    "wrong_number",
    "wrong_date",
    "wrong_period",
    "wrong_document",
    "wrong_clause",
    "wrong_party",
    "wrong_edition",
    "repealed_doc",
    "contradicts_evidence",
    "unsupported_claim",
    "missing_citation",
    "unverifiable",
}
_SEVERITIES = {"low", "medium", "high"}
_HIGH_FORBIDDEN = {
    "citation_not_in_context",
    "citation_check_missing",
    "missing_citation",
    "unsupported_claim",
    "unverifiable",
}
_FINDING_KEYS = {
    "code",
    "severity",
    "claim_id",
    "quote",
    "evidence_ids",
    "expected",
    "actual",
    "fixable",
    "hint",
}
_PREMISE_KEYS = {"statement", "evidence_quote", "expected", "actual"}


class _ModelCallError(Exception):
    def __init__(self, reason: FailureReason) -> None:
        self.reason = reason


def _env_value(names: tuple[str, ...]) -> str:
    for name in names:
        raw = os.environ.get(name, "")
        if isinstance(raw, str) and raw.strip():
            return raw.strip()
    return ""


def _system_text() -> str:
    schema = _SCHEMA_PATH.read_text(encoding="utf-8")
    return (
        "Respond with one json object and no other text. "
        "The object must match this schema:\n"
        + schema
    )


def _degrade(core, reason: FailureReason):
    return core.model_copy(
        update={
            "layer_status": LayerStatus.degraded,
            "failure_reason": reason,
        }
    )


def _optional_strings(value: object, allowed: set[str] | None, *, limit: int | None) -> bool:
    if value is None:
        return True
    if not isinstance(value, list):
        return False
    if limit is not None and len(value) > limit:
        return False
    if any(not isinstance(item, str) for item in value):
        return False
    if allowed is not None and any(item not in allowed for item in value):
        return False
    if len(set(value)) != len(value):
        return False
    return True


def _premises_ok(value: object) -> bool:
    if value is None:
        return True
    if not isinstance(value, list):
        return False
    for item in value:
        if not isinstance(item, dict) or set(item) - _PREMISE_KEYS:
            return False
        if "statement" not in item or "actual" not in item:
            return False
        if not all(isinstance(item.get(key, ""), str) for key in item):
            return False
    return True


def _findings_ok(value: object) -> bool:
    if not isinstance(value, list):
        return False
    for item in value:
        if not isinstance(item, dict) or set(item) - _FINDING_KEYS:
            return False
        if not {"code", "severity", "claim_id", "fixable"} <= set(item):
            return False
        if item["code"] not in _CODES or item["severity"] not in _SEVERITIES:
            return False
        if not isinstance(item["claim_id"], str) or not isinstance(item["fixable"], bool):
            return False
        for key in ("quote", "expected", "actual", "hint"):
            if key in item and not isinstance(item[key], str):
                return False
        ids = item.get("evidence_ids")
        if ids is not None and not _optional_strings(ids, None, limit=None):
            return False
    return True


def _schema_ok(data: object, kind: str) -> bool:
    if not isinstance(data, dict) or set(data) - _TOP or data.get("kind") != kind:
        return False
    if kind == "coverage":
        if data.get("suggested_decision") not in _DECISIONS:
            return False
        if not _optional_strings(data.get("reason_codes"), _REASONS, limit=None):
            return False
        if "reason_codes" not in data:
            return False
        if not _optional_strings(data.get("missing_fields"), _MISSING, limit=3):
            return False
        if not _optional_strings(data.get("candidate_doc_ids"), None, limit=None):
            return False
        if not _premises_ok(data.get("premise_errors")):
            return False
        if "findings" in data and not _findings_ok(data["findings"]):
            return False
        if "suggested_action" in data and data["suggested_action"] not in _ACTIONS:
            return False
        return True
    if data.get("suggested_action") not in _ACTIONS or "findings" not in data:
        return False
    return _findings_ok(data["findings"])


def _grounded(value: object, parts: list[str]) -> bool:
    return isinstance(value, str) and bool(value) and any(value in part for part in parts)


def _coverage_rejected(model: dict, core, request, config) -> bool:
    decision = model["suggested_decision"]
    retrieval = request.retrieval
    if decision == "clarify":
        if not retrieval.documents and not retrieval.evidence:
            return True
        known = {
            candidate.doc_id
            for group in retrieval.ambiguity
            for candidate in group.candidates
        }
        for doc_id in model.get("candidate_doc_ids") or []:
            if doc_id not in known:
                return True
    if decision == "reject_partial_coverage":
        from decision.policy import required_clauses
        from decision.questioner import collect

        if not required_clauses(request.question_kind, request.policy):
            return True
        clarification = collect(
            request.question,
            request.dialogue,
            retrieval,
            config,
        )
        if clarification.applies:
            return True
    return decision != core.suggested_decision.value


def _open_claim_ids(core) -> set[str]:
    from decision.factcheck import FindingCode

    opened = set(core.unverified_claims)
    opened.update(
        item.claim_id
        for item in core.findings
        if item.code == FindingCode.unverifiable
    )
    return opened


def _finding_rejected(item: dict, request, core, parts: list[str]) -> bool:
    from decision.factcheck import ClaimKind

    code = item["code"]
    if item["severity"] == "high" and code in _HIGH_FORBIDDEN:
        return True
    expected = item.get("expected")
    actual = item.get("actual")
    if item["fixable"] and code != "citation_check_missing" and not _grounded(expected, parts):
        return True
    if code.startswith("wrong_") or code == "missing_citation":
        if not _grounded(expected, parts):
            return True
        if not isinstance(actual, str) or (
            actual not in request.answer and actual not in request.question
        ):
            return True
        claim = next(
            (one for one in (request.claims or []) if one.claim_id == item["claim_id"]),
            None,
        )
        if (
            claim is not None
            and claim.kind == ClaimKind.number
            and claim.subject
            and claim.subject.casefold()
            not in "\n".join(piece.text for piece in request.retrieval.evidence).casefold()
        ):
            return True
        if item["claim_id"] not in _open_claim_ids(core):
            return True
    elif item["claim_id"] not in _open_claim_ids(core):
        return True
    return False


def _fact_rejected(model: dict, core, request) -> bool:
    from decision.factcheck import _parts

    if model["suggested_action"] == "accept" and request.citation_check is None:
        return True
    parts = _parts(request.retrieval)
    return any(_finding_rejected(item, request, core, parts) for item in model["findings"])


def _apply_findings(core, model: dict):
    from decision.factcheck import FactFinding, FindingCode, Severity, _action

    incoming = model["findings"]
    replaced = {item["claim_id"] for item in incoming}
    kept = [item for item in core.findings if item.claim_id not in replaced]
    built = [
        FactFinding(
            code=FindingCode(item["code"]),
            severity=Severity(item["severity"]),
            claim_id=item["claim_id"],
            claim=item.get("quote") or item["claim_id"],
            quote=item.get("quote") or "",
            evidence_ids=list(item.get("evidence_ids") or []),
            expected=item.get("expected"),
            actual=item.get("actual"),
            fixable=item["fixable"],
            hint=item.get("hint") or "Use the value from the materials.",
        )
        for item in incoming
    ]
    findings = kept + built
    return core.model_copy(
        update={
            "suggested_action": _action(findings),
            "findings": findings,
            "checked_claims": [
                claim_id for claim_id in core.checked_claims if claim_id not in replaced
            ],
            "unverified_claims": [
                claim_id
                for claim_id in core.unverified_claims
                if claim_id not in replaced
            ],
            "layer_status": LayerStatus.ok,
            "failure_reason": None,
        }
    )


def _post(config, transport, user_text: str) -> dict:
    import httpx

    key = _env_value(_KEY_NAMES)
    base = _env_value(_BASE_NAMES) or _DEFAULT_BASE
    model_name = _env_value(_MODEL_NAMES) or _DEFAULT_MODEL
    body = {
        "model": model_name,
        "temperature": 0,
        "max_tokens": 1024,
        "thinking": {"type": "disabled"},
        "response_format": {"type": "json_object"},
        "messages": [
            {"role": "system", "content": _system_text()},
            {"role": "user", "content": user_text},
        ],
    }
    headers = {
        "Authorization": f"Bearer {key}",
        "Content-Type": "application/json",
    }
    url = base.rstrip("/") + "/chat/completions"
    with httpx.Client(transport=transport, timeout=config.model_timeout_seconds) as client:
        for attempt in (1, 2):
            try:
                response = client.post(url, headers=headers, json=body)
            except httpx.TimeoutException as exc:
                _LOGGER.warning("%s", type(exc).__name__)
                raise _ModelCallError(FailureReason.model_timeout) from None
            except httpx.HTTPError as exc:
                _LOGGER.warning("%s", type(exc).__name__)
                raise _ModelCallError(FailureReason.model_network) from None
            status = response.status_code
            if status == 429 or 500 <= status <= 599:
                _LOGGER.warning("%s", status)
                if attempt == 1:
                    continue
                raise _ModelCallError(FailureReason.model_network)
            if status != 200:
                _LOGGER.warning("%s", status)
                raise _ModelCallError(FailureReason.model_network)
            text = response.text
            if not text.strip():
                if attempt == 1:
                    continue
                _LOGGER.warning("%s", FailureReason.model_invalid_json.value)
                raise _ModelCallError(FailureReason.model_invalid_json)
            try:
                data = response.json()
            except json.JSONDecodeError:
                _LOGGER.warning("%s", FailureReason.model_invalid_json.value)
                raise _ModelCallError(FailureReason.model_invalid_json) from None
            return _unwrap(data)
    _LOGGER.warning("%s", FailureReason.model_invalid_json.value)
    raise _ModelCallError(FailureReason.model_invalid_json)


def _unwrap(data: object) -> dict:
    if isinstance(data, dict) and "choices" in data:
        try:
            content = data["choices"][0]["message"]["content"]
        except (KeyError, IndexError, TypeError):
            raise _ModelCallError(FailureReason.model_invalid_json) from None
        if not isinstance(content, str) or not content.strip():
            _LOGGER.warning("%s", FailureReason.model_invalid_json.value)
            raise _ModelCallError(FailureReason.model_invalid_json)
        try:
            data = json.loads(content)
        except json.JSONDecodeError:
            _LOGGER.warning("%s", FailureReason.model_invalid_json.value)
            raise _ModelCallError(FailureReason.model_invalid_json) from None
    if not isinstance(data, dict):
        _LOGGER.warning("%s", FailureReason.model_schema.value)
        raise _ModelCallError(FailureReason.model_schema)
    return data


def _user_coverage(request) -> str:
    payload = {
        "question": request.question,
        "dialogue": [turn.model_dump(mode="json") for turn in request.dialogue],
        "retrieval": request.retrieval.model_dump(mode="json"),
    }
    return json.dumps(payload, ensure_ascii=False)


def _user_fact(request) -> str:
    claims = request.claims or []
    payload = {
        "question": request.question,
        "draft": request.answer,
        "retrieval": request.retrieval.model_dump(mode="json"),
        "claims": [claim.model_dump(mode="json") for claim in claims],
    }
    return json.dumps(payload, ensure_ascii=False)


def _lawyer_text(request) -> str:
    parts = [request.question]
    for turn in request.dialogue:
        if getattr(turn.role, "value", turn.role) == "user":
            parts.append(turn.content)
    return "\n".join(parts)


def _merge_premise_errors(data, request):
    from decision.coverage import PremiseError
    from decision.factcheck import _parts

    parts = _parts(request.retrieval)
    lawyer = _lawyer_text(request)
    merged: list[PremiseError] = []
    for item in data.get("premise_errors") or []:
        if not isinstance(item, dict):
            continue
        expected = item.get("expected", "")
        actual = item.get("actual", "")
        if not isinstance(expected, str) or not isinstance(actual, str):
            continue
        if not _grounded(expected, parts) or actual not in lawyer:
            continue
        merged.append(
            PremiseError(
                statement=str(item.get("statement", "")),
                evidence_quote=str(item.get("evidence_quote", "")),
                expected=expected,
                actual=actual,
            )
        )
    return merged


def _apply_coverage(core, data, request):
    from decision.coverage import ReasonCode as RC

    model_errors = _merge_premise_errors(data, request)
    if not model_errors:
        return core
    existing = {(item.statement, item.expected, item.actual) for item in core.premise_errors}
    premise_errors = list(core.premise_errors)
    for item in model_errors:
        key = (item.statement, item.expected, item.actual)
        if key not in existing:
            existing.add(key)
            premise_errors.append(item)
    codes = list(core.reason_codes)
    if premise_errors and RC.premise_error not in codes:
        codes.append(RC.premise_error)
    return core.model_copy(
        update={
            "premise_errors": premise_errors,
            "reason_codes": codes,
            "layer_status": LayerStatus.ok,
            "failure_reason": None,
        }
    )


def _review(core, config, transport, user_text: str, kind: str, rejected, request=None):
    if not _env_value(_KEY_NAMES):
        return _degrade(core, FailureReason.model_unconfigured)
    try:
        data = _post(config, transport, user_text)
    except _ModelCallError as exc:
        return _degrade(core, exc.reason)
    if not _schema_ok(data, kind) or rejected(data):
        _LOGGER.warning("%s", FailureReason.model_schema.value)
        return _degrade(core, FailureReason.model_schema)
    if kind == "coverage" and request is not None:
        return _apply_coverage(core, data, request)
    if kind == "fact_check" and data["findings"]:
        return _apply_findings(core, data)
    return core


def review_coverage(core, request, config, transport):
    """Keep the core unless the classifier returns a schema that passes the gates."""

    return _review(
        core,
        config,
        transport,
        _user_coverage(request),
        "coverage",
        lambda data: _coverage_rejected(data, core, request, config),
        request=request,
    )


def review_fact_check(core, request, config, transport):
    """Keep the core. A gated finding may fill a claim the core left unverifiable."""

    return _review(
        core,
        config,
        transport,
        _user_fact(request),
        "fact_check",
        lambda data: _fact_rejected(data, core, request),
    )
