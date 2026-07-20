"""Provider-independent LLM interpretation of deterministically collected evidence."""

import json
import math
from dataclasses import dataclass
from typing import Literal, Protocol

from ai_investigation.evidence import CollectedEvidence
from ai_investigation.models import InvestigationResult

DiagnosisId = Literal[
    "health_check_timeout",
    "missing_environment_variable",
    "database_migration_failure",
    "missing_database_configuration",
    "database_contention_blocked_migration",
]
AbstentionReason = Literal[
    "insufficient_evidence",
    "conflicting_evidence",
    "low_confidence",
]
PromptVersion = Literal["v1", "v2", "v3"]

DIAGNOSIS_IDS = (
    "health_check_timeout",
    "missing_environment_variable",
    "database_migration_failure",
    "missing_database_configuration",
    "database_contention_blocked_migration",
)
PROMPT_VERSION_V1 = "llm-investigator-v1"
PROMPT_VERSION_V2 = "llm-investigator-v2"
PROMPT_VERSION_V3 = "llm-investigator-v3"
PROMPT_VERSION = PROMPT_VERSION_V1
DEFAULT_PROMPT_VERSION: PromptVersion = "v1"
RESPONSE_SCHEMA_VERSION = "llm-decision-v1"
ABSTENTION_REASONS = (
    "insufficient_evidence",
    "conflicting_evidence",
    "low_confidence",
)
RESPONSE_FIELDS = {
    "outcome",
    "diagnosis_id",
    "confidence",
    "evidence_references",
    "abstention_reason",
}
REQUIRED_EVIDENCE_SOURCES = {
    "health_check_timeout": {"deployment", "logs", "service_health"},
    "missing_environment_variable": {"deployment", "logs"},
    "database_migration_failure": {"deployment", "logs"},
    "missing_database_configuration": {"deployment", "logs"},
    "database_contention_blocked_migration": {"deployment", "logs"},
}
LLM_RESPONSE_JSON_SCHEMA = {
    "type": "object",
    "additionalProperties": False,
    "required": sorted(RESPONSE_FIELDS),
    "properties": {
        "outcome": {"type": "string", "enum": ["diagnosis", "abstain"]},
        "diagnosis_id": {
            "anyOf": [
                {"type": "string", "enum": list(DIAGNOSIS_IDS)},
                {"type": "null"},
            ]
        },
        "confidence": {"type": "number", "minimum": 0.0, "maximum": 1.0},
        "evidence_references": {
            "type": "array",
            "items": {"type": "integer", "minimum": 1},
            "uniqueItems": True,
        },
        "abstention_reason": {
            "anyOf": [
                {"type": "string", "enum": list(ABSTENTION_REASONS)},
                {"type": "null"},
            ]
        },
    },
}


class StructuredModel(Protocol):
    @property
    def provider_name(self) -> str:
        """Stable provider identifier for experiment metadata."""

    @property
    def model_name(self) -> str:
        """Stable model identifier for experiment metadata."""

    def generate(self, prompt: str) -> str:
        """Return one raw structured response for the supplied prompt."""


class ModelRefusalError(Exception):
    """Raised when a model explicitly refuses or filters the request."""


class ModelProviderError(Exception):
    """Raised when a provider cannot complete the model request."""


@dataclass(frozen=True, slots=True)
class LLMDecision:
    """A strictly parsed model diagnosis or explicit abstention."""

    outcome: Literal["diagnosis", "abstain"]
    diagnosis_id: DiagnosisId | None
    confidence: float
    evidence_references: tuple[int, ...]
    abstention_reason: AbstentionReason | None


@dataclass(frozen=True, slots=True)
class LLMInvestigationSuccess:
    """A valid model decision converted into a public investigation result."""

    status: Literal["ok"]
    decision: LLMDecision
    result: InvestigationResult


@dataclass(frozen=True, slots=True)
class LLMInvestigationFailure:
    """A model execution that did not produce a valid investigation."""

    status: Literal[
        "not_evaluated",
        "invalid_response",
        "invalid_references",
        "refused",
        "provider_failure",
    ]
    errors: tuple[str, ...]


LLMInvestigationOutcome = LLMInvestigationSuccess | LLMInvestigationFailure


def build_prompt(
    collected: CollectedEvidence,
    prompt_version: PromptVersion = DEFAULT_PROMPT_VERSION,
) -> str:
    """Serialize the complete collected input and ordered evidence references."""

    if prompt_version == "v1":
        return _build_prompt_v1(collected)
    if prompt_version == "v2":
        return _build_prompt_v2(collected)
    if prompt_version == "v3":
        return _build_prompt_v3(collected)
    raise ValueError(f"Unsupported prompt version: {prompt_version}.")


def prompt_version_identifier(prompt_version: PromptVersion) -> str:
    """Return the stable experiment identifier for a validated prompt version."""

    if prompt_version == "v1":
        return PROMPT_VERSION_V1
    if prompt_version == "v2":
        return PROMPT_VERSION_V2
    if prompt_version == "v3":
        return PROMPT_VERSION_V3
    raise ValueError(f"Unsupported prompt version: {prompt_version}.")


def _build_prompt_v1(collected: CollectedEvidence) -> str:
    """Preserve the original Milestone 10 prompt and evidence representation."""

    payload = serialize_evidence(collected)
    return (
        f"Prompt version: {PROMPT_VERSION_V1}. Interpret only the supplied deployment evidence. "
        "Do not invent or request additional evidence. Never use external assumptions. "
        "Choose one supported diagnosis only when the supplied evidence supports it; otherwise "
        "abstain when evidence is insufficient, contradictory, or unsupported. Cite exact numbered "
        "evidence references. Confidence represents uncertainty and must be between 0.0 and 1.0. "
        "Return only the requested JSON object; do not provide hidden reasoning.\n\n"
        "Supported diagnosis IDs: health_check_timeout, missing_environment_variable, "
        "database_migration_failure, missing_database_configuration, "
        "database_contention_blocked_migration.\n"
        "Abstention reasons: insufficient_evidence, conflicting_evidence, low_confidence.\n"
        f"Response schema version: {RESPONSE_SCHEMA_VERSION}.\n"
        "Response JSON schema:\n"
        f"{json.dumps(LLM_RESPONSE_JSON_SCHEMA, sort_keys=True, separators=(',', ':'))}\n\n"
        f"Collected evidence:\n{json.dumps(payload, sort_keys=True, separators=(',', ':'))}"
    )


def _build_prompt_v2(collected: CollectedEvidence) -> str:
    payload = serialize_evidence_v2(collected)
    valid_ids = [item["id"] for item in payload["evidence"]]
    return (
        f"Prompt version: {PROMPT_VERSION_V2}. Derive conclusions only from the supplied evidence. "
        f"The only valid evidence IDs are {valid_ids}; use these exact integer IDs in "
        "evidence_references and never invent an evidence ID. Include only evidence IDs that "
        "directly support the diagnosis. Abstain when evidence is incomplete, conflicting, "
        "unsupported, or leaves multiple diagnoses equally plausible. Abstain for generic errors "
        "that do not establish a supported root cause. Prefer abstention over guessing. "
        "Distinguish missing_environment_variable from missing_database_configuration. A "
        "database-related error is insufficient for missing_database_configuration unless the "
        "evidence specifically supports missing or invalid database configuration. Confidence "
        "represents uncertainty and must be between 0.0 and 1.0. Return only the requested JSON "
        "object; do not provide hidden reasoning.\n\n"
        "Supported diagnosis IDs: health_check_timeout, missing_environment_variable, "
        "database_migration_failure, missing_database_configuration, "
        "database_contention_blocked_migration.\n"
        "Abstention reasons: insufficient_evidence, conflicting_evidence, low_confidence.\n"
        f"Response schema version: {RESPONSE_SCHEMA_VERSION}.\n"
        "Response JSON schema:\n"
        f"{json.dumps(LLM_RESPONSE_JSON_SCHEMA, sort_keys=True, separators=(',', ':'))}\n\n"
        f"Collected evidence:\n{json.dumps(payload, sort_keys=True, separators=(',', ':'))}"
    )


def _build_prompt_v3(collected: CollectedEvidence) -> str:
    payload = serialize_evidence_v2(collected)
    valid_ids = [item["id"] for item in payload["evidence"]]
    return (
        f"Prompt version: {PROMPT_VERSION_V3}. Derive conclusions only from the supplied evidence. "
        f"The only valid evidence IDs are {valid_ids}; return only these exact integer IDs and "
        "never invent missing evidence. Evidence references are part of the validity contract, not "
        "optional explanatory citations. Every diagnosis must reference the deployment evidence "
        "that identifies and contextualizes the investigated deployment, plus all log evidence "
        "necessary to establish the claimed cause. Reference service-health evidence whenever a "
        "diagnosis depends on health-check state, timeout behavior, readiness, liveness, or service "
        "availability. Do not return a diagnosis when a required evidence source is missing. "
        "Abstentions should reference relevant conflicting or insufficient evidence when "
        "appropriate, but must not invent missing evidence.\n\n"
        "Diagnosis boundaries: A generic database error does not establish "
        "database_migration_failure. Diagnose database_migration_failure only when evidence "
        "explicitly connects the failure to migration execution, migration application, schema "
        "migration, or an equivalent migration operation. A timeout symptom does not establish "
        "health_check_timeout. Diagnose health_check_timeout only when evidence directly connects "
        "the timeout to a deployment health check, readiness check, liveness check, or equivalent "
        "health-check mechanism. Never choose the closest supported diagnosis merely because "
        "symptoms overlap; similarity is not sufficient causal evidence. Abstain when evidence is "
        "incomplete, conflicting, unsupported, or supports multiple incompatible causes. If the "
        "available cause is outside the supported diagnosis set, abstain instead of mapping it to "
        "the nearest label. Prefer abstention over guessing. Distinguish "
        "missing_environment_variable from missing_database_configuration; a database-related "
        "error is insufficient for missing_database_configuration unless evidence specifically "
        "supports missing or invalid database configuration. Confidence represents uncertainty "
        "and must be between 0.0 and 1.0. Return only the requested JSON object; do not provide "
        "hidden reasoning.\n\n"
        "Supported diagnosis IDs: health_check_timeout, missing_environment_variable, "
        "database_migration_failure, missing_database_configuration, "
        "database_contention_blocked_migration.\n"
        "Abstention reasons: insufficient_evidence, conflicting_evidence, low_confidence.\n"
        f"Response schema version: {RESPONSE_SCHEMA_VERSION}.\n"
        "Response JSON schema:\n"
        f"{json.dumps(LLM_RESPONSE_JSON_SCHEMA, sort_keys=True, separators=(',', ':'))}\n\n"
        f"Collected evidence:\n{json.dumps(payload, sort_keys=True, separators=(',', ':'))}"
    )


def serialize_evidence(collected: CollectedEvidence) -> dict[str, object]:
    """Return stable JSON-compatible evidence without evaluation metadata."""

    payload = {
        "request": {
            "question": collected.request.question,
            "deployment_id": collected.request.deployment_id,
        },
        "deployment": dict(collected.deployment) if collected.deployment is not None else None,
        "error_logs": [dict(log) for log in collected.error_logs],
        "service_health": (
            dict(collected.service_health) if collected.service_health is not None else None
        ),
        "evidence": [
            {"reference": index, "source": item.source, "summary": item.summary}
            for index, item in enumerate(collected.evidence, start=1)
        ],
        "collection_limitations": list(collected.limitations),
    }
    return payload


def serialize_evidence_v2(collected: CollectedEvidence) -> dict[str, object]:
    """Present self-contained, numbered evidence to the LLM without changing domain evidence."""

    items: list[dict[str, object]] = []
    if collected.deployment is not None:
        items.append(
            _llm_evidence_item(
                len(items) + 1,
                "deployment",
                "deployment",
                collected.evidence[len(items)].summary,
                collected.deployment,
            )
        )
    for log in collected.error_logs:
        items.append(
            _llm_evidence_item(
                len(items) + 1,
                "error_log",
                "logs",
                collected.evidence[len(items)].summary,
                log,
            )
        )
    if collected.service_health is not None:
        items.append(
            _llm_evidence_item(
                len(items) + 1,
                "service_health",
                "service_health",
                collected.evidence[len(items)].summary,
                collected.service_health,
            )
        )
    return {
        "request": {
            "question": collected.request.question,
            "deployment_id": collected.request.deployment_id,
        },
        "evidence": items,
        "collection_limitations": list(collected.limitations),
    }


def _llm_evidence_item(
    evidence_id: int,
    evidence_type: str,
    source: str,
    observation: str,
    content: object,
) -> dict[str, object]:
    return {
        "id": evidence_id,
        "type": evidence_type,
        "source": source,
        "observation": observation,
        "content": dict(content),
    }


def parse_decision(raw_response: str) -> LLMDecision:
    """Parse the exact structured response contract without provider assumptions."""

    try:
        value = json.loads(raw_response)
    except (json.JSONDecodeError, TypeError) as error:
        raise ValueError("Model response is not valid JSON.") from error
    if not isinstance(value, dict):
        raise ValueError("Model response must be a JSON object.")

    fields = set(value)
    if fields != RESPONSE_FIELDS:
        missing = sorted(RESPONSE_FIELDS - fields)
        unexpected = sorted(fields - RESPONSE_FIELDS)
        details = []
        if missing:
            details.append(f"missing fields: {', '.join(missing)}")
        if unexpected:
            details.append(f"unexpected fields: {', '.join(unexpected)}")
        raise ValueError("Invalid response fields (" + "; ".join(details) + ").")

    outcome = value["outcome"]
    diagnosis_id = value["diagnosis_id"]
    confidence = value["confidence"]
    references = value["evidence_references"]
    abstention_reason = value["abstention_reason"]

    if outcome not in ("diagnosis", "abstain"):
        raise ValueError("outcome must be 'diagnosis' or 'abstain'.")
    if isinstance(confidence, bool) or not isinstance(confidence, (int, float)):
        raise ValueError("confidence must be a non-boolean number.")
    confidence_value = float(confidence)
    if not math.isfinite(confidence_value) or not 0.0 <= confidence_value <= 1.0:
        raise ValueError("confidence must be between 0.0 and 1.0.")
    if not isinstance(references, list) or not all(
        isinstance(reference, int) and not isinstance(reference, bool)
        for reference in references
    ):
        raise ValueError("evidence_references must be a list of integers.")

    if outcome == "diagnosis":
        if diagnosis_id not in DIAGNOSIS_IDS:
            raise ValueError("diagnosis_id must identify a supported diagnosis.")
        if abstention_reason is not None:
            raise ValueError("abstention_reason must be null for a diagnosis.")
    else:
        if diagnosis_id is not None:
            raise ValueError("diagnosis_id must be null for an abstention.")
        if abstention_reason not in ABSTENTION_REASONS:
            raise ValueError("abstention_reason must identify a supported reason.")

    return LLMDecision(
        outcome=outcome,
        diagnosis_id=diagnosis_id,
        confidence=confidence_value,
        evidence_references=tuple(references),
        abstention_reason=abstention_reason,
    )


def validate_evidence_references(
    decision: LLMDecision,
    collected: CollectedEvidence,
) -> tuple[str, ...]:
    """Validate reference integrity and required source coverage."""

    errors: list[str] = []
    references = decision.evidence_references
    if len(references) != len(set(references)):
        errors.append("Evidence references must be unique.")

    invalid = tuple(
        reference for reference in references if not 1 <= reference <= len(collected.evidence)
    )
    if invalid:
        errors.append(f"Evidence references do not exist: {invalid!r}.")

    if decision.outcome == "diagnosis" and not references:
        errors.append("A diagnosis must reference evidence.")

    if not invalid and decision.diagnosis_id is not None:
        sources = {collected.evidence[reference - 1].source for reference in references}
        missing_sources = REQUIRED_EVIDENCE_SOURCES[decision.diagnosis_id] - sources
        if missing_sources:
            errors.append(
                "Diagnosis references are missing required sources: "
                + ", ".join(sorted(missing_sources))
                + "."
            )
    return tuple(errors)


class LLMInvestigator:
    """Interpret pre-collected evidence through one structured model call."""

    def __init__(
        self,
        model: StructuredModel,
        prompt_version: PromptVersion = DEFAULT_PROMPT_VERSION,
    ) -> None:
        self._model = model
        prompt_version_identifier(prompt_version)
        self._prompt_version = prompt_version

    def investigate(self, collected: CollectedEvidence) -> LLMInvestigationOutcome:
        if collected.request.deployment_id is None:
            return LLMInvestigationFailure(
                "not_evaluated",
                ("No deployment ID was provided.",),
            )
        if collected.deployment is None:
            return LLMInvestigationFailure(
                "not_evaluated",
                (f"Deployment {collected.request.deployment_id} was not found.",),
            )

        try:
            raw_response = self._model.generate(
                build_prompt(collected, self._prompt_version)
            )
        except ModelRefusalError as error:
            return LLMInvestigationFailure("refused", (str(error) or "Model refused.",))
        except ModelProviderError as error:
            return LLMInvestigationFailure(
                "provider_failure",
                (str(error) or "Model provider failed.",),
            )

        try:
            decision = parse_decision(raw_response)
        except ValueError as error:
            return LLMInvestigationFailure("invalid_response", (str(error),))

        reference_errors = validate_evidence_references(decision, collected)
        if reference_errors:
            return LLMInvestigationFailure("invalid_references", reference_errors)

        return LLMInvestigationSuccess(
            status="ok",
            decision=decision,
            result=_result_from_decision(decision, collected),
        )


def _result_from_decision(
    decision: LLMDecision,
    collected: CollectedEvidence,
) -> InvestigationResult:
    referenced = set(decision.evidence_references)
    evidence = tuple(
        item
        for index, item in enumerate(collected.evidence, start=1)
        if index in referenced
    )
    deployment_id = collected.request.deployment_id

    if decision.diagnosis_id is not None:
        root_cause = _root_cause(decision.diagnosis_id)
        answer = (
            f"Deployment {deployment_id} failed during its health check. {root_cause}"
            if decision.diagnosis_id == "health_check_timeout"
            else f"Deployment {deployment_id} failed. {root_cause}"
        )
        return InvestigationResult(
            answer=answer,
            root_cause=root_cause,
            evidence=evidence,
            confidence=decision.confidence,
            limitations=(),
        )

    limitations = {
        "insufficient_evidence": "The model found insufficient evidence for a supported diagnosis.",
        "conflicting_evidence": "The model found conflicting evidence for supported diagnoses.",
        "low_confidence": "The model abstained because its confidence was low.",
    }
    return InvestigationResult(
        answer=f"The cause of deployment {deployment_id}'s failure is inconclusive.",
        root_cause=None,
        evidence=evidence,
        confidence=decision.confidence,
        limitations=(limitations[decision.abstention_reason],),
    )


def _root_cause(diagnosis_id: DiagnosisId) -> str:
    return {
        "health_check_timeout": (
            "The deployment health check timed out because the target service was unhealthy."
        ),
        "missing_environment_variable": (
            "The deployment failed because a required environment variable was missing."
        ),
        "database_migration_failure": (
            "The deployment failed because a database migration could not be applied."
        ),
        "missing_database_configuration": (
            "The deployment failed because required database connection configuration was missing."
        ),
        "database_contention_blocked_migration": (
            "The deployment migration was blocked by database contention."
        ),
    }[diagnosis_id]
