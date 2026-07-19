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
]
AbstentionReason = Literal[
    "insufficient_evidence",
    "conflicting_evidence",
    "low_confidence",
]

DIAGNOSIS_IDS = (
    "health_check_timeout",
    "missing_environment_variable",
    "database_migration_failure",
)
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


def build_prompt(collected: CollectedEvidence) -> str:
    """Serialize the complete collected input and ordered evidence references."""

    payload = {
        "question": collected.request.question,
        "deployment_id": collected.request.deployment_id,
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
    return (
        "Interpret only the supplied deployment evidence. Do not invent or request additional "
        "evidence. Choose one supported diagnosis or explicitly abstain. Cite only the numbered "
        "evidence references. Confidence must be between 0.0 and 1.0.\n\n"
        "Supported diagnosis IDs: health_check_timeout, missing_environment_variable, "
        "database_migration_failure.\n"
        "Abstention reasons: insufficient_evidence, conflicting_evidence, low_confidence.\n\n"
        f"Collected evidence:\n{json.dumps(payload, sort_keys=True, separators=(',', ':'))}"
    )


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

    def __init__(self, model: StructuredModel) -> None:
        self._model = model

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
            raw_response = self._model.generate(build_prompt(collected))
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
    }[diagnosis_id]
