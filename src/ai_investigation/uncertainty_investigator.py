"""Single-pass LLM uncertainty proposal controlled by deterministic decision policy."""

import json
import math
from copy import deepcopy
from dataclasses import dataclass
from enum import Enum
from typing import Literal

from ai_investigation.decision_policy import (
    CandidateAssessment,
    CandidateDiagnosis,
    DecisionOutcome,
    DecisionReason,
    EvidenceStrength,
    InvestigationDecision,
    SUPPORTED_DIAGNOSES,
    UncertaintyAssessment,
    decide_investigation,
)
from ai_investigation.diagnosis_catalogue import (
    DIAGNOSIS_CATALOGUE_VERSION,
    render_diagnosis_catalogue,
)
from ai_investigation.evidence import CollectedEvidence
from ai_investigation.llm_investigator import (
    ModelProviderError,
    ModelRefusalError,
    REQUIRED_EVIDENCE_SOURCES,
    StructuredModel,
    serialize_evidence_v2,
)
from ai_investigation.models import InvestigationResult

UNCERTAINTY_PROMPT_VERSION = "llm-investigator-v4-uncertainty-contract-v2"
UNCERTAINTY_RESPONSE_SCHEMA_VERSION = "llm-uncertainty-proposal-v2"
CANDIDATE_SEMANTICS_PROMPT_VERSION = (
    "llm-investigator-v4-uncertainty-candidate-semantics-contract-v3-catalogue-v1"
)
CANDIDATE_SEMANTICS_RESPONSE_SCHEMA_VERSION = "llm-uncertainty-proposal-v4"
HISTORICAL_CANDIDATE_SEMANTICS_PROMPT_VERSION = (
    "llm-investigator-v4-uncertainty-candidate-semantics"
)
HISTORICAL_CANDIDATE_SEMANTICS_RESPONSE_SCHEMA_VERSION = (
    "llm-uncertainty-proposal-v3"
)
DECISION_POLICY_VERSION = "deterministic-decision-policy-v1"
UNCERTAINTY_PROMPT_SELECTION = "v4-uncertainty"
CANDIDATE_SEMANTICS_PROMPT_SELECTION = "v4-uncertainty-candidate-semantics"
_PROPOSAL_FIELDS = {
    "candidates",
    "unsupported_signals_present",
    "unsupported_signals_material",
    "conflicting_supported_candidates",
    "insufficient_evidence",
    "reasoning_summary",
}
_CANDIDATE_FIELDS = {
    "diagnosis_id",
    "supporting_evidence_references",
    "contradicting_evidence_references",
    "evidence_strength",
    "confidence",
}
_CANDIDATE_SEMANTICS_FIELDS = {
    "supported_candidates",
    "rejected_hypotheses",
    "unsupported_signals_present",
    "unsupported_signals_material",
    "insufficient_evidence",
    "reasoning_summary",
}
_REJECTED_HYPOTHESIS_FIELDS = {
    "diagnosis_id",
    "reason_code",
    "reason_summary",
    "relevant_evidence_references",
    "confidence",
}

UNCERTAINTY_RESPONSE_JSON_SCHEMA = {
    "type": "object",
    "additionalProperties": False,
    "required": sorted(_PROPOSAL_FIELDS),
    "properties": {
        "candidates": {
            "type": "array",
            "items": {
                "type": "object",
                "additionalProperties": False,
                "required": sorted(_CANDIDATE_FIELDS),
                "properties": {
                    "diagnosis_id": {
                        "type": "string",
                        "enum": sorted(SUPPORTED_DIAGNOSES),
                    },
                    "supporting_evidence_references": {
                        "type": "array",
                        "items": {"type": "string", "pattern": "^E[1-9][0-9]*$"},
                        "uniqueItems": True,
                    },
                    "contradicting_evidence_references": {
                        "type": "array",
                        "items": {"type": "string", "pattern": "^E[1-9][0-9]*$"},
                        "uniqueItems": True,
                    },
                    "evidence_strength": {
                        "type": "string",
                        "enum": [item.value for item in EvidenceStrength],
                    },
                    "confidence": {
                        "type": "number",
                        "minimum": 0.0,
                        "maximum": 1.0,
                    },
                },
            },
        },
        "unsupported_signals_present": {"type": "boolean"},
        "unsupported_signals_material": {"type": "boolean"},
        "conflicting_supported_candidates": {"type": "boolean"},
        "insufficient_evidence": {"type": "boolean"},
        "reasoning_summary": {"type": "string", "minLength": 1},
    },
}


class RejectionReason(str, Enum):
    NO_DIRECT_SUPPORT = "no_direct_support"
    ONLY_SHARED_VOCABULARY = "only_shared_vocabulary"
    SYMPTOM_NOT_ROOT_CAUSE = "symptom_not_root_cause"
    CONTRADICTED_BY_EVIDENCE = "contradicted_by_evidence"
    MISSING_REQUIRED_EVIDENCE = "missing_required_evidence"
    WEAKER_THAN_SUPPORTED_CAUSE = "weaker_than_supported_cause"
    UNSUPPORTED_BY_AVAILABLE_DIAGNOSIS_SET = (
        "unsupported_by_available_diagnosis_set"
    )


_SUPPORTED_CANDIDATES_SCHEMA = deepcopy(
    UNCERTAINTY_RESPONSE_JSON_SCHEMA["properties"]["candidates"]
)
_SUPPORTED_CANDIDATES_SCHEMA["items"]["properties"][
    "supporting_evidence_references"
]["minItems"] = 1

CANDIDATE_SEMANTICS_RESPONSE_JSON_SCHEMA = {
    "type": "object",
    "additionalProperties": False,
    "required": sorted(_CANDIDATE_SEMANTICS_FIELDS),
    "properties": {
        "supported_candidates": _SUPPORTED_CANDIDATES_SCHEMA,
        "rejected_hypotheses": {
            "type": "array",
            "items": {
                "type": "object",
                "additionalProperties": False,
                "required": sorted(_REJECTED_HYPOTHESIS_FIELDS),
                "properties": {
                    "diagnosis_id": {
                        "type": "string",
                        "enum": sorted(SUPPORTED_DIAGNOSES),
                    },
                    "reason_code": {
                        "type": "string",
                        "enum": [reason.value for reason in RejectionReason],
                    },
                    "reason_summary": {"type": "string", "minLength": 1},
                    "confidence": {
                        "type": "number",
                        "minimum": 0.0,
                        "maximum": 1.0,
                    },
                    "relevant_evidence_references": {
                        "type": "array",
                        "items": {
                            "type": "string",
                            "pattern": "^E[1-9][0-9]*$",
                        },
                        "uniqueItems": True,
                    },
                },
            },
        },
        "unsupported_signals_present": {"type": "boolean"},
        "unsupported_signals_material": {"type": "boolean"},
        "insufficient_evidence": {"type": "boolean"},
        "reasoning_summary": {"type": "string", "minLength": 1},
    },
}


@dataclass(frozen=True, slots=True)
class ProposedCandidate:
    diagnosis_id: str
    supporting_evidence_references: tuple[str, ...]
    contradicting_evidence_references: tuple[str, ...]
    evidence_strength: EvidenceStrength
    confidence: float


@dataclass(frozen=True, slots=True)
class RejectedHypothesis:
    diagnosis_id: str
    reason_code: RejectionReason
    reason_summary: str
    relevant_evidence_references: tuple[str, ...]
    reported_confidence: float


@dataclass(frozen=True, slots=True)
class LLMUncertaintyProposal:
    candidates: tuple[ProposedCandidate, ...]
    unsupported_signals_present: bool
    unsupported_signals_material: bool
    conflicting_supported_candidates: bool
    insufficient_evidence: bool
    reasoning_summary: str
    rejected_hypotheses: tuple[RejectedHypothesis, ...] = ()


@dataclass(frozen=True, slots=True)
class LLMPolicyInvestigationSuccess:
    status: Literal["ok"]
    proposal: LLMUncertaintyProposal
    uncertainty: UncertaintyAssessment
    decision: InvestigationDecision
    result: InvestigationResult
    raw_response: str


@dataclass(frozen=True, slots=True)
class LLMPolicyInvestigationFailure:
    status: Literal[
        "not_evaluated",
        "invalid_response",
        "invalid_references",
        "adapter_failure",
        "refused",
        "provider_failure",
    ]
    errors: tuple[str, ...]
    raw_response: str | None = None


LLMPolicyInvestigationOutcome = (
    LLMPolicyInvestigationSuccess | LLMPolicyInvestigationFailure
)


class UncertaintyAdapterError(ValueError):
    """Raised when a valid proposal is incompatible with collected evidence."""

    def __init__(self, message: str, *, invalid_references: bool = False) -> None:
        super().__init__(message)
        self.invalid_references = invalid_references


def build_uncertainty_prompt(
    collected: CollectedEvidence,
    prompt_selection: str = UNCERTAINTY_PROMPT_SELECTION,
) -> str:
    if prompt_selection == CANDIDATE_SEMANTICS_PROMPT_SELECTION:
        return _build_candidate_semantics_prompt(collected)
    if prompt_selection != UNCERTAINTY_PROMPT_SELECTION:
        raise ValueError(f"Unsupported uncertainty prompt: {prompt_selection}.")
    payload = serialize_uncertainty_evidence(collected)
    valid_ids = [item["id"] for item in payload["evidence"]]
    return (
        f"Prompt version: {UNCERTAINTY_PROMPT_VERSION}. Propose uncertainty facts only; do not "
        "choose a final diagnosis, abstention, or review outcome. Return zero, one, or multiple "
        "supported diagnosis candidates without hiding competing candidates or collapsing ambiguity "
        "into the most likely label. Confidence is reported belief, not a substitute for evidence "
        "strength or completeness. Evidence identity and source availability are validated by the "
        "application. Do not infer or return source-availability or missing-source metadata. Use "
        "only exact evidence IDs from the supplied list "
        f"{valid_ids}; no line number, list position, timestamp, bare integer, or other value is "
        "permitted. Distinguish material contradictions from irrelevant distractors. Represent "
        "unsupported causes as unsupported signals rather than the nearest supported diagnosis. "
        "Use insufficient_evidence for semantic insufficiency without attempting to calculate "
        "source coverage. Return only the requested JSON object and a short reasoning_summary; do not "
        "provide hidden reasoning.\n\n"
        "Supported diagnosis IDs: "
        + ", ".join(sorted(SUPPORTED_DIAGNOSES))
        + ".\n"
        f"Response schema version: {UNCERTAINTY_RESPONSE_SCHEMA_VERSION}.\n"
        "Response JSON schema:\n"
        f"{json.dumps(UNCERTAINTY_RESPONSE_JSON_SCHEMA, sort_keys=True, separators=(',', ':'))}\n\n"
        f"Collected evidence:\n{json.dumps(payload, sort_keys=True, separators=(',', ':'))}"
    )


def _build_candidate_semantics_prompt(collected: CollectedEvidence) -> str:
    payload = serialize_uncertainty_evidence(collected)
    valid_ids = [item["id"] for item in payload["evidence"]]
    return (
        f"Prompt version: {CANDIDATE_SEMANTICS_PROMPT_VERSION}. Propose semantic uncertainty "
        "facts only; do not choose a final diagnosis, abstention, or review outcome. Do not return "
        "every diagnosis you considered as a supported candidate. A supported candidate has direct "
        "positive, diagnosis-specific evidence and could independently explain the observed "
        "failure. Shared vocabulary, thematic relevance, and plausibility are insufficient without "
        "direct diagnosis-specific support. Do not map generic evidence to the nearest available "
        "diagnosis. A rejected hypothesis was considered but lacks direct support, is contradicted, "
        "is only a symptom, or is weaker than a supported cause. Absence of contradiction is not "
        "support. Semantic similarity is not support. Shared database vocabulary is not enough to "
        "support multiple database diagnoses. Do not reuse the same evidence as direct support for "
        "unrelated diagnoses without explicit justification. A timeout is not automatically a health-check "
        "failure, and an application crash is not automatically a migration failure. Do not make a "
        "downstream health symptom a second root-cause candidate when another diagnosis directly "
        "explains it; prefer the supported upstream root cause and reject the downstream symptom as "
        "not independently causal. Do not add a second supported candidate merely to express "
        "uncertainty. Use "
        "zero supported candidates when no diagnosis is directly supported, one when one diagnosis "
        "clearly explains the evidence, and multiple only when each could independently explain the "
        "failure and genuine unresolved causal ambiguity remains after considering all evidence. "
        "Place considered but unsupported alternatives in rejected_hypotheses. A diagnosis ID "
        "must appear in at most one collection: it may appear in supported_candidates or "
        "rejected_hypotheses, but never both. Confidence is "
        "metadata, never a "
        "substitute for evidence. Evidence identity and source availability are validated by the "
        "application; do not infer or return missing-source metadata. Use only exact evidence IDs "
        f"from {valid_ids}; no other reference value is permitted. Return only the requested JSON "
        "object with a concise reasoning_summary; do not provide hidden reasoning.\n\n"
        "Supported diagnosis IDs: "
        + ", ".join(sorted(SUPPORTED_DIAGNOSES))
        + ".\n"
        f"Diagnosis catalogue version: {DIAGNOSIS_CATALOGUE_VERSION}.\n"
        "Operational diagnosis catalogue:\n"
        f"{render_diagnosis_catalogue()}\n"
        f"Response schema version: {CANDIDATE_SEMANTICS_RESPONSE_SCHEMA_VERSION}.\n"
        "Response JSON schema:\n"
        f"{json.dumps(CANDIDATE_SEMANTICS_RESPONSE_JSON_SCHEMA, sort_keys=True, separators=(',', ':'))}\n\n"
        f"Collected evidence:\n{json.dumps(payload, sort_keys=True, separators=(',', ':'))}"
    )


def uncertainty_prompt_identifier(prompt_selection: str) -> str:
    if prompt_selection == UNCERTAINTY_PROMPT_SELECTION:
        return UNCERTAINTY_PROMPT_VERSION
    if prompt_selection == CANDIDATE_SEMANTICS_PROMPT_SELECTION:
        return CANDIDATE_SEMANTICS_PROMPT_VERSION
    raise ValueError(f"Unsupported uncertainty prompt: {prompt_selection}.")


def uncertainty_schema_identifier(prompt_selection: str) -> str:
    if prompt_selection == UNCERTAINTY_PROMPT_SELECTION:
        return UNCERTAINTY_RESPONSE_SCHEMA_VERSION
    if prompt_selection == CANDIDATE_SEMANTICS_PROMPT_SELECTION:
        return CANDIDATE_SEMANTICS_RESPONSE_SCHEMA_VERSION
    raise ValueError(f"Unsupported uncertainty prompt: {prompt_selection}.")


def parse_uncertainty_proposal(raw_response: str) -> LLMUncertaintyProposal:
    try:
        value = json.loads(raw_response)
    except (json.JSONDecodeError, TypeError) as error:
        raise ValueError("Model uncertainty response is not valid JSON.") from error
    if not isinstance(value, dict):
        raise ValueError("Model uncertainty response must be a JSON object.")
    _require_exact_fields(value, _PROPOSAL_FIELDS, "uncertainty response")

    candidates_value = value["candidates"]
    if not isinstance(candidates_value, list):
        raise ValueError("candidates must be a list.")
    candidates = tuple(
        _parse_candidate(item, index) for index, item in enumerate(candidates_value)
    )
    diagnosis_ids = tuple(candidate.diagnosis_id for candidate in candidates)
    if len(diagnosis_ids) != len(set(diagnosis_ids)):
        raise ValueError("Candidate diagnosis IDs must be unique.")

    flags = {}
    for field in (
        "unsupported_signals_present",
        "unsupported_signals_material",
        "conflicting_supported_candidates",
        "insufficient_evidence",
    ):
        if not isinstance(value[field], bool):
            raise ValueError(f"{field} must be boolean.")
        flags[field] = value[field]
    if flags["unsupported_signals_material"] and not flags[
        "unsupported_signals_present"
    ]:
        raise ValueError("Material unsupported signals must be marked as present.")
    if flags["conflicting_supported_candidates"] and len(candidates) < 2:
        raise ValueError("Conflicting supported candidates require at least two candidates.")

    summary = value["reasoning_summary"]
    if not isinstance(summary, str) or not summary.strip():
        raise ValueError("reasoning_summary must be a non-empty string.")

    return LLMUncertaintyProposal(
        candidates=candidates,
        unsupported_signals_present=flags["unsupported_signals_present"],
        unsupported_signals_material=flags["unsupported_signals_material"],
        conflicting_supported_candidates=flags["conflicting_supported_candidates"],
        insufficient_evidence=flags["insufficient_evidence"],
        reasoning_summary=summary.strip(),
    )


def parse_candidate_semantics_proposal(raw_response: str) -> LLMUncertaintyProposal:
    try:
        value = json.loads(raw_response)
    except (json.JSONDecodeError, TypeError) as error:
        raise ValueError("Model uncertainty response is not valid JSON.") from error
    if not isinstance(value, dict):
        raise ValueError("Model uncertainty response must be a JSON object.")
    actual_fields = set(value)
    legacy_fields = _CANDIDATE_SEMANTICS_FIELDS | {
        "conflicting_supported_candidates"
    }
    if actual_fields not in (_CANDIDATE_SEMANTICS_FIELDS, legacy_fields):
        raise ValueError("Invalid uncertainty response fields.")

    supported_value = value["supported_candidates"]
    rejected_value = value["rejected_hypotheses"]
    if not isinstance(supported_value, list):
        raise ValueError("supported_candidates must be a list.")
    if not isinstance(rejected_value, list):
        raise ValueError("rejected_hypotheses must be a list.")
    candidates = tuple(
        _parse_candidate(item, index) for index, item in enumerate(supported_value)
    )
    if any(not candidate.supporting_evidence_references for candidate in candidates):
        raise ValueError("Every supported candidate requires direct supporting evidence.")
    rejected = tuple(
        _parse_rejected_hypothesis(item, index)
        for index, item in enumerate(rejected_value)
    )
    candidate_ids = tuple(candidate.diagnosis_id for candidate in candidates)
    rejected_ids = tuple(item.diagnosis_id for item in rejected)
    if len(candidate_ids) != len(set(candidate_ids)):
        raise ValueError("Supported candidate diagnosis IDs must be unique.")
    if len(rejected_ids) != len(set(rejected_ids)):
        raise ValueError("Rejected hypothesis diagnosis IDs must be unique.")
    overlapping_ids = sorted(set(candidate_ids) & set(rejected_ids))
    if overlapping_ids:
        raise ValueError(
            "Diagnoses cannot be both supported and rejected: "
            + ", ".join(overlapping_ids)
        )

    flags: dict[str, bool] = {}
    for field in (
        "unsupported_signals_present",
        "unsupported_signals_material",
        "insufficient_evidence",
    ):
        if not isinstance(value[field], bool):
            raise ValueError(f"{field} must be boolean.")
        flags[field] = value[field]
    if flags["unsupported_signals_material"] and not flags[
        "unsupported_signals_present"
    ]:
        raise ValueError("Material unsupported signals must be marked as present.")
    conflicting_supported_candidates = len(candidates) > 1
    summary = value["reasoning_summary"]
    if not isinstance(summary, str) or not summary.strip():
        raise ValueError("reasoning_summary must be a non-empty string.")
    return LLMUncertaintyProposal(
        candidates=candidates,
        rejected_hypotheses=rejected,
        unsupported_signals_present=flags["unsupported_signals_present"],
        unsupported_signals_material=flags["unsupported_signals_material"],
        conflicting_supported_candidates=conflicting_supported_candidates,
        insufficient_evidence=flags["insufficient_evidence"],
        reasoning_summary=summary.strip(),
    )


def proposal_to_uncertainty(
    proposal: LLMUncertaintyProposal,
    collected: CollectedEvidence,
) -> UncertaintyAssessment:
    evidence_by_id = {
        f"E{index}": (index, item)
        for index, item in enumerate(collected.evidence, start=1)
    }
    available_sources = {item.source for item in collected.evidence}
    rejected_references = tuple(
        reference
        for rejected in proposal.rejected_hypotheses
        for reference in rejected.relevant_evidence_references
    )
    invalid_rejected = tuple(
        reference for reference in rejected_references if reference not in evidence_by_id
    )
    if invalid_rejected:
        raise UncertaintyAdapterError(
            f"Rejected-hypothesis evidence references do not exist: {invalid_rejected!r}.",
            invalid_references=True,
        )
    assessments: list[CandidateAssessment] = []
    derived_missing: set[str] = set()
    for proposed in proposal.candidates:
        references = (
            *proposed.supporting_evidence_references,
            *proposed.contradicting_evidence_references,
        )
        invalid = tuple(reference for reference in references if reference not in evidence_by_id)
        if invalid:
            raise UncertaintyAdapterError(
                f"Evidence references do not exist: {invalid!r}.",
                invalid_references=True,
            )
        supporting_references = tuple(
            evidence_by_id[reference][0]
            for reference in proposed.supporting_evidence_references
        )
        contradicting_references = tuple(
            evidence_by_id[reference][0]
            for reference in proposed.contradicting_evidence_references
        )
        required_sources = REQUIRED_EVIDENCE_SOURCES[proposed.diagnosis_id]
        missing = required_sources - available_sources
        derived_missing.update(missing)
        assessments.append(
            CandidateAssessment(
                candidate=CandidateDiagnosis(
                    diagnosis=proposed.diagnosis_id,
                    supporting_evidence_references=supporting_references,
                    contradicting_evidence_references=contradicting_references,
                    evidence_strength=proposed.evidence_strength,
                ),
                evidence_references_valid=True,
                required_sources_present=not missing,
                reported_confidence=proposed.confidence,
            )
        )
    return UncertaintyAssessment(
        candidates=tuple(assessments),
        unsupported_signals_present=proposal.unsupported_signals_present,
        unsupported_signals_material=proposal.unsupported_signals_material,
        conflicting_supported_candidates=proposal.conflicting_supported_candidates,
        insufficient_evidence=proposal.insufficient_evidence,
        missing_required_sources=tuple(sorted(derived_missing)),
    )


class LLMPolicyInvestigator:
    """Ask an LLM for uncertainty facts, then delegate the outcome to policy."""

    def __init__(
        self,
        model: StructuredModel,
        prompt_selection: str = UNCERTAINTY_PROMPT_SELECTION,
    ) -> None:
        self._model = model
        self._prompt_selection = prompt_selection

    def investigate(self, collected: CollectedEvidence) -> LLMPolicyInvestigationOutcome:
        if collected.request.deployment_id is None:
            return LLMPolicyInvestigationFailure(
                "not_evaluated", ("No deployment ID was provided.",)
            )
        if collected.deployment is None:
            return LLMPolicyInvestigationFailure(
                "not_evaluated",
                (f"Deployment {collected.request.deployment_id} was not found.",),
            )
        try:
            raw = self._model.generate(
                build_uncertainty_prompt(collected, self._prompt_selection)
            )
        except ModelRefusalError as error:
            return LLMPolicyInvestigationFailure("refused", (str(error) or "Model refused.",))
        except ModelProviderError as error:
            return LLMPolicyInvestigationFailure(
                "provider_failure", (str(error) or "Model provider failed.",)
            )
        try:
            proposal = (
                parse_candidate_semantics_proposal(raw)
                if self._prompt_selection == CANDIDATE_SEMANTICS_PROMPT_SELECTION
                else parse_uncertainty_proposal(raw)
            )
        except ValueError as error:
            return LLMPolicyInvestigationFailure(
                "invalid_response", (str(error),), raw
            )
        try:
            uncertainty = proposal_to_uncertainty(proposal, collected)
        except UncertaintyAdapterError as error:
            status = "invalid_references" if error.invalid_references else "adapter_failure"
            return LLMPolicyInvestigationFailure(status, (str(error),), raw)
        decision = decide_investigation(uncertainty)
        return LLMPolicyInvestigationSuccess(
            status="ok",
            proposal=proposal,
            uncertainty=uncertainty,
            decision=decision,
            result=decision_to_result(decision, collected),
            raw_response=raw,
        )


def decision_to_result(
    decision: InvestigationDecision,
    collected: CollectedEvidence,
) -> InvestigationResult:
    deployment_id = collected.request.deployment_id
    candidates = decision.uncertainty.candidates
    if decision.outcome is DecisionOutcome.DIAGNOSIS:
        selected = next(
            item for item in candidates if item.candidate.diagnosis == decision.diagnosis
        )
        references = selected.candidate.supporting_evidence_references
        root_cause = _root_cause(decision.diagnosis)
        confidence = selected.reported_confidence or 0.0
        limitations = (
            "Final diagnosis selected by deterministic decision policy.",
        )
        if decision.uncertainty.unsupported_signals_present:
            limitations += ("Additional unsupported signals were present but not material.",)
        return InvestigationResult(
            answer=f"Policy-controlled decision for deployment {deployment_id}. {root_cause}",
            root_cause=root_cause,
            evidence=_selected_evidence(collected, references),
            confidence=confidence,
            limitations=limitations,
        )

    references = tuple(
        dict.fromkeys(
            reference
            for item in candidates
            for reference in (
                *item.candidate.supporting_evidence_references,
                *item.candidate.contradicting_evidence_references,
            )
        )
    )
    candidate_names = tuple(item.candidate.diagnosis for item in candidates)
    limitations = (
        f"Policy outcome: {decision.outcome.value}.",
        f"Policy reason: {decision.reason.value}.",
    )
    if candidate_names:
        limitations += ("Candidate diagnoses: " + ", ".join(candidate_names) + ".",)
    return InvestigationResult(
        answer=f"The cause of deployment {deployment_id}'s failure is inconclusive.",
        root_cause=None,
        evidence=_selected_evidence(collected, references),
        confidence=max(
            (item.reported_confidence or 0.0 for item in candidates),
            default=0.0,
        ),
        limitations=limitations,
    )


def _parse_candidate(value: object, index: int) -> ProposedCandidate:
    if not isinstance(value, dict):
        raise ValueError(f"Candidate {index} must be an object.")
    _require_exact_fields(value, _CANDIDATE_FIELDS, f"candidate {index}")
    diagnosis = value["diagnosis_id"]
    if diagnosis not in SUPPORTED_DIAGNOSES:
        raise ValueError(f"Candidate {index} has an unsupported diagnosis ID.")
    supporting = _parse_references(value["supporting_evidence_references"], "supporting")
    contradicting = _parse_references(
        value["contradicting_evidence_references"], "contradicting"
    )
    if set(supporting) & set(contradicting):
        raise ValueError("Candidate evidence cannot both support and contradict the diagnosis.")
    strength = value["evidence_strength"]
    try:
        evidence_strength = EvidenceStrength(strength)
    except (ValueError, TypeError) as error:
        raise ValueError(f"Candidate {index} has invalid evidence_strength.") from error
    confidence = value["confidence"]
    if isinstance(confidence, bool) or not isinstance(confidence, (int, float)):
        raise ValueError(f"Candidate {index} confidence must be numeric.")
    confidence_value = float(confidence)
    if not math.isfinite(confidence_value) or not 0.0 <= confidence_value <= 1.0:
        raise ValueError(f"Candidate {index} confidence must be between 0.0 and 1.0.")
    return ProposedCandidate(
        diagnosis_id=diagnosis,
        supporting_evidence_references=supporting,
        contradicting_evidence_references=contradicting,
        evidence_strength=evidence_strength,
        confidence=confidence_value,
    )


def _parse_rejected_hypothesis(value: object, index: int) -> RejectedHypothesis:
    if not isinstance(value, dict):
        raise ValueError(f"Rejected hypothesis {index} must be an object.")
    _require_exact_fields(
        value, _REJECTED_HYPOTHESIS_FIELDS, f"rejected hypothesis {index}"
    )
    diagnosis = value["diagnosis_id"]
    if diagnosis not in SUPPORTED_DIAGNOSES:
        raise ValueError(f"Rejected hypothesis {index} has an unsupported diagnosis ID.")
    try:
        reason = RejectionReason(value["reason_code"])
    except (ValueError, TypeError) as error:
        raise ValueError(
            f"Rejected hypothesis {index} has an invalid reason_code."
        ) from error
    summary = value["reason_summary"]
    if not isinstance(summary, str) or not summary.strip():
        raise ValueError(f"Rejected hypothesis {index} needs a reason_summary.")
    references = _parse_references(
        value["relevant_evidence_references"], "rejected hypothesis"
    )
    confidence = value["confidence"]
    if isinstance(confidence, bool) or not isinstance(confidence, (int, float)):
        raise ValueError(f"Rejected hypothesis {index} confidence must be numeric.")
    confidence_value = float(confidence)
    if not math.isfinite(confidence_value) or not 0.0 <= confidence_value <= 1.0:
        raise ValueError(
            f"Rejected hypothesis {index} confidence must be between 0.0 and 1.0."
        )
    return RejectedHypothesis(
        diagnosis_id=diagnosis,
        reason_code=reason,
        reason_summary=summary.strip(),
        relevant_evidence_references=references,
        reported_confidence=confidence_value,
    )


def _parse_references(value: object, label: str) -> tuple[str, ...]:
    if not isinstance(value, list) or not all(
        isinstance(reference, str)
        and len(reference) >= 2
        and reference[0] == "E"
        and reference[1:].isdigit()
        and reference[1] != "0"
        for reference in value
    ):
        raise ValueError(f"{label} evidence references must use E<number> identifiers.")
    if len(value) != len(set(value)):
        raise ValueError(f"{label} evidence references must be unique.")
    return tuple(value)


def serialize_uncertainty_evidence(collected: CollectedEvidence) -> dict[str, object]:
    """Give the uncertainty model capability-like IDs without changing domain evidence."""

    payload = serialize_evidence_v2(collected)
    evidence = payload["evidence"]
    assert isinstance(evidence, list)
    payload["evidence"] = [
        {**item, "id": f"E{index}"}
        for index, item in enumerate(evidence, start=1)
    ]
    return payload


def _require_exact_fields(
    value: dict[str, object], expected: set[str], label: str
) -> None:
    if set(value) != expected:
        raise ValueError(f"Invalid {label} fields.")


def _selected_evidence(
    collected: CollectedEvidence,
    references: tuple[int, ...],
) -> tuple:
    selected = set(references)
    return tuple(
        item
        for index, item in enumerate(collected.evidence, start=1)
        if index in selected
    )


def _root_cause(diagnosis: str | None) -> str:
    assert diagnosis is not None
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
    }[diagnosis]
