"""Explicit uncertainty models and a pure deterministic investigation policy."""

from dataclasses import dataclass
from enum import Enum


SUPPORTED_DIAGNOSES = frozenset(
    {
        "health_check_timeout",
        "missing_environment_variable",
        "database_migration_failure",
        "missing_database_configuration",
        "database_contention_blocked_migration",
    }
)


class DecisionOutcome(str, Enum):
    DIAGNOSIS = "diagnosis"
    ABSTENTION = "abstention"
    NEEDS_REVIEW = "needs_review"


class EvidenceStrength(str, Enum):
    WEAK = "weak"
    MODERATE = "moderate"
    STRONG = "strong"


class DecisionReason(str, Enum):
    SINGLE_SUPPORTED_CANDIDATE = "single_supported_candidate"
    NO_SUPPORTED_CANDIDATE = "no_supported_candidate"
    INSUFFICIENT_EVIDENCE = "insufficient_evidence"
    MISSING_REQUIRED_EVIDENCE = "missing_required_evidence"
    CONFLICTING_SUPPORTED_CANDIDATES = "conflicting_supported_candidates"
    UNSUPPORTED_SIGNALS_ONLY = "unsupported_signals_only"
    INVALID_CANDIDATE_EVIDENCE = "invalid_candidate_evidence"
    UNRESOLVED_CONTRADICTORY_EVIDENCE = "unresolved_contradictory_evidence"


@dataclass(frozen=True, slots=True)
class CandidateDiagnosis:
    """One supported cause and the evidence offered for and against it."""

    diagnosis: str
    supporting_evidence_references: tuple[int, ...]
    contradicting_evidence_references: tuple[int, ...]
    evidence_strength: EvidenceStrength

    def __post_init__(self) -> None:
        if not isinstance(self.evidence_strength, EvidenceStrength):
            raise ValueError("evidence_strength must be an EvidenceStrength value.")
        if self.diagnosis not in SUPPORTED_DIAGNOSES:
            raise ValueError(f"Unsupported diagnosis: {self.diagnosis}.")
        _validate_references(
            self.supporting_evidence_references,
            "supporting_evidence_references",
        )
        _validate_references(
            self.contradicting_evidence_references,
            "contradicting_evidence_references",
        )
        overlap = set(self.supporting_evidence_references) & set(
            self.contradicting_evidence_references
        )
        if overlap:
            raise ValueError("Evidence cannot both support and contradict one candidate.")


@dataclass(frozen=True, slots=True)
class CandidateAssessment:
    """Deterministic validity and source-completeness facts for one candidate."""

    candidate: CandidateDiagnosis
    evidence_references_valid: bool = True
    required_sources_present: bool = True
    reported_confidence: float | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.evidence_references_valid, bool):
            raise ValueError("evidence_references_valid must be boolean.")
        if not isinstance(self.required_sources_present, bool):
            raise ValueError("required_sources_present must be boolean.")
        confidence = self.reported_confidence
        if confidence is not None and (
            isinstance(confidence, bool) or not 0.0 <= confidence <= 1.0
        ):
            raise ValueError("Reported confidence must be between 0.0 and 1.0.")


@dataclass(frozen=True, slots=True)
class UncertaintyAssessment:
    """Candidate-independent uncertainty facts supplied to decision policy."""

    candidates: tuple[CandidateAssessment, ...] = ()
    unsupported_signals_present: bool = False
    unsupported_signals_material: bool = False
    conflicting_supported_candidates: bool = False
    insufficient_evidence: bool = False
    missing_required_sources: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if not isinstance(self.candidates, tuple):
            raise ValueError("candidates must be an immutable tuple.")
        if not isinstance(self.missing_required_sources, tuple):
            raise ValueError("missing_required_sources must be an immutable tuple.")
        for field_name in (
            "unsupported_signals_present",
            "unsupported_signals_material",
            "conflicting_supported_candidates",
            "insufficient_evidence",
        ):
            if not isinstance(getattr(self, field_name), bool):
                raise ValueError(f"{field_name} must be boolean.")
        diagnoses = tuple(item.candidate.diagnosis for item in self.candidates)
        if len(diagnoses) != len(set(diagnoses)):
            raise ValueError("Candidate diagnoses must be unique.")
        if self.unsupported_signals_material and not self.unsupported_signals_present:
            raise ValueError("Material unsupported signals must be marked as present.")
        if self.conflicting_supported_candidates and len(self.candidates) < 2:
            raise ValueError("Conflicting supported candidates require at least two candidates.")
        if any(not item.required_sources_present for item in self.candidates) and not (
            self.missing_required_sources
        ):
            raise ValueError("Missing required candidate sources must be identified.")
        if len(self.missing_required_sources) != len(set(self.missing_required_sources)):
            raise ValueError("Missing required sources must be unique.")
        if any(not source for source in self.missing_required_sources):
            raise ValueError("Missing required sources must be non-empty identifiers.")


@dataclass(frozen=True, slots=True)
class InvestigationDecision:
    """A validated policy outcome, separate from any investigator confidence."""

    outcome: DecisionOutcome
    diagnosis: str | None
    reason: DecisionReason
    uncertainty: UncertaintyAssessment
    requires_review: bool
    explanation: str | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.outcome, DecisionOutcome):
            raise ValueError("outcome must be a DecisionOutcome value.")
        if not isinstance(self.reason, DecisionReason):
            raise ValueError("reason must be a DecisionReason value.")
        if not isinstance(self.requires_review, bool):
            raise ValueError("requires_review must be boolean.")
        if self.outcome is DecisionOutcome.DIAGNOSIS:
            if self.diagnosis is None:
                raise ValueError("A diagnosis outcome requires a diagnosis.")
            if self.diagnosis not in SUPPORTED_DIAGNOSES:
                raise ValueError(f"Unsupported diagnosis: {self.diagnosis}.")
            if self.requires_review:
                raise ValueError("A diagnosis outcome cannot require review.")
            candidates = {
                item.candidate.diagnosis for item in self.uncertainty.candidates
            }
            if self.diagnosis not in candidates:
                raise ValueError("The selected diagnosis must be an assessed candidate.")
        elif self.diagnosis is not None:
            raise ValueError("Abstention and review outcomes cannot select a diagnosis.")

        expected_review = self.outcome is DecisionOutcome.NEEDS_REVIEW
        if self.requires_review is not expected_review:
            raise ValueError("requires_review must agree with the decision outcome.")


def decide_investigation(assessment: UncertaintyAssessment) -> InvestigationDecision:
    """Apply the fixed uncertainty policy without using confidence or side effects."""

    if any(
        not item.evidence_references_valid
        or not item.candidate.supporting_evidence_references
        for item in assessment.candidates
    ):
        return _abstain(assessment, DecisionReason.INVALID_CANDIDATE_EVIDENCE)
    if assessment.missing_required_sources or any(
        not item.required_sources_present for item in assessment.candidates
    ):
        return _abstain(assessment, DecisionReason.MISSING_REQUIRED_EVIDENCE)
    if assessment.insufficient_evidence:
        return _abstain(assessment, DecisionReason.INSUFFICIENT_EVIDENCE)
    if not assessment.candidates:
        reason = (
            DecisionReason.UNSUPPORTED_SIGNALS_ONLY
            if assessment.unsupported_signals_present
            else DecisionReason.NO_SUPPORTED_CANDIDATE
        )
        return _abstain(assessment, reason)
    if assessment.conflicting_supported_candidates or len(assessment.candidates) > 1:
        return _review(assessment, DecisionReason.CONFLICTING_SUPPORTED_CANDIDATES)

    candidate = assessment.candidates[0].candidate
    if candidate.contradicting_evidence_references or assessment.unsupported_signals_material:
        return _review(assessment, DecisionReason.UNRESOLVED_CONTRADICTORY_EVIDENCE)
    if candidate.evidence_strength is EvidenceStrength.WEAK:
        return _abstain(assessment, DecisionReason.INSUFFICIENT_EVIDENCE)
    return InvestigationDecision(
        outcome=DecisionOutcome.DIAGNOSIS,
        diagnosis=candidate.diagnosis,
        reason=DecisionReason.SINGLE_SUPPORTED_CANDIDATE,
        uncertainty=assessment,
        requires_review=False,
    )


def _abstain(
    assessment: UncertaintyAssessment,
    reason: DecisionReason,
) -> InvestigationDecision:
    return InvestigationDecision(
        outcome=DecisionOutcome.ABSTENTION,
        diagnosis=None,
        reason=reason,
        uncertainty=assessment,
        requires_review=False,
    )


def _review(
    assessment: UncertaintyAssessment,
    reason: DecisionReason,
) -> InvestigationDecision:
    return InvestigationDecision(
        outcome=DecisionOutcome.NEEDS_REVIEW,
        diagnosis=None,
        reason=reason,
        uncertainty=assessment,
        requires_review=True,
    )


def _validate_references(references: tuple[int, ...], field: str) -> None:
    if not isinstance(references, tuple):
        raise ValueError(f"{field} must be an immutable tuple.")
    if any(
        isinstance(reference, bool) or not isinstance(reference, int) or reference < 1
        for reference in references
    ):
        raise ValueError(f"{field} must contain positive integer evidence references.")
    if len(references) != len(set(references)):
        raise ValueError(f"{field} must not contain duplicate evidence references.")
