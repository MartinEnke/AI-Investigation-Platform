from dataclasses import FrozenInstanceError

import pytest

from ai_investigation.decision_policy import (
    CandidateAssessment,
    CandidateDiagnosis,
    DecisionOutcome,
    DecisionReason,
    EvidenceStrength,
    InvestigationDecision,
    UncertaintyAssessment,
    decide_investigation,
)


def candidate(
    diagnosis: str = "health_check_timeout",
    *,
    strength: EvidenceStrength = EvidenceStrength.STRONG,
    supporting: tuple[int, ...] = (1, 2, 3),
    contradicting: tuple[int, ...] = (),
    references_valid: bool = True,
    sources_present: bool = True,
    confidence: float | None = None,
) -> CandidateAssessment:
    return CandidateAssessment(
        CandidateDiagnosis(diagnosis, supporting, contradicting, strength),
        evidence_references_valid=references_valid,
        required_sources_present=sources_present,
        reported_confidence=confidence,
    )


def test_single_strong_complete_candidate_produces_diagnosis() -> None:
    assessment = UncertaintyAssessment(candidates=(candidate(),))

    decision = decide_investigation(assessment)

    assert decision.outcome is DecisionOutcome.DIAGNOSIS
    assert decision.diagnosis == "health_check_timeout"
    assert decision.reason is DecisionReason.SINGLE_SUPPORTED_CANDIDATE
    assert decision.requires_review is False


def test_irrelevant_unsupported_signals_do_not_undermine_diagnosis() -> None:
    assessment = UncertaintyAssessment(
        candidates=(candidate(),),
        unsupported_signals_present=True,
        unsupported_signals_material=False,
    )

    assert decide_investigation(assessment).outcome is DecisionOutcome.DIAGNOSIS


@pytest.mark.parametrize(
    ("assessment", "reason"),
    (
        (UncertaintyAssessment(), DecisionReason.NO_SUPPORTED_CANDIDATE),
        (
            UncertaintyAssessment(unsupported_signals_present=True),
            DecisionReason.UNSUPPORTED_SIGNALS_ONLY,
        ),
        (
            UncertaintyAssessment(candidates=(candidate(),), insufficient_evidence=True),
            DecisionReason.INSUFFICIENT_EVIDENCE,
        ),
        (
            UncertaintyAssessment(
                candidates=(candidate(sources_present=False),),
                missing_required_sources=("service_health",),
            ),
            DecisionReason.MISSING_REQUIRED_EVIDENCE,
        ),
        (
            UncertaintyAssessment(candidates=(candidate(references_valid=False),)),
            DecisionReason.INVALID_CANDIDATE_EVIDENCE,
        ),
        (
            UncertaintyAssessment(candidates=(candidate(supporting=()),)),
            DecisionReason.INVALID_CANDIDATE_EVIDENCE,
        ),
        (
            UncertaintyAssessment(
                candidates=(candidate(strength=EvidenceStrength.WEAK),)
            ),
            DecisionReason.INSUFFICIENT_EVIDENCE,
        ),
    ),
)
def test_abstention_policy_branches(
    assessment: UncertaintyAssessment,
    reason: DecisionReason,
) -> None:
    decision = decide_investigation(assessment)

    assert decision.outcome is DecisionOutcome.ABSTENTION
    assert decision.diagnosis is None
    assert decision.reason is reason
    assert decision.requires_review is False


def test_two_strong_supported_candidates_require_review() -> None:
    assessment = UncertaintyAssessment(
        candidates=(
            candidate("health_check_timeout"),
            candidate("database_migration_failure", supporting=(1, 4)),
        ),
        conflicting_supported_candidates=True,
    )

    decision = decide_investigation(assessment)

    assert decision.outcome is DecisionOutcome.NEEDS_REVIEW
    assert decision.reason is DecisionReason.CONFLICTING_SUPPORTED_CANDIDATES
    assert decision.diagnosis is None
    assert decision.requires_review is True


def test_strong_and_moderate_conflicting_candidates_require_review() -> None:
    assessment = UncertaintyAssessment(
        candidates=(
            candidate("health_check_timeout"),
            candidate(
                "missing_environment_variable",
                strength=EvidenceStrength.MODERATE,
                supporting=(1, 4),
            ),
        ),
        conflicting_supported_candidates=True,
    )

    assert decide_investigation(assessment).outcome is DecisionOutcome.NEEDS_REVIEW


@pytest.mark.parametrize(
    "assessment",
    (
        UncertaintyAssessment(candidates=(candidate(contradicting=(4,)),)),
        UncertaintyAssessment(
            candidates=(candidate(),),
            unsupported_signals_present=True,
            unsupported_signals_material=True,
        ),
    ),
)
def test_material_contradictory_evidence_requires_review(
    assessment: UncertaintyAssessment,
) -> None:
    decision = decide_investigation(assessment)

    assert decision.outcome is DecisionOutcome.NEEDS_REVIEW
    assert decision.reason is DecisionReason.UNRESOLVED_CONTRADICTORY_EVIDENCE


def test_duplicate_candidate_diagnoses_are_rejected_deterministically() -> None:
    with pytest.raises(ValueError, match="must be unique"):
        UncertaintyAssessment(candidates=(candidate(), candidate(supporting=(1, 5))))


@pytest.mark.parametrize(
    "candidate_value",
    (
        lambda: CandidateDiagnosis(
            "unsupported_label", (1,), (), EvidenceStrength.STRONG
        ),
        lambda: CandidateDiagnosis(
            "health_check_timeout", (0,), (), EvidenceStrength.STRONG
        ),
        lambda: CandidateDiagnosis(
            "health_check_timeout", (1, 1), (), EvidenceStrength.STRONG
        ),
        lambda: CandidateDiagnosis(
            "health_check_timeout", (1, 2), (2,), EvidenceStrength.STRONG
        ),
    ),
)
def test_candidate_invariants_reject_invalid_diagnoses_and_references(
    candidate_value,
) -> None:
    with pytest.raises(ValueError):
        candidate_value()


@pytest.mark.parametrize(
    "outcome,diagnosis,requires_review",
    (
        (DecisionOutcome.DIAGNOSIS, None, False),
        (DecisionOutcome.ABSTENTION, "health_check_timeout", False),
        (DecisionOutcome.NEEDS_REVIEW, "health_check_timeout", True),
        (DecisionOutcome.ABSTENTION, None, True),
        (DecisionOutcome.NEEDS_REVIEW, None, False),
    ),
)
def test_decision_invariants_reject_contradictory_outcomes(
    outcome: DecisionOutcome,
    diagnosis: str | None,
    requires_review: bool,
) -> None:
    with pytest.raises(ValueError):
        InvestigationDecision(
            outcome=outcome,
            diagnosis=diagnosis,
            reason=DecisionReason.INSUFFICIENT_EVIDENCE,
            uncertainty=UncertaintyAssessment(candidates=(candidate(),)),
            requires_review=requires_review,
        )


def test_uncertainty_invariants_reject_contradictory_states() -> None:
    with pytest.raises(ValueError, match="marked as present"):
        UncertaintyAssessment(unsupported_signals_material=True)
    with pytest.raises(ValueError, match="at least two"):
        UncertaintyAssessment(
            candidates=(candidate(),), conflicting_supported_candidates=True
        )
    with pytest.raises(ValueError, match="must be identified"):
        UncertaintyAssessment(candidates=(candidate(sources_present=False),))


def test_confidence_is_observed_but_does_not_control_policy() -> None:
    low = decide_investigation(
        UncertaintyAssessment(candidates=(candidate(confidence=0.05),))
    )
    high = decide_investigation(
        UncertaintyAssessment(candidates=(candidate(confidence=0.99),))
    )

    assert (low.outcome, low.diagnosis, low.reason) == (
        high.outcome,
        high.diagnosis,
        high.reason,
    )


def test_uncertainty_models_are_immutable() -> None:
    assessment = UncertaintyAssessment(candidates=(candidate(),))

    with pytest.raises(FrozenInstanceError):
        assessment.insufficient_evidence = True  # type: ignore[misc]
