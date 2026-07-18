"""Typed models for deterministic evaluation scenarios and results."""

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class EvaluationScenario:
    """Expected investigator behavior for one synthetic question."""

    id: str
    question: str
    expected_root_cause: str | None
    expected_inconclusive: bool
    expected_evidence_sources: tuple[str, ...]
    expected_confidence: float | None = None
    expected_limitations: tuple[str, ...] | None = None
    expected_decision_outcome: str | None = None
    expected_matched_rule_ids: tuple[str, ...] | None = None


@dataclass(frozen=True, slots=True)
class EvaluationResult:
    """Exact comparison results for one evaluation scenario."""

    scenario_id: str
    passed: bool
    root_cause_matches: bool
    inconclusive_matches: bool
    evidence_sources_match: bool
    confidence_matches: bool | None
    limitations_match: bool | None
    decision_outcome_matches: bool | None
    matched_rule_ids_match: bool | None
    actual_root_cause: str | None
    actual_inconclusive: bool
    actual_evidence_sources: tuple[str, ...]
    expected_confidence: float | None
    actual_confidence: float
    expected_limitations: tuple[str, ...] | None
    actual_limitations: tuple[str, ...]
    expected_decision_outcome: str | None
    actual_decision_outcome: str | None
    expected_matched_rule_ids: tuple[str, ...] | None
    actual_matched_rule_ids: tuple[str, ...] | None
    failures: tuple[str, ...]
