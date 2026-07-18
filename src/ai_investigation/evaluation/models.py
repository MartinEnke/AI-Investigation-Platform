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


@dataclass(frozen=True, slots=True)
class EvaluationResult:
    """Exact comparison results for one evaluation scenario."""

    scenario_id: str
    passed: bool
    root_cause_matches: bool
    inconclusive_matches: bool
    evidence_sources_match: bool
    actual_root_cause: str | None
    actual_inconclusive: bool
    actual_evidence_sources: tuple[str, ...]
    failures: tuple[str, ...]

