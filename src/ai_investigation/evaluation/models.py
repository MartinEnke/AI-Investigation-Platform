"""Typed models for deterministic evaluation scenarios and results."""

from dataclasses import dataclass
from typing import Literal

ErrorCategory = Literal[
    "false_diagnosis",
    "unnecessary_abstention",
    "wrong_diagnosis",
    "invalid_evidence_reference",
    "missing_required_source",
    "provider_failure",
    "invalid_structured_response",
    "not_evaluated",
]


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
    expected_diagnosis_id: str | None = None
    expected_should_abstain: bool | None = None
    expected_execution_status: str | None = None
    robustness_categories: tuple[str, ...] = ()


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


@dataclass(frozen=True, slots=True)
class ScenarioRunResult:
    """Independent evaluation dimensions for one investigator and scenario."""

    scenario_id: str
    investigator: Literal["deterministic", "gemini", "llm"]
    execution_status: str
    expected_execution_status: str | None
    execution_status_matches: bool | None
    expected_diagnosis_id: str | None
    actual_diagnosis_id: str | None
    diagnosis_correct: bool | None
    expected_abstention: bool
    actual_abstention: bool | None
    abstention_correct: bool
    evidence_references_valid: bool | None
    structured_response_valid: bool | None
    expected_sources: tuple[str, ...]
    referenced_sources: tuple[str, ...]
    missing_sources: tuple[str, ...]
    unexpected_sources: tuple[str, ...]
    confidence: float | None
    latency_ms: float
    error: str | None
    semantic_correctness_status: Literal["correct", "incorrect", "not_evaluated"]
    deterministic_model_agreement: bool | None = None
    robustness_categories: tuple[str, ...] = ()
    error_category: ErrorCategory | None = None


@dataclass(frozen=True, slots=True)
class AggregateMetrics:
    """Transparent counts and denominators for an evaluation run."""

    total_scenarios: int
    total_runs: int
    completed_runs: int
    correct_diagnoses: int
    diagnosis_cases: int
    correct_abstentions: int
    abstention_cases: int
    valid_structured_responses: int
    structured_responses_assessed: int
    valid_evidence_references: int
    evidence_references_assessed: int
    provider_failures: int
    invalid_responses: int
    invalid_references: int
    semantic_failures: int
    average_latency_ms: float | None
    investigator_agreements: int
    agreement_cases: int
    average_confidence_correct: float | None
    average_confidence_incorrect: float | None
    error_categories: tuple[tuple[str, int], ...] = ()


@dataclass(frozen=True, slots=True)
class EvaluationReport:
    """A complete reusable experiment report without provider-specific fields."""

    investigator_mode: Literal["deterministic", "gemini", "llm", "both"]
    scenarios: tuple[ScenarioRunResult, ...]
    aggregate: AggregateMetrics
    confidence_disclaimer: str = "Model confidence is self-reported and uncalibrated."
