"""Typed scenario-level comparison semantics for stored evaluation experiments."""

from dataclasses import asdict, dataclass
from enum import Enum
import json
from statistics import median
from typing import TYPE_CHECKING

from ai_investigation.evaluation.models import EvaluationReport, ScenarioRunResult

if TYPE_CHECKING:
    from ai_investigation.evaluation.tracking import ExperimentRecord


class ScenarioChange(str, Enum):
    IMPROVED = "improved"
    REGRESSED = "regressed"
    UNCHANGED_CORRECT = "unchanged_correct"
    UNCHANGED_INCORRECT = "unchanged_incorrect"
    NOT_COMPARABLE = "not_comparable"


class FailureCategory(str, Enum):
    WRONG_DIAGNOSIS = "wrong_diagnosis"
    FAILED_TO_ABSTAIN = "failed_to_abstain"
    UNNECESSARY_ABSTENTION = "unnecessary_abstention"
    INVALID_EVIDENCE_REFERENCE = "invalid_evidence_reference"
    INVALID_RESPONSE = "invalid_response"
    PROVIDER_FAILURE = "provider_failure"
    STRUCTURAL_VALIDATION_FAILURE = "structural_validation_failure"
    UNKNOWN = "unknown"


@dataclass(frozen=True, slots=True)
class DimensionChange:
    baseline: str | bool | float | None
    candidate: str | bool | float | None
    comparable: bool
    changed: bool | None


@dataclass(frozen=True, slots=True)
class ScenarioComparison:
    scenario_id: str
    baseline_investigator: str | None
    candidate_investigator: str | None
    change: ScenarioChange
    baseline_semantic_status: str | None
    candidate_semantic_status: str | None
    baseline_failure: FailureCategory | None
    candidate_failure: FailureCategory | None
    semantic_correctness: DimensionChange
    abstention_correctness: DimensionChange
    abstention_behavior: DimensionChange
    structural_validity: DimensionChange
    evidence_reference_validity: DimensionChange
    execution_status: DimensionChange
    investigator_agreement: DimensionChange
    latency_ms: DimensionChange


@dataclass(frozen=True, slots=True)
class MetricDelta:
    metric: str
    baseline_value: float | None
    candidate_value: float | None
    delta: float | None
    comparable: bool
    baseline_numerator: int | None = None
    baseline_denominator: int | None = None
    candidate_numerator: int | None = None
    candidate_denominator: int | None = None

    @property
    def before(self) -> float | None:
        return self.baseline_value

    @property
    def after(self) -> float | None:
        return self.candidate_value


@dataclass(frozen=True, slots=True)
class ExperimentIdentity:
    experiment_id: str
    investigator_mode: str
    provider: str | None
    model: str | None
    scenario_count: int


@dataclass(frozen=True, slots=True)
class ComparisonSummary:
    improved: int
    regressed: int
    unchanged_correct: int
    unchanged_incorrect: int
    not_comparable: int
    baseline_failure_categories: tuple[tuple[str, int], ...]
    candidate_failure_categories: tuple[tuple[str, int], ...]


@dataclass(frozen=True, slots=True)
class ComparisonReport:
    baseline: ExperimentIdentity
    candidate: ExperimentIdentity
    metrics: tuple[MetricDelta, ...]
    scenarios: tuple[ScenarioComparison, ...]
    summary: ComparisonSummary
    recommendation: str
    recommendation_code: str
    only_baseline: tuple[str, ...]
    only_candidate: tuple[str, ...]

    @property
    def before_experiment_id(self) -> str:
        return self.baseline.experiment_id

    @property
    def after_experiment_id(self) -> str:
        return self.candidate.experiment_id

    @property
    def regressions(self) -> tuple[str, ...]:
        return tuple(
            scenario.scenario_id
            for scenario in self.scenarios
            if scenario.change is ScenarioChange.REGRESSED
        )

    @property
    def improvements(self) -> tuple[str, ...]:
        return tuple(
            scenario.scenario_id
            for scenario in self.scenarios
            if scenario.change is ScenarioChange.IMPROVED
        )

    @property
    def only_before(self) -> tuple[str, ...]:
        return self.only_baseline

    @property
    def only_after(self) -> tuple[str, ...]:
        return self.only_candidate


def compare_experiments(
    baseline: "ExperimentRecord",
    candidate: "ExperimentRecord",
) -> ComparisonReport:
    baseline_results = _result_index(baseline.report)
    candidate_results = _result_index(candidate.report)
    shared = sorted(baseline_results.keys() & candidate_results.keys())
    only_baseline = tuple(sorted(baseline_results.keys() - candidate_results.keys()))
    only_candidate = tuple(sorted(candidate_results.keys() - baseline_results.keys()))
    scenarios = tuple(
        _compare_scenario(key, baseline_results[key], candidate_results[key])
        for key in shared
    ) + tuple(
        _missing_scenario(key, baseline_results.get(key), candidate_results.get(key))
        for key in (*only_baseline, *only_candidate)
    )
    scenarios = tuple(sorted(scenarios, key=lambda item: item.scenario_id))
    summary = _summary(scenarios, baseline.report, candidate.report)
    regressions = tuple(
        item.scenario_id for item in scenarios if item.change is ScenarioChange.REGRESSED
    )
    if regressions:
        recommendation_code = "regression_warning"
        recommendation = (
            "Regression detected. Candidate should not replace the baseline without review."
        )
    else:
        recommendation_code = "no_semantic_regressions"
        recommendation = "No semantic scenario regressions detected."
        if only_baseline or only_candidate or any(
            item.change is ScenarioChange.NOT_COMPARABLE for item in scenarios
        ):
            recommendation += " Review non-comparable scenarios before replacement."
    return ComparisonReport(
        baseline=_identity(baseline),
        candidate=_identity(candidate),
        metrics=_aggregate_metrics(baseline.report, candidate.report),
        scenarios=scenarios,
        summary=summary,
        recommendation=recommendation,
        recommendation_code=recommendation_code,
        only_baseline=only_baseline,
        only_candidate=only_candidate,
    )


def comparison_to_json(report: ComparisonReport) -> str:
    return json.dumps(asdict(report), indent=2, sort_keys=True) + "\n"


def render_comparison(report: ComparisonReport) -> str:
    lines = [
        "Experiment Comparison",
        "=====================",
        "",
        f"Baseline:  {report.baseline.experiment_id}",
        f"Candidate: {report.candidate.experiment_id}",
        f"Baseline investigator:  {_identity_label(report.baseline)}",
        f"Candidate investigator: {_identity_label(report.candidate)}",
        "",
        "Summary",
        "-------",
        _presentation_recommendation(report),
        f"Scenarios: {report.baseline.scenario_count} baseline, "
        f"{report.candidate.scenario_count} candidate",
        f"Changes: {report.summary.improved} improved, "
        f"{report.summary.regressed} regressed, "
        f"{report.summary.not_comparable} not comparable",
    ]
    lines.extend(
        (
            "",
            "Scenario Changes",
            "----------------",
            f"Improvements:        {report.summary.improved}",
            f"Regressions:         {report.summary.regressed}",
            f"Unchanged correct:   {report.summary.unchanged_correct}",
            f"Unchanged incorrect: {report.summary.unchanged_incorrect}",
            f"Not comparable:      {report.summary.not_comparable}",
        )
    )
    _append_changed_scenarios(lines, "Regressions", report, ScenarioChange.REGRESSED)
    _append_changed_scenarios(lines, "Improvements", report, ScenarioChange.IMPROVED)
    if report.only_baseline or report.only_candidate:
        lines.extend(
            (
                "",
                "Unmatched Scenarios",
                "-------------------",
                f"Only baseline:  {', '.join(report.only_baseline) or 'none'}",
                f"Only candidate: {', '.join(report.only_candidate) or 'none'}",
            )
        )
    lines.extend(("", "Metrics", "-------"))
    for metric in report.metrics:
        lines.append(_render_metric(metric))
    lines.extend(
        (
            "",
            "Failure Categories",
            "------------------",
            "Baseline:  " + _render_category_counts(report.summary.baseline_failure_categories),
            "Candidate: " + _render_category_counts(report.summary.candidate_failure_categories),
        )
    )
    return "\n".join(lines) + "\n"


def failure_category(result: ScenarioRunResult) -> FailureCategory | None:
    """Return one primary engineering failure using explicit precedence.

    Execution and validation failures take precedence over semantic failures. For valid semantic
    failures, incorrect abstention behavior takes precedence over a generic wrong diagnosis.
    """

    if result.execution_status == "provider_failure":
        return FailureCategory.PROVIDER_FAILURE
    if result.execution_status == "invalid_response":
        return FailureCategory.INVALID_RESPONSE
    if result.execution_status == "invalid_references" or result.evidence_references_valid is False:
        return FailureCategory.INVALID_EVIDENCE_REFERENCE
    if result.structured_response_valid is False:
        return FailureCategory.STRUCTURAL_VALIDATION_FAILURE
    if result.semantic_correctness_status == "correct":
        return None
    if result.expected_abstention and result.actual_abstention is False:
        return FailureCategory.FAILED_TO_ABSTAIN
    if not result.expected_abstention and result.actual_abstention is True:
        return FailureCategory.UNNECESSARY_ABSTENTION
    if (
        result.expected_diagnosis_id is not None
        and result.actual_diagnosis_id is not None
        and result.expected_diagnosis_id != result.actual_diagnosis_id
    ):
        return FailureCategory.WRONG_DIAGNOSIS
    return FailureCategory.UNKNOWN


def _compare_scenario(
    scenario_id: str,
    baseline: ScenarioRunResult,
    candidate: ScenarioRunResult,
) -> ScenarioComparison:
    before = _semantic_bool(baseline)
    after = _semantic_bool(candidate)
    change = _classify(before, after)
    return ScenarioComparison(
        scenario_id=scenario_id,
        baseline_investigator=baseline.investigator,
        candidate_investigator=candidate.investigator,
        change=change,
        baseline_semantic_status=baseline.semantic_correctness_status,
        candidate_semantic_status=candidate.semantic_correctness_status,
        baseline_failure=failure_category(baseline),
        candidate_failure=failure_category(candidate),
        semantic_correctness=_dimension(before, after),
        abstention_correctness=_dimension(
            baseline.abstention_correct, candidate.abstention_correct
        ),
        abstention_behavior=_dimension(
            baseline.actual_abstention, candidate.actual_abstention
        ),
        structural_validity=_dimension(
            baseline.structured_response_valid, candidate.structured_response_valid
        ),
        evidence_reference_validity=_dimension(
            baseline.evidence_references_valid,
            candidate.evidence_references_valid,
        ),
        execution_status=_dimension(
            baseline.execution_status, candidate.execution_status
        ),
        investigator_agreement=_dimension(
            baseline.deterministic_model_agreement,
            candidate.deterministic_model_agreement,
        ),
        latency_ms=_dimension(baseline.latency_ms, candidate.latency_ms),
    )


def _missing_scenario(
    scenario_id: str,
    baseline: ScenarioRunResult | None,
    candidate: ScenarioRunResult | None,
) -> ScenarioComparison:
    unavailable = _dimension(None, None)
    return ScenarioComparison(
        scenario_id=scenario_id,
        baseline_investigator=baseline.investigator if baseline else None,
        candidate_investigator=candidate.investigator if candidate else None,
        change=ScenarioChange.NOT_COMPARABLE,
        baseline_semantic_status=(baseline.semantic_correctness_status if baseline else None),
        candidate_semantic_status=(candidate.semantic_correctness_status if candidate else None),
        baseline_failure=failure_category(baseline) if baseline else None,
        candidate_failure=failure_category(candidate) if candidate else None,
        semantic_correctness=unavailable,
        abstention_correctness=unavailable,
        abstention_behavior=unavailable,
        structural_validity=unavailable,
        evidence_reference_validity=unavailable,
        execution_status=unavailable,
        investigator_agreement=unavailable,
        latency_ms=unavailable,
    )


def _classify(before: bool | None, after: bool | None) -> ScenarioChange:
    if before is None or after is None:
        return ScenarioChange.NOT_COMPARABLE
    if not before and after:
        return ScenarioChange.IMPROVED
    if before and not after:
        return ScenarioChange.REGRESSED
    return ScenarioChange.UNCHANGED_CORRECT if before else ScenarioChange.UNCHANGED_INCORRECT


def _semantic_bool(result: ScenarioRunResult) -> bool | None:
    if result.semantic_correctness_status == "correct":
        return True
    if result.semantic_correctness_status == "incorrect":
        return False
    return None


def _dimension(
    baseline: str | bool | float | None,
    candidate: str | bool | float | None,
) -> DimensionChange:
    comparable = baseline is not None and candidate is not None
    return DimensionChange(
        baseline=baseline,
        candidate=candidate,
        comparable=comparable,
        changed=baseline != candidate if comparable else None,
    )


def _result_index(report: EvaluationReport) -> dict[str, ScenarioRunResult]:
    counts: dict[str, int] = {}
    for result in report.scenarios:
        counts[result.scenario_id] = counts.get(result.scenario_id, 0) + 1
    return {
        (
            result.scenario_id
            if counts[result.scenario_id] == 1
            else f"{result.scenario_id}:{result.investigator}"
        ): result
        for result in report.scenarios
    }


def _identity(record: "ExperimentRecord") -> ExperimentIdentity:
    metadata = record.metadata
    return ExperimentIdentity(
        experiment_id=metadata.experiment_id,
        investigator_mode=metadata.investigator_mode,
        provider=metadata.provider,
        model=metadata.model,
        scenario_count=metadata.scenario_count,
    )


def _aggregate_metrics(
    baseline: EvaluationReport,
    candidate: EvaluationReport,
) -> tuple[MetricDelta, ...]:
    baseline_semantic = _semantic_counts(baseline)
    candidate_semantic = _semantic_counts(candidate)
    return (
        _counted_metric("semantic_accuracy", baseline_semantic, candidate_semantic),
        _counted_metric(
            "abstention_accuracy",
            (baseline.aggregate.correct_abstentions, baseline.aggregate.abstention_cases),
            (candidate.aggregate.correct_abstentions, candidate.aggregate.abstention_cases),
        ),
        _counted_metric(
            "evidence_reference_validity",
            (
                baseline.aggregate.valid_evidence_references,
                baseline.aggregate.evidence_references_assessed,
            ),
            (
                candidate.aggregate.valid_evidence_references,
                candidate.aggregate.evidence_references_assessed,
            ),
        ),
        _counted_metric(
            "structured_response_validity",
            (
                baseline.aggregate.valid_structured_responses,
                baseline.aggregate.structured_responses_assessed,
            ),
            (
                candidate.aggregate.valid_structured_responses,
                candidate.aggregate.structured_responses_assessed,
            ),
        ),
        _absolute_metric(
            "provider_failures",
            baseline.aggregate.provider_failures,
            candidate.aggregate.provider_failures,
        ),
        _optional_metric(
            "validation_failures",
            _validation_failure_count(baseline),
            _validation_failure_count(candidate),
        ),
        _optional_metric(
            "average_latency_ms",
            baseline.aggregate.average_latency_ms,
            candidate.aggregate.average_latency_ms,
        ),
        _optional_metric(
            "median_scenario_latency_ms",
            _median_latency(baseline),
            _median_latency(candidate),
        ),
        _counted_metric(
            "agreement_rate",
            (baseline.aggregate.investigator_agreements, baseline.aggregate.agreement_cases),
            (candidate.aggregate.investigator_agreements, candidate.aggregate.agreement_cases),
        ),
    )


def _semantic_counts(report: EvaluationReport) -> tuple[int, int]:
    comparable = tuple(
        result
        for result in report.scenarios
        if result.semantic_correctness_status in ("correct", "incorrect")
    )
    return (
        sum(result.semantic_correctness_status == "correct" for result in comparable),
        len(comparable),
    )


def _median_latency(report: EvaluationReport) -> float | None:
    values = tuple(result.latency_ms for result in report.scenarios)
    return median(values) if values else None


def _validation_failure_count(report: EvaluationReport) -> float | None:
    if not any(result.investigator == "gemini" for result in report.scenarios):
        return None
    return float(report.aggregate.invalid_responses + report.aggregate.invalid_references)


def _counted_metric(
    name: str,
    baseline: tuple[int, int],
    candidate: tuple[int, int],
) -> MetricDelta:
    before = baseline[0] / baseline[1] if baseline[1] else None
    after = candidate[0] / candidate[1] if candidate[1] else None
    comparable = baseline[1] > 0 and baseline[1] == candidate[1]
    return MetricDelta(
        metric=name,
        baseline_value=before,
        candidate_value=after,
        delta=after - before if comparable and before is not None and after is not None else None,
        comparable=comparable,
        baseline_numerator=baseline[0],
        baseline_denominator=baseline[1],
        candidate_numerator=candidate[0],
        candidate_denominator=candidate[1],
    )


def _absolute_metric(name: str, baseline: int, candidate: int) -> MetricDelta:
    return MetricDelta(
        metric=name,
        baseline_value=float(baseline),
        candidate_value=float(candidate),
        delta=float(candidate - baseline),
        comparable=True,
    )


def _optional_metric(name: str, baseline: float | None, candidate: float | None) -> MetricDelta:
    comparable = baseline is not None and candidate is not None
    return MetricDelta(
        metric=name,
        baseline_value=baseline,
        candidate_value=candidate,
        delta=candidate - baseline if comparable else None,
        comparable=comparable,
    )


def _summary(
    scenarios: tuple[ScenarioComparison, ...],
    baseline: EvaluationReport,
    candidate: EvaluationReport,
) -> ComparisonSummary:
    return ComparisonSummary(
        improved=sum(item.change is ScenarioChange.IMPROVED for item in scenarios),
        regressed=sum(item.change is ScenarioChange.REGRESSED for item in scenarios),
        unchanged_correct=sum(
            item.change is ScenarioChange.UNCHANGED_CORRECT for item in scenarios
        ),
        unchanged_incorrect=sum(
            item.change is ScenarioChange.UNCHANGED_INCORRECT for item in scenarios
        ),
        not_comparable=sum(item.change is ScenarioChange.NOT_COMPARABLE for item in scenarios),
        baseline_failure_categories=_failure_counts(baseline),
        candidate_failure_categories=_failure_counts(candidate),
    )


def _failure_counts(report: EvaluationReport) -> tuple[tuple[str, int], ...]:
    counts: dict[str, int] = {}
    for result in report.scenarios:
        category = failure_category(result)
        if category is not None:
            counts[category.value] = counts.get(category.value, 0) + 1
    return tuple(sorted(counts.items()))


def _render_metric(metric: MetricDelta) -> str:
    label = _metric_label(metric.metric)
    if not metric.comparable:
        return f"{label}: not comparable"
    if metric.baseline_denominator is not None:
        return (
            f"{label}: "
            f"{metric.baseline_numerator}/{metric.baseline_denominator} -> "
            f"{metric.candidate_numerator}/{metric.candidate_denominator} "
            f"(delta {metric.delta:+.4f})"
        )
    if metric.metric.endswith("latency_ms"):
        return (
            f"{label}: {metric.baseline_value:.2f} ms -> "
            f"{metric.candidate_value:.2f} ms (delta {metric.delta:+.2f} ms)"
        )
    if metric.metric in ("provider_failures", "validation_failures"):
        return (
            f"{label}: {metric.baseline_value:.0f} -> "
            f"{metric.candidate_value:.0f} (delta {metric.delta:+.0f})"
        )
    return (
        f"{label}: {metric.baseline_value:.3f} -> "
        f"{metric.candidate_value:.3f} (delta {metric.delta:+.3f})"
    )


def _presentation_recommendation(report: ComparisonReport) -> str:
    if report.summary.regressed:
        noun = "regression" if report.summary.regressed == 1 else "regressions"
        return (
            f"Regression detected: {report.summary.regressed} semantic {noun}. "
            "Candidate should not replace the baseline without review."
        )
    if report.summary.not_comparable:
        return (
            "No semantic regressions detected, but non-comparable scenarios require review "
            "before replacement."
        )
    return "No semantic regressions detected. Candidate passes the current semantic regression gate."


def _metric_label(metric: str) -> str:
    return {
        "semantic_accuracy": "Semantic accuracy",
        "abstention_accuracy": "Abstention accuracy",
        "evidence_reference_validity": "Evidence-reference validity",
        "structured_response_validity": "Structured-response validity",
        "provider_failures": "Provider failures",
        "validation_failures": "Validation failures",
        "average_latency_ms": "Average latency",
        "median_scenario_latency_ms": "Median scenario latency",
        "agreement_rate": "Agreement rate",
    }[metric]


def _append_changed_scenarios(
    lines: list[str],
    title: str,
    report: ComparisonReport,
    change: ScenarioChange,
) -> None:
    selected = tuple(item for item in report.scenarios if item.change is change)
    if not selected:
        return
    lines.extend(("", title, "-" * len(title)))
    for scenario in selected:
        lines.append(f"- {scenario.scenario_id}")
        lines.append(f"  Before: {scenario.baseline_semantic_status}")
        lines.append(f"  After:  {scenario.candidate_semantic_status}")
        if scenario.candidate_failure is not None:
            lines.append(f"  Failure: {scenario.candidate_failure.value}")


def _render_category_counts(counts: tuple[tuple[str, int], ...]) -> str:
    return ", ".join(f"{name}={count}" for name, count in counts) or "none"


def _identity_label(identity: ExperimentIdentity) -> str:
    provider = "/".join(item for item in (identity.provider, identity.model) if item)
    return f"{identity.investigator_mode} ({provider})" if provider else identity.investigator_mode
