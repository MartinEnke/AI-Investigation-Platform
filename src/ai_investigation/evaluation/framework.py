"""Reusable deterministic and model experiment evaluation over shared evidence."""

from collections import Counter
from collections.abc import Callable, Iterable
from dataclasses import asdict, replace
import json
import time
from typing import Literal

from ai_investigation.evaluation.models import (
    AggregateMetrics,
    ErrorCategory,
    EvaluationReport,
    EvaluationScenario,
    ScenarioRunResult,
)
from ai_investigation.evidence import CollectedEvidence, EvidenceCollector
from ai_investigation.investigator import DeploymentFailureInvestigator, request_from_question
from ai_investigation.investigators import (
    DeterministicInvestigatorAdapter,
    InvestigatorExecution,
    LLMInvestigatorAdapter,
)
from ai_investigation.llm_investigator import (
    DEFAULT_PROMPT_VERSION,
    LLMInvestigator,
    PromptVersion,
    StructuredModel,
)
from ai_investigation.models import InvestigationResult

InvestigatorMode = Literal["deterministic", "gemini", "llm", "both"]
EventObserver = Callable[
    [str, str | None, str | None, str, str, float | None, tuple[tuple[str, str], ...]],
    None,
]


def run_experiment(
    scenarios: Iterable[EvaluationScenario],
    collector: EvidenceCollector,
    deterministic_investigator: DeploymentFailureInvestigator,
    *,
    investigator_mode: InvestigatorMode = "deterministic",
    structured_model: StructuredModel | None = None,
    prompt_version: PromptVersion = DEFAULT_PROMPT_VERSION,
    request_delay_seconds: float = 0.0,
    sleep: Callable[[float], None] = time.sleep,
    clock: Callable[[], float] = time.perf_counter,
    observer: EventObserver | None = None,
) -> EvaluationReport:
    """Evaluate selected reasoning paths, sharing one evidence collection per scenario."""

    if investigator_mode not in ("deterministic", "gemini", "llm", "both"):
        raise ValueError(f"Unknown investigator mode: {investigator_mode}.")
    if investigator_mode in ("gemini", "llm", "both") and structured_model is None:
        raise ValueError("A structured model is required for Gemini evaluation.")
    if request_delay_seconds < 0:
        raise ValueError("Request delay must be non-negative.")

    selected_scenarios = tuple(scenarios)
    experiment_started = clock()
    _emit(observer, "experiment_started", None, None, "experiment", "started")
    results: list[ScenarioRunResult] = []
    deterministic_adapter = DeterministicInvestigatorAdapter(deterministic_investigator)
    provider_investigator = (
        LLMInvestigator(structured_model)
        if structured_model is not None and prompt_version == DEFAULT_PROMPT_VERSION
        else LLMInvestigator(structured_model, prompt_version)
        if structured_model is not None
        else None
    )
    llm_investigator = (
        LLMInvestigatorAdapter(
            structured_model,
            provider_investigator,
            prompt_version,
        )
        if structured_model is not None and investigator_mode in ("gemini", "llm", "both")
        else None
    )
    provider_request_completed = False

    for scenario in selected_scenarios:
        scenario_started = clock()
        _emit(observer, "scenario_started", scenario.id, None, "scenario", "started")
        _emit(
            observer,
            "evidence_collection_started",
            scenario.id,
            None,
            "evidence_collection",
            "started",
        )
        evidence_started = clock()
        collected = collector.collect(request_from_question(scenario.question))
        evidence_duration = (clock() - evidence_started) * 1000
        _emit(
            observer,
            "evidence_collection_completed",
            scenario.id,
            None,
            "evidence_collection",
            "completed",
            evidence_duration,
        )
        deterministic_result: ScenarioRunResult | None = None
        model_result: ScenarioRunResult | None = None

        if investigator_mode in ("deterministic", "both"):
            _emit(
                observer,
                "deterministic_investigation_started",
                scenario.id,
                "deterministic",
                "investigation",
                "started",
            )
            started = clock()
            execution = deterministic_adapter.investigate(collected)
            assert execution.result is not None
            investigation = execution.result
            latency_ms = (clock() - started) * 1000
            _emit(
                observer,
                "deterministic_investigation_completed",
                scenario.id,
                "deterministic",
                "investigation",
                "completed",
                latency_ms,
            )
            evaluation_started = clock()
            deterministic_result = _evaluate_deterministic(
                scenario, collected, investigation, latency_ms
            )
            evaluation_duration = (clock() - evaluation_started) * 1000
            _emit(
                observer,
                "scenario_evaluated",
                scenario.id,
                "deterministic",
                "evaluation",
                deterministic_result.semantic_correctness_status,
                evaluation_duration,
            )

        if investigator_mode in ("gemini", "llm", "both"):
            assert llm_investigator is not None
            model_investigator_name = "llm" if investigator_mode == "llm" else "gemini"
            provider_call_expected = (
                collected.request.deployment_id is not None
                and collected.deployment is not None
            )
            if (
                provider_call_expected
                and provider_request_completed
                and request_delay_seconds > 0
            ):
                sleep(request_delay_seconds)
            _emit(
                observer,
                "model_investigation_started",
                scenario.id,
                model_investigator_name,
                "investigation",
                "started",
            )
            started = clock()
            outcome = llm_investigator.investigate(collected)
            if provider_call_expected:
                provider_request_completed = True
            latency_ms = (clock() - started) * 1000
            outcome_status = outcome.status
            _emit(
                observer,
                "model_investigation_completed",
                scenario.id,
                model_investigator_name,
                "investigation",
                outcome_status,
                latency_ms,
            )
            _emit(
                observer,
                "validation_completed",
                scenario.id,
                model_investigator_name,
                "validation",
                outcome_status,
            )
            evaluation_started = clock()
            model_result = _evaluate_model(
                scenario,
                outcome,
                latency_ms,
                investigator=model_investigator_name,
            )
            evaluation_duration = (clock() - evaluation_started) * 1000
            event_type = (
                "scenario_failed"
                if model_result.execution_status not in ("completed", "not_evaluated")
                else "scenario_evaluated"
            )
            _emit(
                observer,
                event_type,
                scenario.id,
                model_investigator_name,
                "evaluation",
                model_result.semantic_correctness_status,
                evaluation_duration,
                (("execution_status", model_result.execution_status),),
            )

        if deterministic_result is not None and model_result is not None:
            agreement = _agreement(deterministic_result, model_result)
            deterministic_result = replace(
                deterministic_result,
                deterministic_model_agreement=agreement,
            )
            model_result = replace(
                model_result,
                deterministic_model_agreement=agreement,
            )

        if deterministic_result is not None:
            results.append(deterministic_result)
        if model_result is not None:
            results.append(model_result)
        _emit(
            observer,
            "scenario_completed",
            scenario.id,
            None,
            "scenario",
            "completed",
            (clock() - scenario_started) * 1000,
        )

    collected_results = tuple(results)
    report = EvaluationReport(
        investigator_mode=investigator_mode,
        scenarios=collected_results,
        aggregate=_aggregate(len(selected_scenarios), collected_results),
    )
    _emit(
        observer,
        "experiment_completed",
        None,
        None,
        "experiment",
        "completed",
        (clock() - experiment_started) * 1000,
    )
    return report


def _emit(
    observer: EventObserver | None,
    event_type: str,
    scenario_id: str | None,
    investigator: str | None,
    stage: str,
    status: str,
    duration_ms: float | None = None,
    details: tuple[tuple[str, str], ...] = (),
) -> None:
    if observer is not None:
        observer(
            event_type,
            scenario_id,
            investigator,
            stage,
            status,
            duration_ms,
            details,
        )


def report_to_json(report: EvaluationReport) -> str:
    """Serialize a report with deterministic keys and formatting."""

    return json.dumps(asdict(report), indent=2, sort_keys=True) + "\n"


def render_text_report(report: EvaluationReport) -> str:
    """Render all aggregate and scenario dimensions without hiding failures."""

    aggregate = report.aggregate
    lines = [
        "# AI Investigation Evaluation",
        "",
        f"Investigator: {report.investigator_mode}",
        f"Scenarios: {aggregate.total_scenarios}",
        "",
        "## Summary",
        "",
    ]
    if aggregate.diagnosis_cases:
        lines.append(
            _ratio("Diagnosis accuracy", aggregate.correct_diagnoses, aggregate.diagnosis_cases)
        )
    if aggregate.abstention_cases:
        lines.append(
            _ratio(
                "Abstention accuracy",
                aggregate.correct_abstentions,
                aggregate.abstention_cases,
            )
        )
    if aggregate.structured_responses_assessed:
        lines.append(
            _ratio(
                "Structured-response validity",
                aggregate.valid_structured_responses,
                aggregate.structured_responses_assessed,
            )
        )
    if aggregate.evidence_references_assessed:
        lines.append(
            _ratio(
                "Evidence-reference validity",
                aggregate.valid_evidence_references,
                aggregate.evidence_references_assessed,
            )
        )
    lines.extend(
        (
            f"Provider failures: {aggregate.provider_failures}",
            f"Invalid responses: {aggregate.invalid_responses}",
            f"Invalid references: {aggregate.invalid_references}",
            f"Semantic failures: {aggregate.semantic_failures}",
        )
    )
    if aggregate.error_categories:
        lines.extend(("", "Error categories:"))
        lines.extend(
            f"- {category}: {count}"
            for category, count in aggregate.error_categories
        )
    if aggregate.average_latency_ms is not None:
        lines.append(f"Average latency: {aggregate.average_latency_ms:.3f} ms")
    if aggregate.agreement_cases:
        lines.append(
            _ratio(
                "Deterministic/model agreement",
                aggregate.investigator_agreements,
                aggregate.agreement_cases,
            )
        )
    if report.investigator_mode in ("gemini", "llm", "both"):
        lines.extend((f"Confidence: {report.confidence_disclaimer}", ""))
    else:
        lines.append("")
    lines.append("## Scenario Results")

    for result in report.scenarios:
        lines.extend(("", *_scenario_lines(result)))
        if result.deterministic_model_agreement is not None:
            lines.append(
                "Investigators agree: "
                + str(result.deterministic_model_agreement).lower()
                + " (agreement is not correctness)"
            )
        if result.error is not None:
            lines.append(f"Error: {result.error}")
    return "\n".join(lines) + "\n"


def _evaluate_deterministic(
    scenario: EvaluationScenario,
    collected: CollectedEvidence,
    investigation: InvestigationResult,
    latency_ms: float,
) -> ScenarioRunResult:
    diagnosis_id = _deterministic_diagnosis_id(investigation)
    abstained = investigation.root_cause is None
    references_valid = all(
        any(item is collected_item for collected_item in collected.evidence)
        for item in investigation.evidence
    )
    sources = tuple(item.source for item in investigation.evidence)
    return _scenario_result(
        scenario=scenario,
        investigator="deterministic",
        execution_status="completed",
        actual_diagnosis_id=diagnosis_id,
        actual_abstention=abstained,
        evidence_references_valid=references_valid,
        structured_response_valid=None,
        referenced_sources=sources,
        confidence=investigation.confidence,
        latency_ms=latency_ms,
        error=None,
    )


def _evaluate_model(
    scenario: EvaluationScenario,
    outcome: InvestigatorExecution,
    latency_ms: float,
    *,
    investigator: Literal["gemini", "llm"] = "gemini",
) -> ScenarioRunResult:
    if outcome.result is not None:
        return _scenario_result(
            scenario=scenario,
            investigator=investigator,
            execution_status="completed",
            actual_diagnosis_id=outcome.diagnosis_id,
            actual_abstention=outcome.result.root_cause is None,
            evidence_references_valid=True,
            structured_response_valid=True,
            referenced_sources=tuple(item.source for item in outcome.result.evidence),
            confidence=outcome.result.confidence,
            latency_ms=latency_ms,
            error=None,
        )

    structured_valid = outcome.structured_response_valid
    references_valid = outcome.evidence_references_valid
    actual_abstention = True if outcome.status == "not_evaluated" else None
    return _scenario_result(
        scenario=scenario,
        investigator=investigator,
        execution_status=outcome.status,
        actual_diagnosis_id=None,
        actual_abstention=actual_abstention,
        evidence_references_valid=references_valid,
        structured_response_valid=structured_valid,
        referenced_sources=(),
        confidence=None,
        latency_ms=latency_ms,
        error="; ".join(outcome.errors),
    )


def _scenario_result(
    *,
    scenario: EvaluationScenario,
    investigator: Literal["deterministic", "gemini", "llm"],
    execution_status: str,
    actual_diagnosis_id: str | None,
    actual_abstention: bool | None,
    evidence_references_valid: bool | None,
    structured_response_valid: bool | None,
    referenced_sources: tuple[str, ...],
    confidence: float | None,
    latency_ms: float,
    error: str | None,
) -> ScenarioRunResult:
    expected_diagnosis = _expected_diagnosis_id(scenario)
    expected_abstention = _expected_abstention(scenario)
    diagnosis_correct = (
        actual_diagnosis_id == expected_diagnosis
        if expected_diagnosis is not None and actual_abstention is not None
        else None
    )
    abstention_correct = actual_abstention == expected_abstention
    if actual_abstention is None:
        semantic_status = "not_evaluated"
    elif expected_diagnosis is not None:
        semantic_status = "correct" if diagnosis_correct else "incorrect"
    else:
        semantic_status = "correct" if abstention_correct else "incorrect"
    missing, unexpected = _source_differences(
        scenario.expected_evidence_sources,
        referenced_sources,
    )
    semantic_correctness_status: Literal["correct", "incorrect", "not_evaluated"] = (
        semantic_status
    )
    return ScenarioRunResult(
        scenario_id=scenario.id,
        investigator=investigator,
        execution_status=execution_status,
        expected_execution_status=scenario.expected_execution_status,
        execution_status_matches=(
            None
            if scenario.expected_execution_status is None
            else execution_status == scenario.expected_execution_status
        ),
        expected_diagnosis_id=expected_diagnosis,
        actual_diagnosis_id=actual_diagnosis_id,
        diagnosis_correct=diagnosis_correct,
        expected_abstention=expected_abstention,
        actual_abstention=actual_abstention,
        abstention_correct=abstention_correct,
        evidence_references_valid=evidence_references_valid,
        structured_response_valid=structured_response_valid,
        expected_sources=scenario.expected_evidence_sources,
        referenced_sources=referenced_sources,
        missing_sources=missing,
        unexpected_sources=unexpected,
        confidence=confidence,
        latency_ms=latency_ms,
        error=error,
        semantic_correctness_status=semantic_correctness_status,
        robustness_categories=scenario.robustness_categories,
        error_category=_error_category(
            execution_status=execution_status,
            expected_diagnosis_id=expected_diagnosis,
            expected_abstention=expected_abstention,
            actual_diagnosis_id=actual_diagnosis_id,
            actual_abstention=actual_abstention,
            semantic_status=semantic_correctness_status,
            error=error,
        ),
    )


def _expected_diagnosis_id(scenario: EvaluationScenario) -> str | None:
    if scenario.expected_diagnosis_id is not None:
        return scenario.expected_diagnosis_id
    if scenario.expected_matched_rule_ids is not None and len(
        scenario.expected_matched_rule_ids
    ) == 1:
        return scenario.expected_matched_rule_ids[0]
    return None


def _expected_abstention(scenario: EvaluationScenario) -> bool:
    return (
        scenario.expected_should_abstain
        if scenario.expected_should_abstain is not None
        else scenario.expected_inconclusive
    )


def _deterministic_diagnosis_id(result: InvestigationResult) -> str | None:
    trace = result.decision_trace
    if trace is not None and trace.outcome == "single_match" and len(trace.matched_rule_ids) == 1:
        return trace.matched_rule_ids[0]
    return None


def _source_differences(
    expected: tuple[str, ...], actual: tuple[str, ...]
) -> tuple[tuple[str, ...], tuple[str, ...]]:
    missing_counts = Counter(expected) - Counter(actual)
    unexpected_counts = Counter(actual) - Counter(expected)
    missing = _ordered_counter_values(expected, missing_counts)
    unexpected = _ordered_counter_values(actual, unexpected_counts)
    return missing, unexpected


def _ordered_counter_values(
    values: tuple[str, ...], counts: Counter[str]
) -> tuple[str, ...]:
    selected: list[str] = []
    for value in values:
        if counts[value] > 0:
            selected.append(value)
            counts[value] -= 1
    return tuple(selected)


def _agreement(
    deterministic: ScenarioRunResult,
    model: ScenarioRunResult,
) -> bool | None:
    if model.execution_status != "completed" or model.actual_abstention is None:
        return None
    return (
        deterministic.actual_diagnosis_id == model.actual_diagnosis_id
        and deterministic.actual_abstention == model.actual_abstention
    )


def _aggregate(
    total_scenarios: int,
    results: tuple[ScenarioRunResult, ...],
) -> AggregateMetrics:
    diagnosis_results = tuple(
        result for result in results if result.expected_diagnosis_id is not None
    )
    abstention_results = tuple(result for result in results if result.expected_abstention)
    structured = tuple(
        result for result in results if result.structured_response_valid is not None
    )
    references = tuple(
        result for result in results if result.evidence_references_valid is not None
    )
    agreements = tuple(
        result
        for result in results
        if result.investigator in ("gemini", "llm")
        and result.deterministic_model_agreement is not None
    )
    correct_confidence = tuple(
        result.confidence
        for result in results
        if result.investigator in ("gemini", "llm")
        and result.semantic_correctness_status == "correct"
        and result.confidence is not None
    )
    incorrect_confidence = tuple(
        result.confidence
        for result in results
        if result.investigator in ("gemini", "llm")
        and result.semantic_correctness_status == "incorrect"
        and result.confidence is not None
    )
    error_counts = Counter(
        result.error_category for result in results if result.error_category is not None
    )
    return AggregateMetrics(
        total_scenarios=total_scenarios,
        total_runs=len(results),
        completed_runs=sum(result.execution_status == "completed" for result in results),
        correct_diagnoses=sum(result.diagnosis_correct is True for result in diagnosis_results),
        diagnosis_cases=len(diagnosis_results),
        correct_abstentions=sum(result.abstention_correct for result in abstention_results),
        abstention_cases=len(abstention_results),
        valid_structured_responses=sum(
            result.structured_response_valid is True for result in structured
        ),
        structured_responses_assessed=len(structured),
        valid_evidence_references=sum(
            result.evidence_references_valid is True for result in references
        ),
        evidence_references_assessed=len(references),
        provider_failures=sum(result.execution_status == "provider_failure" for result in results),
        invalid_responses=sum(result.execution_status == "invalid_response" for result in results),
        invalid_references=sum(
            result.execution_status == "invalid_references" for result in results
        ),
        semantic_failures=sum(
            result.semantic_correctness_status == "incorrect" for result in results
        ),
        average_latency_ms=_average(tuple(result.latency_ms for result in results)),
        investigator_agreements=sum(
            result.deterministic_model_agreement is True for result in agreements
        ),
        agreement_cases=len(agreements),
        average_confidence_correct=_average(correct_confidence),
        average_confidence_incorrect=_average(incorrect_confidence),
        error_categories=tuple(
            (category, error_counts[category])
            for category in (
                "false_diagnosis",
                "unnecessary_abstention",
                "wrong_diagnosis",
                "invalid_evidence_reference",
                "missing_required_source",
                "provider_failure",
                "invalid_structured_response",
                "not_evaluated",
            )
            if error_counts[category]
        ),
    )


def _error_category(
    *,
    execution_status: str,
    expected_diagnosis_id: str | None,
    expected_abstention: bool,
    actual_diagnosis_id: str | None,
    actual_abstention: bool | None,
    semantic_status: str,
    error: str | None,
) -> ErrorCategory | None:
    if execution_status == "provider_failure":
        return "provider_failure"
    if execution_status == "invalid_response":
        return "invalid_structured_response"
    if execution_status == "invalid_references":
        if error is not None and "missing required sources" in error.casefold():
            return "missing_required_source"
        return "invalid_evidence_reference"
    if execution_status == "not_evaluated" and semantic_status != "correct":
        return "not_evaluated"
    if semantic_status != "incorrect":
        return None
    if expected_abstention and actual_diagnosis_id is not None:
        return "false_diagnosis"
    if expected_diagnosis_id is not None and actual_abstention is True:
        return "unnecessary_abstention"
    if expected_diagnosis_id is not None and actual_diagnosis_id is not None:
        return "wrong_diagnosis"
    return "not_evaluated"


def _average(values: tuple[float, ...]) -> float | None:
    return sum(values) / len(values) if values else None


def _ratio(label: str, numerator: int, denominator: int) -> str:
    return f"{label}: {numerator}/{denominator}" if denominator else f"{label}: not applicable"


def _actual_outcome_label(result: ScenarioRunResult) -> str:
    if result.actual_diagnosis_id is not None:
        return f"diagnosis: {result.actual_diagnosis_id}"
    if result.actual_abstention is True:
        return "abstention"
    return "no valid outcome"


def _scenario_lines(result: ScenarioRunResult) -> list[str]:
    lines = [
        f"### {result.scenario_id} [{result.investigator}]",
        f"Execution status: {result.execution_status}",
    ]
    if result.expected_execution_status is not None:
        lines.extend(
            (
                f"Expected execution status: {result.expected_execution_status}",
                "Execution status matches: "
                + str(result.execution_status_matches).lower(),
            )
        )
    lines.extend(
        (
            f"Semantic correctness: {result.semantic_correctness_status}",
            "Expected outcome: "
            + (
                f"diagnosis: {result.expected_diagnosis_id}"
                if result.expected_diagnosis_id is not None
                else "abstention"
            ),
            f"Actual outcome: {_actual_outcome_label(result)}",
        )
    )
    if result.expected_abstention or result.actual_abstention is True:
        lines.append(
            "Abstention assessment: "
            + ("correct" if result.abstention_correct else "incorrect")
        )
    if result.evidence_references_valid is not None:
        lines.append(
            "Evidence references: "
            + ("valid" if result.evidence_references_valid else "invalid")
        )
    if result.structured_response_valid is not None:
        lines.append(
            "Structured response: "
            + ("valid" if result.structured_response_valid else "invalid")
        )
    if result.referenced_sources:
        lines.append(
            f"Referenced sources: {', '.join(_ordered_unique(result.referenced_sources))}"
        )
    if result.missing_sources:
        lines.append(f"Missing sources: {', '.join(result.missing_sources)}")
    if result.unexpected_sources:
        lines.append(f"Unexpected sources: {', '.join(result.unexpected_sources)}")
    if result.confidence is not None:
        lines.append(f"Confidence: {result.confidence:.2f}")
    if result.error_category is not None:
        lines.append(f"Error category: {result.error_category}")
    lines.append(f"Latency: {result.latency_ms:.3f} ms")
    return lines


def _ordered_unique(values: tuple[str, ...]) -> tuple[str, ...]:
    return tuple(dict.fromkeys(values))
