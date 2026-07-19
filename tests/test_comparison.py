import json
from dataclasses import replace
from pathlib import Path

import pytest

from ai_investigation.evaluation.comparison import (
    FailureCategory,
    ScenarioChange,
    compare_experiments,
    comparison_to_json,
    failure_category,
    render_comparison,
)
from ai_investigation.evaluation.models import (
    AggregateMetrics,
    EvaluationReport,
    ScenarioRunResult,
)
from ai_investigation.evaluation.tracking import (
    ExperimentMetadata,
    ExperimentRecord,
    TimingSummary,
    save_experiment,
)
from ai_investigation.experiments import main as experiments_main


def scenario(
    scenario_id: str,
    semantic: str,
    *,
    expected_abstention: bool = False,
    actual_abstention: bool | None = False,
    expected_diagnosis: str | None = "health_check_timeout",
    actual_diagnosis: str | None = "health_check_timeout",
    execution_status: str = "completed",
    references_valid: bool | None = True,
    structured_valid: bool | None = True,
    agreement: bool | None = None,
    latency: float = 10.0,
) -> ScenarioRunResult:
    return ScenarioRunResult(
        scenario_id=scenario_id,
        investigator="gemini",
        execution_status=execution_status,
        expected_execution_status=None,
        execution_status_matches=None,
        expected_diagnosis_id=expected_diagnosis,
        actual_diagnosis_id=actual_diagnosis,
        diagnosis_correct=(
            actual_diagnosis == expected_diagnosis
            if expected_diagnosis is not None and actual_abstention is not None
            else None
        ),
        expected_abstention=expected_abstention,
        actual_abstention=actual_abstention,
        abstention_correct=actual_abstention == expected_abstention,
        evidence_references_valid=references_valid,
        structured_response_valid=structured_valid,
        expected_sources=("deployment", "logs"),
        referenced_sources=("deployment", "logs"),
        missing_sources=(),
        unexpected_sources=(),
        confidence=0.9,
        latency_ms=latency,
        error=None,
        semantic_correctness_status=semantic,
        deterministic_model_agreement=agreement,
    )


def record(experiment_id: str, results: tuple[ScenarioRunResult, ...]) -> ExperimentRecord:
    comparable = tuple(
        item for item in results if item.semantic_correctness_status in ("correct", "incorrect")
    )
    abstentions = tuple(item for item in results if item.expected_abstention)
    structured = tuple(item for item in results if item.structured_response_valid is not None)
    references = tuple(item for item in results if item.evidence_references_valid is not None)
    aggregate = AggregateMetrics(
        total_scenarios=len({item.scenario_id for item in results}),
        total_runs=len(results),
        completed_runs=sum(item.execution_status == "completed" for item in results),
        correct_diagnoses=sum(
            item.diagnosis_correct is True for item in results if item.expected_diagnosis_id
        ),
        diagnosis_cases=sum(item.expected_diagnosis_id is not None for item in results),
        correct_abstentions=sum(item.abstention_correct for item in abstentions),
        abstention_cases=len(abstentions),
        valid_structured_responses=sum(item.structured_response_valid is True for item in structured),
        structured_responses_assessed=len(structured),
        valid_evidence_references=sum(item.evidence_references_valid is True for item in references),
        evidence_references_assessed=len(references),
        provider_failures=sum(item.execution_status == "provider_failure" for item in results),
        invalid_responses=sum(item.execution_status == "invalid_response" for item in results),
        invalid_references=sum(item.execution_status == "invalid_references" for item in results),
        semantic_failures=sum(item.semantic_correctness_status == "incorrect" for item in comparable),
        average_latency_ms=(
            sum(item.latency_ms for item in results) / len(results) if results else None
        ),
        investigator_agreements=sum(item.deterministic_model_agreement is True for item in results),
        agreement_cases=sum(item.deterministic_model_agreement is not None for item in results),
        average_confidence_correct=None,
        average_confidence_incorrect=None,
    )
    metadata = ExperimentMetadata(
        experiment_id=experiment_id,
        created_at="2026-07-19T12:00:00Z",
        investigator_mode="gemini",
        scenario_source="scenarios.json",
        scenario_count=len({item.scenario_id for item in results}),
        scenario_ids=tuple(item.scenario_id for item in results),
        provider="fake-provider",
        model="fake-model",
        repository_revision=None,
        python_version="3.13",
        platform="test",
        configuration=(),
        tags=(),
        notes=None,
    )
    timing = TimingSummary(0.0, None, 0.0, None, 0.0, 0.0, 0.0)
    return ExperimentRecord(
        schema_version=1,
        metadata=metadata,
        report=EvaluationReport("gemini", results, aggregate),
        timing=timing,
        events=(),
    )


@pytest.mark.parametrize(
    ("before", "after", "expected"),
    (
        ("incorrect", "correct", ScenarioChange.IMPROVED),
        ("correct", "incorrect", ScenarioChange.REGRESSED),
        ("correct", "correct", ScenarioChange.UNCHANGED_CORRECT),
        ("incorrect", "incorrect", ScenarioChange.UNCHANGED_INCORRECT),
        ("not_evaluated", "correct", ScenarioChange.NOT_COMPARABLE),
    ),
)
def test_primary_scenario_change_classification(
    before: str,
    after: str,
    expected: ScenarioChange,
) -> None:
    comparison = compare_experiments(
        record("baseline", (scenario("shared", before),)),
        record("candidate", (scenario("shared", after),)),
    )

    assert comparison.scenarios[0].change is expected


@pytest.mark.parametrize(
    ("result", "expected"),
    (
        (
            scenario("x", "incorrect", actual_diagnosis="database_migration_failure"),
            FailureCategory.WRONG_DIAGNOSIS,
        ),
        (
            scenario(
                "x",
                "incorrect",
                expected_abstention=True,
                actual_abstention=False,
                expected_diagnosis=None,
                actual_diagnosis="database_migration_failure",
            ),
            FailureCategory.FAILED_TO_ABSTAIN,
        ),
        (
            scenario("x", "incorrect", actual_abstention=True, actual_diagnosis=None),
            FailureCategory.UNNECESSARY_ABSTENTION,
        ),
        (
            scenario("x", "not_evaluated", execution_status="invalid_references", references_valid=False),
            FailureCategory.INVALID_EVIDENCE_REFERENCE,
        ),
        (
            scenario("x", "not_evaluated", execution_status="provider_failure"),
            FailureCategory.PROVIDER_FAILURE,
        ),
        (
            scenario("x", "not_evaluated", execution_status="invalid_response", structured_valid=False),
            FailureCategory.INVALID_RESPONSE,
        ),
        (
            scenario("x", "not_evaluated", execution_status="completed", structured_valid=False),
            FailureCategory.STRUCTURAL_VALIDATION_FAILURE,
        ),
        (
            scenario("x", "not_evaluated", execution_status="refused", actual_abstention=None),
            FailureCategory.UNKNOWN,
        ),
    ),
)
def test_failure_category_precedence(
    result: ScenarioRunResult,
    expected: FailureCategory,
) -> None:
    assert failure_category(result) is expected


def test_scenarios_match_by_id_with_deterministic_order_and_unmatched_visibility() -> None:
    baseline = record(
        "baseline",
        (scenario("b", "correct"), scenario("a", "incorrect"), scenario("old", "correct")),
    )
    candidate = record(
        "candidate",
        (scenario("new", "correct"), scenario("a", "correct"), scenario("b", "correct")),
    )

    comparison = compare_experiments(baseline, candidate)

    assert [item.scenario_id for item in comparison.scenarios] == ["a", "b", "new", "old"]
    assert comparison.only_baseline == ("old",)
    assert comparison.only_candidate == ("new",)
    assert comparison.summary.improved == 1
    assert comparison.summary.unchanged_correct == 1
    assert comparison.summary.not_comparable == 2


def test_equal_accuracy_can_contain_improvement_and_regression() -> None:
    baseline = record(
        "baseline", (scenario("fixed", "incorrect"), scenario("broken", "correct"))
    )
    candidate = record(
        "candidate", (scenario("fixed", "correct"), scenario("broken", "incorrect"))
    )

    comparison = compare_experiments(baseline, candidate)
    accuracy = next(item for item in comparison.metrics if item.metric == "semantic_accuracy")

    assert accuracy.delta == 0.0
    assert comparison.improvements == ("fixed",)
    assert comparison.regressions == ("broken",)
    assert comparison.recommendation_code == "regression_warning"


def test_agreement_does_not_replace_semantic_correctness() -> None:
    baseline = record("baseline", (scenario("x", "incorrect", agreement=True),))
    candidate = record("candidate", (scenario("x", "incorrect", agreement=True),))

    compared = compare_experiments(baseline, candidate).scenarios[0]

    assert compared.change is ScenarioChange.UNCHANGED_INCORRECT
    assert compared.investigator_agreement.changed is False


def test_incompatible_denominators_are_not_comparable() -> None:
    baseline = record(
        "baseline", (scenario("x", "correct", structured_valid=None),)
    )
    candidate = record(
        "candidate", (scenario("x", "correct", structured_valid=True),)
    )

    metric = next(
        item
        for item in compare_experiments(baseline, candidate).metrics
        if item.metric == "structured_response_validity"
    )

    assert not metric.comparable
    assert metric.delta is None


def test_text_and_json_render_the_same_comparison_model() -> None:
    comparison = compare_experiments(
        record("baseline", (scenario("x", "correct", latency=10.123456),)),
        record("candidate", (scenario("x", "incorrect", latency=20.987654),)),
    )

    text = render_comparison(comparison)
    first_json = comparison_to_json(comparison)

    assert "Regression detected" in text
    assert "Failure: unknown" in text
    assert "Average latency: 10.12 ms -> 20.99 ms (delta +10.86 ms)" in text
    assert text.index("Summary") < text.index("Scenario Changes")
    assert text.index("Scenario Changes") < text.index("Metrics")
    assert text.index("Metrics") < text.index("Failure Categories")
    assert first_json == comparison_to_json(comparison)
    parsed = json.loads(first_json)
    assert parsed["scenarios"][0]["change"] == "regressed"
    assert parsed["scenarios"][0]["latency_ms"]["baseline"] == 10.123456
    assert parsed["scenarios"][0]["latency_ms"]["candidate"] == 20.987654
    assert parsed["recommendation_code"] == "regression_warning"


def test_comparison_cli_exit_codes_and_json_gate(tmp_path: Path, capsys) -> None:
    baseline = record("baseline", (scenario("x", "correct"),))
    passing = record("passing", (scenario("x", "correct"),))
    regressed = record("regressed", (scenario("x", "incorrect"),))
    for item in (baseline, passing, regressed):
        save_experiment(item, "report", tmp_path)

    assert experiments_main(
        ["--experiment-dir", str(tmp_path), "compare", "baseline", "passing", "--fail-on-regression"]
    ) == 0
    capsys.readouterr()
    assert experiments_main(
        [
            "--experiment-dir",
            str(tmp_path),
            "compare",
            "baseline",
            "regressed",
            "--json",
            "--fail-on-regression",
        ]
    ) == 1
    assert json.loads(capsys.readouterr().out)["summary"]["regressed"] == 1

    assert experiments_main(
        ["--experiment-dir", str(tmp_path), "compare", "missing", "passing"]
    ) == 2
    assert "Error:" in capsys.readouterr().err
