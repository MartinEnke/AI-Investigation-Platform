from dataclasses import replace
from io import StringIO
import json
from pathlib import Path

import pytest

from ai_investigation.evaluation.loader import load_scenarios
from ai_investigation.evaluation.runner import evaluate_scenario, print_report, run_evaluation
from ai_investigation.investigator import DeploymentFailureInvestigator


def scenarios(fixture_directory: Path):
    return load_scenarios(fixture_directory / "evaluation_scenarios.json")


def write_scenario(path: Path, **overrides: object) -> Path:
    scenario = {
        "id": "schema-test",
        "question": "Why did deployment deploy-1042 fail?",
        "expected_root_cause": None,
        "expected_inconclusive": True,
        "expected_evidence_sources": ["deployment"],
    }
    scenario.update(overrides)
    path.write_text(json.dumps([scenario]), encoding="utf-8")
    return path


def test_scenario_loading(fixture_directory: Path) -> None:
    loaded = scenarios(fixture_directory)

    assert len(loaded) == 11
    assert loaded[0].id == "supported-health-check-timeout"
    assert loaded[0].expected_evidence_sources == ("deployment", "logs", "service_health")


def test_existing_scenario_without_optional_expectations_loads(
    fixture_directory: Path,
) -> None:
    scenario = scenarios(fixture_directory)[1]

    assert scenario.expected_confidence is None
    assert scenario.expected_limitations is None
    assert scenario.expected_decision_outcome is None
    assert scenario.expected_matched_rule_ids is None


def test_selective_trace_expectations_are_loaded(fixture_directory: Path) -> None:
    scenario = scenarios(fixture_directory)[0]

    assert scenario.expected_decision_outcome == "single_match"
    assert scenario.expected_matched_rule_ids == ("health_check_timeout",)


def test_valid_optional_confidence_is_parsed(tmp_path: Path) -> None:
    scenario = load_scenarios(
        write_scenario(tmp_path / "scenarios.json", expected_confidence=0.25)
    )[0]

    assert scenario.expected_confidence == 0.25


def test_valid_optional_limitations_are_parsed(tmp_path: Path) -> None:
    scenario = load_scenarios(
        write_scenario(
            tmp_path / "scenarios.json",
            expected_limitations=["First limitation.", "Second limitation."],
        )
    )[0]

    assert scenario.expected_limitations == ("First limitation.", "Second limitation.")


def test_optional_experiment_expectations_are_parsed(tmp_path: Path) -> None:
    scenario = load_scenarios(
        write_scenario(
            tmp_path / "scenarios.json",
            expected_diagnosis_id="health_check_timeout",
            expected_should_abstain=False,
            expected_execution_status="provider_failure",
        )
    )[0]

    assert scenario.expected_diagnosis_id == "health_check_timeout"
    assert scenario.expected_should_abstain is False
    assert scenario.expected_execution_status == "provider_failure"


def test_scenario_without_experiment_expectations_remains_valid(tmp_path: Path) -> None:
    scenario = load_scenarios(write_scenario(tmp_path / "scenarios.json"))[0]

    assert scenario.expected_diagnosis_id is None
    assert scenario.expected_should_abstain is None
    assert scenario.expected_execution_status is None


def test_invalid_experiment_execution_status_is_rejected(tmp_path: Path) -> None:
    path = write_scenario(
        tmp_path / "scenarios.json",
        expected_execution_status="retried",
    )

    with pytest.raises(ValueError, match="invalid expected_execution_status"):
        load_scenarios(path)


@pytest.mark.parametrize("invalid_confidence", (True, "high", [0.25]))
def test_invalid_confidence_type_is_rejected(
    tmp_path: Path,
    invalid_confidence: object,
) -> None:
    path = write_scenario(
        tmp_path / "scenarios.json",
        expected_confidence=invalid_confidence,
    )

    with pytest.raises(ValueError, match="invalid expected_confidence"):
        load_scenarios(path)


@pytest.mark.parametrize("invalid_limitations", ("one limitation", ["valid", 2]))
def test_invalid_limitations_are_rejected(
    tmp_path: Path,
    invalid_limitations: object,
) -> None:
    path = write_scenario(
        tmp_path / "scenarios.json",
        expected_limitations=invalid_limitations,
    )

    with pytest.raises(ValueError, match="invalid expected_limitations"):
        load_scenarios(path)


def test_supported_diagnosis_evaluates_correctly(
    fixture_directory: Path,
    investigator: DeploymentFailureInvestigator,
) -> None:
    result = evaluate_scenario(scenarios(fixture_directory)[0], investigator)

    assert result.passed
    assert result.root_cause_matches
    assert not result.actual_inconclusive


def test_remaining_scenarios_evaluate_correctly(
    fixture_directory: Path,
    investigator: DeploymentFailureInvestigator,
) -> None:
    results = run_evaluation(scenarios(fixture_directory)[1:], investigator)

    assert all(result.passed for result in results)


def test_mismatched_expectation_is_detected(
    fixture_directory: Path,
    investigator: DeploymentFailureInvestigator,
) -> None:
    scenario = replace(
        scenarios(fixture_directory)[0],
        expected_evidence_sources=("logs", "deployment", "service_health"),
    )

    result = evaluate_scenario(scenario, investigator)

    assert not result.passed
    assert not result.evidence_sources_match
    assert result.failures == (
        "evidence sources: expected ('logs', 'deployment', 'service_health'), "
        "got ('deployment', 'logs', 'service_health')",
    )


def test_confidence_mismatch_causes_scenario_failure(
    fixture_directory: Path,
    investigator: DeploymentFailureInvestigator,
) -> None:
    scenario = replace(scenarios(fixture_directory)[-1], expected_confidence=0.5)

    result = evaluate_scenario(scenario, investigator)

    assert not result.passed
    assert result.confidence_matches is False
    assert result.expected_confidence == 0.5
    assert result.actual_confidence == 0.25
    assert result.failures == ("confidence: expected 0.5, got 0.25",)


def test_limitations_mismatch_causes_scenario_failure(
    fixture_directory: Path,
    investigator: DeploymentFailureInvestigator,
) -> None:
    scenario = replace(
        scenarios(fixture_directory)[-1],
        expected_limitations=("Different limitation.",),
    )

    result = evaluate_scenario(scenario, investigator)

    assert not result.passed
    assert result.limitations_match is False
    assert result.expected_limitations == ("Different limitation.",)
    assert result.actual_limitations == (
        "Conflicting supported failure patterns matched the available evidence.",
    )
    assert result.failures == (
        "limitations: expected ('Different limitation.',), got "
        "('Conflicting supported failure patterns matched the available evidence.',)",
    )


def test_limitations_are_compared_in_exact_order(
    fixture_directory: Path,
    investigator: DeploymentFailureInvestigator,
) -> None:
    scenario = replace(
        scenarios(fixture_directory)[2],
        question="Why did deployment deploy-1043 fail?",
        expected_evidence_sources=("deployment",),
        expected_limitations=(
            "No service-health record is available for the target service.",
            "No error log is available for the deployment.",
        ),
    )

    result = evaluate_scenario(scenario, investigator)

    assert result.limitations_match is False
    assert result.actual_limitations == (
        "No error log is available for the deployment.",
        "No service-health record is available for the target service.",
    )


def test_decision_outcome_mismatch_is_reported(
    fixture_directory: Path,
    investigator: DeploymentFailureInvestigator,
) -> None:
    scenario = replace(
        scenarios(fixture_directory)[0],
        expected_decision_outcome="no_match",
    )

    result = evaluate_scenario(scenario, investigator)

    assert not result.passed
    assert result.decision_outcome_matches is False
    assert result.failures == (
        "decision outcome: expected 'no_match', got 'single_match'",
    )


def test_matched_rule_ids_mismatch_is_reported(
    fixture_directory: Path,
    investigator: DeploymentFailureInvestigator,
) -> None:
    scenario = replace(
        scenarios(fixture_directory)[0],
        expected_matched_rule_ids=("database_migration_failure",),
    )

    result = evaluate_scenario(scenario, investigator)

    assert not result.passed
    assert result.matched_rule_ids_match is False
    assert result.failures == (
        "matched rule IDs: expected ('database_migration_failure',), "
        "got ('health_check_timeout',)",
    )


def test_supplied_dataset_passes_and_prints_summary(
    fixture_directory: Path,
    investigator: DeploymentFailureInvestigator,
) -> None:
    results = run_evaluation(scenarios(fixture_directory), investigator)
    output = StringIO()
    print_report(results, output)

    assert all(result.passed for result in results)
    assert output.getvalue().splitlines()[-1] == "Summary: 11 passed, 0 failed"
