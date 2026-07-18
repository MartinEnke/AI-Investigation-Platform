from dataclasses import replace
from io import StringIO
from pathlib import Path

from ai_investigation.evaluation.loader import load_scenarios
from ai_investigation.evaluation.runner import evaluate_scenario, print_report, run_evaluation
from ai_investigation.investigator import DeploymentFailureInvestigator


def scenarios(fixture_directory: Path):
    return load_scenarios(fixture_directory / "evaluation_scenarios.json")


def test_scenario_loading(fixture_directory: Path) -> None:
    loaded = scenarios(fixture_directory)

    assert len(loaded) == 7
    assert loaded[0].id == "supported-health-check-timeout"
    assert loaded[0].expected_evidence_sources == ("deployment", "logs", "service_health")


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


def test_supplied_dataset_passes_and_prints_summary(
    fixture_directory: Path,
    investigator: DeploymentFailureInvestigator,
) -> None:
    results = run_evaluation(scenarios(fixture_directory), investigator)
    output = StringIO()
    print_report(results, output)

    assert all(result.passed for result in results)
    assert output.getvalue().splitlines()[-1] == "Summary: 7 passed, 0 failed"
