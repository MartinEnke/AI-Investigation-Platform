from pathlib import Path

from ai_investigation import evaluate as evaluate_cli
from ai_investigation.evaluation.framework import _error_category, render_text_report, run_experiment
from ai_investigation.evaluation.loader import ROBUSTNESS_CATEGORIES, load_scenarios
from ai_investigation.evaluation.tracking import load_experiment
from ai_investigation.evidence import EvidenceCollector
from ai_investigation.investigator import DeploymentFailureInvestigator
from ai_investigation.llm_investigator import (
    DEFAULT_PROMPT_VERSION,
    PROMPT_VERSION_V1,
    PROMPT_VERSION_V2,
    PROMPT_VERSION_V3,
)
from ai_investigation.tools import JsonDeploymentTool, JsonLogTool, JsonServiceHealthTool


def _dependencies(path: Path):
    tools = (
        JsonDeploymentTool(path / "deployments.json"),
        JsonLogTool(path / "logs.json"),
        JsonServiceHealthTool(path / "service_health.json"),
    )
    return EvidenceCollector(*tools), DeploymentFailureInvestigator(*tools)


def test_holdout_loads_with_unique_balanced_scenarios(fixture_directory: Path) -> None:
    scenarios = load_scenarios(
        fixture_directory / "evaluation_scenarios_holdout_m11.json"
    )

    assert len(scenarios) == 20
    assert len({scenario.id for scenario in scenarios}) == 20
    assert sum(scenario.expected_diagnosis_id is not None for scenario in scenarios) == 10
    assert sum(scenario.expected_should_abstain is True for scenario in scenarios) == 10


def test_holdout_parses_every_robustness_category(fixture_directory: Path) -> None:
    scenarios = load_scenarios(
        fixture_directory / "evaluation_scenarios_holdout_m11.json"
    )
    represented = {
        category for scenario in scenarios for category in scenario.robustness_categories
    }

    assert represented == ROBUSTNESS_CATEGORIES
    assert scenarios[1].robustness_categories == (
        "distractor",
        "evidence_reordering",
        "supported_generalization",
    )


def test_existing_scenario_sets_remain_separate_and_compatible(
    fixture_directory: Path,
) -> None:
    original = load_scenarios(fixture_directory / "evaluation_scenarios.json")
    milestone_10 = load_scenarios(
        fixture_directory / "evaluation_scenarios_m10.json"
    )
    holdout = load_scenarios(
        fixture_directory / "evaluation_scenarios_holdout_m11.json"
    )

    assert len(original) == 11
    assert len(milestone_10) == 16
    assert milestone_10[:11] == original
    assert {scenario.id for scenario in milestone_10}.isdisjoint(
        scenario.id for scenario in holdout
    )
    assert all(not scenario.robustness_categories for scenario in original)


def test_prompt_versions_remain_frozen() -> None:
    assert DEFAULT_PROMPT_VERSION == "v1"
    assert (PROMPT_VERSION_V1, PROMPT_VERSION_V2, PROMPT_VERSION_V3) == (
        "llm-investigator-v1",
        "llm-investigator-v2",
        "llm-investigator-v3",
    )


def test_error_categories_are_derived_from_structured_outcomes() -> None:
    common = {
        "expected_diagnosis_id": None,
        "expected_abstention": False,
        "actual_diagnosis_id": None,
        "actual_abstention": None,
        "semantic_status": "not_evaluated",
        "error": None,
    }

    assert _error_category(execution_status="provider_failure", **common) == "provider_failure"
    assert _error_category(execution_status="invalid_response", **common) == (
        "invalid_structured_response"
    )
    assert _error_category(
        execution_status="invalid_references",
        **{**common, "error": "Diagnosis references are missing required sources: logs."},
    ) == "missing_required_source"
    assert _error_category(
        execution_status="invalid_references",
        **{**common, "error": "Evidence references do not exist: (99,)."},
    ) == "invalid_evidence_reference"
    assert _error_category(
        execution_status="completed",
        expected_diagnosis_id=None,
        expected_abstention=True,
        actual_diagnosis_id="health_check_timeout",
        actual_abstention=False,
        semantic_status="incorrect",
        error=None,
    ) == "false_diagnosis"
    assert _error_category(
        execution_status="completed",
        expected_diagnosis_id="health_check_timeout",
        expected_abstention=False,
        actual_diagnosis_id=None,
        actual_abstention=True,
        semantic_status="incorrect",
        error=None,
    ) == "unnecessary_abstention"
    assert _error_category(
        execution_status="completed",
        expected_diagnosis_id="health_check_timeout",
        expected_abstention=False,
        actual_diagnosis_id="missing_environment_variable",
        actual_abstention=False,
        semantic_status="incorrect",
        error=None,
    ) == "wrong_diagnosis"
    assert _error_category(execution_status="not_evaluated", **common) == "not_evaluated"


def test_error_category_summary_is_supplementary(fixture_directory: Path) -> None:
    scenarios = load_scenarios(
        fixture_directory / "evaluation_scenarios_holdout_m11.json"
    )
    collector, investigator = _dependencies(fixture_directory)

    report = run_experiment(scenarios, collector, investigator)

    assert report.aggregate.error_categories == (("unnecessary_abstention", 4),)
    assert "Error categories:\n- unnecessary_abstention: 4" in render_text_report(report)


def test_saved_experiment_identifies_holdout_source(
    fixture_directory: Path,
    tmp_path: Path,
) -> None:
    experiment_root = tmp_path / "runs"

    assert evaluate_cli.main(
        [
            "--investigator",
            "deterministic",
            "--scenarios",
            str(fixture_directory / "evaluation_scenarios_holdout_m11.json"),
            "--fixtures",
            str(fixture_directory),
            "--save-experiment",
            "--experiment-dir",
            str(experiment_root),
        ]
    ) == 0

    record = load_experiment(next(experiment_root.iterdir()))
    assert record.metadata.scenario_source == (
        "tests/fixtures/evaluation_scenarios_holdout_m11.json"
    )
    assert record.metadata.scenario_count == 20
