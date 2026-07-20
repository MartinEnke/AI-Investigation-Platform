import json
from dataclasses import replace
from pathlib import Path

import pytest

import ai_investigation.evaluate as evaluate_cli
import ai_investigation.evaluation.framework as framework
from ai_investigation.evaluation.framework import (
    render_text_report,
    report_to_json,
    run_experiment,
)
from ai_investigation.evaluation.loader import load_scenarios
from ai_investigation.evidence import EvidenceCollector
from ai_investigation.investigator import DeploymentFailureInvestigator
from ai_investigation.llm_investigator import LLMInvestigator, ModelProviderError
from ai_investigation.tools import JsonDeploymentTool, JsonLogTool, JsonServiceHealthTool


class FakeModel:
    def __init__(self, response: str = "", error: Exception | None = None) -> None:
        self.response = response
        self.error = error
        self.calls = 0

    def generate(self, prompt: str) -> str:
        self.calls += 1
        if self.error is not None:
            raise self.error
        return self.response


class CountingCollector:
    def __init__(self, collector: EvidenceCollector) -> None:
        self.collector = collector
        self.calls = 0

    def collect(self, request):
        self.calls += 1
        return self.collector.collect(request)


def dependencies(
    fixture_directory: Path,
) -> tuple[EvidenceCollector, DeploymentFailureInvestigator]:
    tools = (
        JsonDeploymentTool(fixture_directory / "deployments.json"),
        JsonLogTool(fixture_directory / "logs.json"),
        JsonServiceHealthTool(fixture_directory / "service_health.json"),
    )
    return EvidenceCollector(*tools), DeploymentFailureInvestigator(*tools)


def scenarios(fixture_directory: Path):
    return load_scenarios(fixture_directory / "evaluation_scenarios.json")


def response(
    diagnosis_id: str = "health_check_timeout",
    references: list[int] | None = None,
    confidence: float = 0.9,
) -> str:
    return json.dumps(
        {
            "outcome": "diagnosis",
            "diagnosis_id": diagnosis_id,
            "confidence": confidence,
            "evidence_references": references or [1, 2, 3],
            "abstention_reason": None,
        }
    )


def test_deterministic_evaluation_and_aggregate_denominators(
    fixture_directory: Path,
) -> None:
    collector, investigator = dependencies(fixture_directory)

    report = run_experiment(scenarios(fixture_directory), collector, investigator)

    assert len(report.scenarios) == 11
    assert all(item.semantic_correctness_status == "correct" for item in report.scenarios)
    assert report.aggregate.correct_diagnoses == 5
    assert report.aggregate.diagnosis_cases == 5
    assert report.aggregate.correct_abstentions == 6
    assert report.aggregate.abstention_cases == 6
    assert report.aggregate.structured_responses_assessed == 0
    assert report.aggregate.provider_failures == 0


def test_unknown_deployment_is_a_correct_abstention(fixture_directory: Path) -> None:
    collector, investigator = dependencies(fixture_directory)

    result = run_experiment(
        (scenarios(fixture_directory)[1],), collector, investigator
    ).scenarios[0]

    assert result.expected_abstention
    assert result.actual_abstention
    assert result.abstention_correct
    assert result.semantic_correctness_status == "correct"


def test_valid_but_incorrect_diagnosis_is_a_semantic_failure(
    fixture_directory: Path,
) -> None:
    collector, investigator = dependencies(fixture_directory)
    model = FakeModel(response("missing_environment_variable", [1, 2]))

    report = run_experiment(
        (scenarios(fixture_directory)[0],),
        collector,
        investigator,
        investigator_mode="gemini",
        structured_model=model,
    )
    result = report.scenarios[0]

    assert result.execution_status == "completed"
    assert result.structured_response_valid
    assert result.evidence_references_valid
    assert result.diagnosis_correct is False
    assert result.semantic_correctness_status == "incorrect"
    assert report.aggregate.average_confidence_incorrect == 0.9


@pytest.mark.parametrize(
    ("model", "status", "structured_valid", "references_valid"),
    (
        (FakeModel(response(references=[1, 2])), "invalid_references", True, False),
        (FakeModel("not json"), "invalid_response", False, None),
        (
            FakeModel(error=ModelProviderError("Provider unavailable.")),
            "provider_failure",
            None,
            None,
        ),
    ),
)
def test_model_failures_keep_validation_dimensions_separate(
    fixture_directory: Path,
    model: FakeModel,
    status: str,
    structured_valid: bool | None,
    references_valid: bool | None,
) -> None:
    collector, investigator = dependencies(fixture_directory)

    scenario = replace(
        scenarios(fixture_directory)[0],
        expected_execution_status=status,
    )
    result = run_experiment(
        (scenario,),
        collector,
        investigator,
        investigator_mode="gemini",
        structured_model=model,
    ).scenarios[0]

    assert model.calls == 1
    assert result.execution_status == status
    assert result.execution_status_matches is True
    assert result.structured_response_valid is structured_valid
    assert result.evidence_references_valid is references_valid
    assert result.semantic_correctness_status == "not_evaluated"


def test_agreement_is_recorded_separately_from_correctness(
    fixture_directory: Path,
) -> None:
    collector, investigator = dependencies(fixture_directory)
    scenario = replace(
        scenarios(fixture_directory)[0],
        expected_diagnosis_id="missing_environment_variable",
    )

    report = run_experiment(
        (scenario,),
        collector,
        investigator,
        investigator_mode="both",
        structured_model=FakeModel(response()),
    )

    assert all(item.deterministic_model_agreement for item in report.scenarios)
    assert all(item.semantic_correctness_status == "incorrect" for item in report.scenarios)
    assert report.aggregate.investigator_agreements == 1
    assert report.aggregate.agreement_cases == 1


def test_latency_is_measured_with_a_monotonic_clock(fixture_directory: Path) -> None:
    collector, investigator = dependencies(fixture_directory)
    times = iter((10.0, 10.0, 10.0, 10.001, 10.002, 10.007, 10.008, 10.009, 10.010, 10.011))

    result = run_experiment(
        (scenarios(fixture_directory)[0],),
        collector,
        investigator,
        clock=lambda: next(times),
    ).scenarios[0]

    assert result.latency_ms == pytest.approx(5.0)


def test_json_report_is_stable_and_contains_scenarios(fixture_directory: Path) -> None:
    collector, investigator = dependencies(fixture_directory)
    report = run_experiment((scenarios(fixture_directory)[0],), collector, investigator)

    rendered = report_to_json(report)
    parsed = json.loads(rendered)

    assert rendered == report_to_json(report)
    assert parsed["investigator_mode"] == "deterministic"
    assert parsed["aggregate"]["total_scenarios"] == 1
    assert parsed["scenarios"][0]["scenario_id"] == "supported-health-check-timeout"


def test_text_report_omits_inapplicable_scenario_fields(
    fixture_directory: Path,
) -> None:
    collector, investigator = dependencies(fixture_directory)
    report = run_experiment((scenarios(fixture_directory)[0],), collector, investigator)

    rendered = render_text_report(report)

    assert "Expected execution status:" not in rendered
    assert "Structured response:" not in rendered
    assert "Missing sources:" not in rendered
    assert "Unexpected sources:" not in rendered
    assert "Abstention assessment:" not in rendered
    assert "not specified" not in rendered
    assert "not applicable" not in rendered
    assert "not available" not in rendered


def test_text_report_describes_abstention_as_an_outcome(
    fixture_directory: Path,
) -> None:
    collector, investigator = dependencies(fixture_directory)
    report = run_experiment((scenarios(fixture_directory)[1],), collector, investigator)

    rendered = render_text_report(report)

    assert "Expected outcome: abstention" in rendered
    assert "Actual outcome: abstention" in rendered
    assert "Abstention assessment: correct" in rendered
    assert "Expected diagnosis: abstain" not in rendered


def test_text_report_deduplicates_source_categories_only(
    fixture_directory: Path,
) -> None:
    collector, investigator = dependencies(fixture_directory)
    report = run_experiment((scenarios(fixture_directory)[8],), collector, investigator)

    rendered = render_text_report(report)

    assert "Referenced sources: deployment, logs" in rendered
    assert "Referenced sources: deployment, logs, logs" not in rendered
    assert report.scenarios[0].referenced_sources == ("deployment", "logs", "logs")


def test_evaluation_cli_writes_json_and_does_not_construct_gemini(
    fixture_directory: Path,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    def fail_if_constructed():
        raise AssertionError("Gemini must not be constructed.")

    monkeypatch.setattr(evaluate_cli, "_gemini_model", fail_if_constructed)
    output = tmp_path / "reports" / "baseline.json"

    exit_code = evaluate_cli.main(
        [
            "--investigator",
            "deterministic",
            "--format",
            "json",
            "--output",
            str(output),
            "--scenarios",
            str(fixture_directory / "evaluation_scenarios.json"),
            "--fixtures",
            str(fixture_directory),
        ]
    )

    assert exit_code == 0
    assert output.exists()
    assert json.loads(output.read_text())["aggregate"]["total_scenarios"] == 11
    assert capsys.readouterr().out == output.read_text()


def test_both_mode_collects_once_and_shares_the_collected_instance(
    fixture_directory: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    collector, investigator = dependencies(fixture_directory)
    counting = CountingCollector(collector)
    seen: list[object] = []
    real_investigate_evidence = investigator.investigate_evidence

    def deterministic_spy(collected):
        seen.append(collected)
        return real_investigate_evidence(collected)

    investigator.investigate_evidence = deterministic_spy  # type: ignore[method-assign]
    real_llm = LLMInvestigator(FakeModel(response()))

    class RecordingLLMInvestigator:
        def __init__(self, model: object) -> None:
            pass

        def investigate(self, collected):
            seen.append(collected)
            return real_llm.investigate(collected)

    monkeypatch.setattr(framework, "LLMInvestigator", RecordingLLMInvestigator)

    run_experiment(
        (scenarios(fixture_directory)[0],),
        counting,
        investigator,
        investigator_mode="both",
        structured_model=FakeModel(response()),
    )

    assert counting.calls == 1
    assert len(seen) == 2
    assert seen[0] is seen[1]


def test_llm_pacing_occurs_only_between_provider_requests(
    fixture_directory: Path,
) -> None:
    collector, investigator = dependencies(fixture_directory)
    selected = scenarios(fixture_directory)
    events: list[str] = []

    class SequencedModel:
        provider_name = "fake"
        model_name = "sequenced"

        def __init__(self) -> None:
            self.responses = iter(
                (
                    response(),
                    response("missing_environment_variable", [1, 2]),
                )
            )

        def generate(self, prompt: str) -> str:
            events.append("request")
            return next(self.responses)

    run_experiment(
        (selected[0], selected[1], selected[5]),
        collector,
        investigator,
        investigator_mode="llm",
        structured_model=SequencedModel(),
        request_delay_seconds=2.5,
        sleep=lambda seconds: events.append(f"sleep:{seconds}"),
    )

    assert events == ["request", "sleep:2.5", "request"]


def test_zero_delay_and_deterministic_runs_never_sleep(
    fixture_directory: Path,
) -> None:
    collector, investigator = dependencies(fixture_directory)
    selected = scenarios(fixture_directory)
    sleeps: list[float] = []

    run_experiment(
        (selected[0], selected[5]),
        collector,
        investigator,
        investigator_mode="llm",
        structured_model=FakeModel(response()),
        sleep=sleeps.append,
    )
    run_experiment(
        (selected[0], selected[5]),
        collector,
        investigator,
        investigator_mode="deterministic",
        request_delay_seconds=4,
        sleep=sleeps.append,
    )

    assert sleeps == []
