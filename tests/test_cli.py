import json
from pathlib import Path

import pytest

import ai_investigation.cli as cli
from ai_investigation.cli import build_dependencies, main, run_investigation
from ai_investigation.llm_investigator import ModelProviderError


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
    def __init__(self, collector: object) -> None:
        self.collector = collector
        self.calls = 0

    def collect(self, request: object):
        self.calls += 1
        return self.collector.collect(request)


def dependencies(fixture_directory: Path):
    return build_dependencies(fixture_directory)


def llm_response() -> str:
    return json.dumps(
        {
            "outcome": "diagnosis",
            "diagnosis_id": "health_check_timeout",
            "confidence": 0.82,
            "evidence_references": [1, 2, 3],
            "abstention_reason": None,
        }
    )


def test_deterministic_investigation_displays_collected_evidence_and_result(
    fixture_directory: Path,
) -> None:
    collector, investigator = dependencies(fixture_directory)

    output = run_investigation(
        "Why did deployment deploy-1042 fail?",
        "deterministic",
        collector,
        investigator,
    )

    assert "Collected Evidence\n------------------" in output
    assert "1. [deployment] deploy-1042 has status failed and failed stage health_check." in output
    assert "2. [logs] Health check timed out after 30 seconds." in output
    assert "3. [service_health] checkout-api was unhealthy:" in output
    assert "Diagnosis: The deployment health check timed out" in output
    assert "Decision outcome: single_match" in output
    assert "Deterministic Conclusion\n------------------------" in output
    assert "Deterministic Evidence References\n---------------------------------\n1, 2, 3" in output


def test_llm_investigation_displays_references_and_validation(
    fixture_directory: Path,
) -> None:
    collector, investigator = dependencies(fixture_directory)
    model = FakeModel(llm_response())

    output = run_investigation(
        "Why did deployment deploy-1042 fail?",
        "gemini",
        collector,
        investigator,
        structured_model=model,
    )

    assert model.calls == 1
    assert "Investigator: gemini" in output
    assert "Confidence: 0.82" in output
    assert "Abstained: no" in output
    assert "Model Interpretation\n--------------------" in output
    assert "Model Evidence References\n-------------------------\n1, 2, 3" in output
    assert "Structured response: valid" in output
    assert "Evidence references: valid" in output
    assert "Semantic correctness: not evaluated" in output
    assert "model confidence is uncalibrated and should not be treated as correctness" in output


def test_both_mode_collects_once_and_renders_both_paths(
    fixture_directory: Path,
) -> None:
    collector, investigator = dependencies(fixture_directory)
    counting_collector = CountingCollector(collector)
    model = FakeModel(llm_response())

    output = run_investigation(
        "Why did deployment deploy-1042 fail?",
        "both",
        counting_collector,
        investigator,
        structured_model=model,
    )

    assert counting_collector.calls == 1
    assert model.calls == 1
    assert output.count("Collected Evidence\n------------------") == 1
    assert "Deterministic Conclusion\n------------------------" in output
    assert "Model Interpretation\n--------------------" in output


def test_unknown_deployment_in_both_mode_does_not_call_model(
    fixture_directory: Path,
) -> None:
    collector, investigator = dependencies(fixture_directory)
    model = FakeModel(llm_response())

    output = run_investigation(
        "Why did deployment deploy-9999 fail?",
        "both",
        collector,
        investigator,
        structured_model=model,
    )

    assert model.calls == 0
    assert "Answer: Deployment deploy-9999 was not found." in output
    assert "Execution status: not_evaluated" in output
    assert "Model evaluation was not attempted." in output


def test_unknown_deployment_does_not_construct_gemini(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    def fail_if_constructed():
        raise AssertionError("Gemini must not be constructed.")

    monkeypatch.setattr(cli, "_gemini_model", fail_if_constructed)

    exit_code = main(
        ["--investigator", "both", "Why did deployment deploy-9999 fail?"]
    )

    assert exit_code == 0
    assert "Execution status: not_evaluated" in capsys.readouterr().out


def test_unknown_deployment_is_rendered_without_error(
    fixture_directory: Path,
) -> None:
    collector, investigator = dependencies(fixture_directory)

    output = run_investigation(
        "Why did deployment deploy-9999 fail?",
        "deterministic",
        collector,
        investigator,
    )

    assert "No evidence was collected." in output
    assert "Answer: Deployment deploy-9999 was not found." in output
    assert "No deployment record is available in the local fixtures." in output


def test_provider_failure_is_visible_and_not_retried(fixture_directory: Path) -> None:
    collector, investigator = dependencies(fixture_directory)
    model = FakeModel(error=ModelProviderError("Gemini is unavailable."))

    output = run_investigation(
        "Why did deployment deploy-1042 fail?",
        "gemini",
        collector,
        investigator,
        structured_model=model,
    )

    assert model.calls == 1
    assert "Execution status: provider_failure" in output
    assert "Error\n-----\nGemini is unavailable." in output


def test_invalid_references_are_reported(fixture_directory: Path) -> None:
    collector, investigator = dependencies(fixture_directory)
    response = json.loads(llm_response())
    response["evidence_references"] = [1, 99]

    output = run_investigation(
        "Why did deployment deploy-1042 fail?",
        "gemini",
        collector,
        investigator,
        structured_model=FakeModel(json.dumps(response)),
    )

    assert "Execution status: invalid_references" in output
    assert "Structured response: valid" in output
    assert "Evidence references: invalid" in output
    assert "Evidence references do not exist: (99,)." in output


def test_invalid_investigator_is_rejected(fixture_directory: Path) -> None:
    collector, investigator = dependencies(fixture_directory)

    with pytest.raises(ValueError, match="Unknown investigator: agent"):
        run_investigation("Why did deploy-1042 fail?", "agent", collector, investigator)

    with pytest.raises(SystemExit) as error:
        main(["--investigator", "agent", "Why did deploy-1042 fail?"])
    assert error.value.code == 2
