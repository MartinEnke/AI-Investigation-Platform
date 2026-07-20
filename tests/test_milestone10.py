import json
import os
from pathlib import Path

import pytest

import ai_investigation.evaluate as evaluate_cli
from ai_investigation.evaluation.framework import run_experiment
from ai_investigation.evaluation.loader import load_scenarios
from ai_investigation.evidence import EvidenceCollector
from ai_investigation.investigator import DeploymentFailureInvestigator, request_from_question
from ai_investigation.investigators import (
    DeterministicInvestigatorAdapter,
    Investigator,
    LLMInvestigatorAdapter,
)
from ai_investigation.llm_investigator import (
    ModelProviderError,
    PROMPT_VERSION,
    serialize_evidence,
)
from ai_investigation.tools import JsonDeploymentTool, JsonLogTool, JsonServiceHealthTool


class FakeProvider:
    provider_name = "fake"
    model_name = "fixed-response"

    def __init__(self, responses: list[str | Exception]) -> None:
        self.responses = responses
        self.calls = 0

    def generate(self, prompt: str) -> str:
        response = self.responses[self.calls]
        self.calls += 1
        if isinstance(response, Exception):
            raise response
        return response


def _dependencies(path: Path):
    tools = (
        JsonDeploymentTool(path / "deployments.json"),
        JsonLogTool(path / "logs.json"),
        JsonServiceHealthTool(path / "service_health.json"),
    )
    return EvidenceCollector(*tools), DeploymentFailureInvestigator(*tools)


def _decision(
    diagnosis_id: str = "health_check_timeout",
    references: list[int] | None = None,
) -> str:
    return json.dumps(
        {
            "outcome": "diagnosis",
            "diagnosis_id": diagnosis_id,
            "confidence": 0.8,
            "evidence_references": references or [1, 2, 3],
            "abstention_reason": None,
        }
    )


def _abstention(reason: str, references: list[int]) -> str:
    return json.dumps(
        {
            "outcome": "abstain",
            "diagnosis_id": None,
            "confidence": 0.4,
            "evidence_references": references,
            "abstention_reason": reason,
        }
    )


def test_common_contract_and_deterministic_adapter_preserve_result(
    fixture_directory: Path,
) -> None:
    collector, deterministic = _dependencies(fixture_directory)
    collected = collector.collect(request_from_question("Why did deploy-1042 fail?"))
    adapter: Investigator = DeterministicInvestigatorAdapter(deterministic)

    execution = adapter.investigate(collected)

    assert adapter.identity.mode == "deterministic"
    assert execution.result == deterministic.investigate_evidence(collected)
    assert execution.diagnosis_id == "health_check_timeout"


def test_llm_identity_and_prompt_version_are_stable(fixture_directory: Path) -> None:
    adapter = LLMInvestigatorAdapter(FakeProvider([_decision()]))

    assert adapter.identity.mode == "llm"
    assert adapter.identity.provider == "fake"
    assert adapter.identity.model == "fixed-response"
    assert adapter.identity.prompt_version == PROMPT_VERSION


def test_evidence_serialization_is_stable_and_excludes_expectations(
    fixture_directory: Path,
) -> None:
    collector, _ = _dependencies(fixture_directory)
    collected = collector.collect(request_from_question("Why did deploy-1042 fail?"))

    first = serialize_evidence(collected)

    assert first == serialize_evidence(collected)
    assert [item["reference"] for item in first["evidence"]] == [1, 2, 3]
    assert "expected" not in json.dumps(first).lower()


def test_extended_scenario_set_reuses_baseline_and_has_unique_ids(
    fixture_directory: Path,
) -> None:
    baseline = load_scenarios(fixture_directory / "evaluation_scenarios.json")
    extended = load_scenarios(fixture_directory / "evaluation_scenarios_m10.json")

    assert extended[: len(baseline)] == baseline
    assert len(extended) == 16
    assert len({scenario.id for scenario in extended}) == 16


def test_fake_provider_can_evaluate_all_extended_cases(fixture_directory: Path) -> None:
    collector, deterministic = _dependencies(fixture_directory)
    scenarios = load_scenarios(fixture_directory / "evaluation_scenarios_m10.json")[-5:]
    provider = FakeProvider(
        [
            _decision("missing_database_configuration", [1, 2]),
            _decision("database_contention_blocked_migration", [1, 2, 3, 4, 5, 6]),
            _abstention("insufficient_evidence", [1, 2, 3]),
            _abstention("conflicting_evidence", [1, 2, 3]),
            _decision("missing_environment_variable", [1, 3]),
        ]
    )

    report = run_experiment(
        scenarios,
        collector,
        deterministic,
        investigator_mode="llm",
        structured_model=provider,
    )

    assert provider.calls == 5
    assert all(
        result.semantic_correctness_status == "correct" for result in report.scenarios
    )
    assert report.scenarios[-1].referenced_sources == ("deployment", "logs")


def test_llm_runner_continues_after_provider_failure(fixture_directory: Path) -> None:
    collector, deterministic = _dependencies(fixture_directory)
    loaded = load_scenarios(fixture_directory / "evaluation_scenarios.json")
    scenarios = (loaded[0], loaded[5])
    provider = FakeProvider(
        [
            ModelProviderError("temporary provider failure"),
            _decision("missing_environment_variable", [1, 2]),
        ]
    )

    report = run_experiment(
        scenarios,
        collector,
        deterministic,
        investigator_mode="llm",
        structured_model=provider,
    )

    assert [result.execution_status for result in report.scenarios] == [
        "provider_failure",
        "completed",
    ]
    assert provider.calls == 2


def test_evaluation_cli_selects_llm_with_an_injected_provider(
    fixture_directory: Path,
    tmp_path: Path,
) -> None:
    source = json.loads(
        (fixture_directory / "evaluation_scenarios.json").read_text(encoding="utf-8")
    )
    scenario_path = tmp_path / "scenario.json"
    scenario_path.write_text(json.dumps(source[:1]), encoding="utf-8")

    exit_code = evaluate_cli.main(
        [
            "--investigator",
            "llm",
            "--scenarios",
            str(scenario_path),
            "--fixtures",
            str(fixture_directory),
        ],
        structured_model=FakeProvider([_decision()]),
    )

    assert exit_code == 0


def test_evaluation_cli_rejects_invalid_investigator() -> None:
    with pytest.raises(SystemExit) as error:
        evaluate_cli.main(["--investigator", "unknown"])

    assert error.value.code == 2


@pytest.mark.parametrize(
    ("extra_arguments", "environment_provider", "expected_model"),
    (
        ([], "groq", "environment-model"),
        (["--model", "cli-model"], "groq", "cli-model"),
        (["--provider", "groq"], "unsupported-provider", "environment-model"),
    ),
)
def test_evaluation_cli_loads_dotenv_defaults_with_cli_override(
    fixture_directory: Path,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    extra_arguments: list[str],
    environment_provider: str,
    expected_model: str,
) -> None:
    source = json.loads(
        (fixture_directory / "evaluation_scenarios.json").read_text(encoding="utf-8")
    )
    scenario_path = tmp_path / "scenario.json"
    scenario_path.write_text(json.dumps(source[:1]), encoding="utf-8")
    (tmp_path / ".env").write_text(
        "GROQ_API_KEY=dotenv-secret\n"
        f"AI_INVESTIGATION_PROVIDER={environment_provider}\n"
        "AI_INVESTIGATION_MODEL=environment-model\n",
        encoding="utf-8",
    )
    monkeypatch.chdir(tmp_path)
    for name in (
        "GROQ_API_KEY",
        "AI_INVESTIGATION_PROVIDER",
        "AI_INVESTIGATION_MODEL",
    ):
        monkeypatch.delenv(name, raising=False)
    selected: list[tuple[str | None, str | None, str | None]] = []

    def fake_groq_model(model_name=None, provider_name=None):
        selected.append((model_name, provider_name, os.environ.get("GROQ_API_KEY")))
        return FakeProvider([_decision()])

    monkeypatch.setattr(evaluate_cli, "_groq_model", fake_groq_model)

    exit_code = evaluate_cli.main(
        [
            "--investigator",
            "llm",
            "--scenarios",
            str(scenario_path),
            "--fixtures",
            str(fixture_directory),
            *extra_arguments,
        ]
    )

    assert exit_code == 0
    assert selected == [(expected_model, "groq", "dotenv-secret")]
