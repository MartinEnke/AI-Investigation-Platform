import json
from pathlib import Path

from ai_investigation.diagnose_uncertainty_prompt import (
    VARIANTS,
    build_variant_prompt,
    main,
    parse_variant_result,
    render_diagnostic,
    run_diagnostic,
)
from ai_investigation.evaluation.loader import load_scenarios
from ai_investigation.evidence import EvidenceCollector
from ai_investigation.investigator import request_from_question
from ai_investigation.tools import JsonDeploymentTool, JsonLogTool, JsonServiceHealthTool


class FakeDiagnosticModel:
    def __init__(self, responses: list[str]) -> None:
        self.responses = responses
        self.calls: list[tuple[str, bool]] = []

    def generate(self, prompt: str, *, json_mode: bool) -> str:
        response = self.responses[len(self.calls)]
        self.calls.append((prompt, json_mode))
        return response


def _collector(path: Path) -> EvidenceCollector:
    return EvidenceCollector(
        JsonDeploymentTool(path / "deployments.json"),
        JsonLogTool(path / "logs.json"),
        JsonServiceHealthTool(path / "service_health.json"),
    )


def _collected(path: Path):
    return _collector(path).collect(
        request_from_question("Why did deployment deploy-1047 fail?")
    )


def _plain_a() -> str:
    return """SUPPORTED CANDIDATES
missing_environment_variable — E1, E2
REJECTED HYPOTHESES
database_migration_failure — E2
EXPLANATION
Only one diagnosis has direct support."""


def _plain_b() -> str:
    return """health_check_timeout: NOT RELEVANT — E1
missing_environment_variable: SUPPORTED — E1 E2
database_migration_failure: REJECTED — E2
missing_database_configuration: NOT RELEVANT — E1
database_contention_blocked_migration: NOT RELEVANT — E2"""


def _minimal() -> str:
    return json.dumps(
        {
            "supported_candidates": [
                {
                    "diagnosis_id": "missing_environment_variable",
                    "evidence_references": ["E1", "E2"],
                    "reason": "Direct evidence.",
                }
            ],
            "rejected_hypotheses": [
                {
                    "diagnosis_id": "database_migration_failure",
                    "evidence_references": ["E2"],
                    "reason": "No migration evidence.",
                }
            ],
        }
    )


def _production() -> str:
    return json.dumps(
        {
            "supported_candidates": [
                {
                    "diagnosis_id": "missing_environment_variable",
                    "supporting_evidence_references": ["E1", "E2"],
                    "contradicting_evidence_references": [],
                    "evidence_strength": "strong",
                    "confidence": 0.8,
                }
            ],
            "rejected_hypotheses": [
                {
                    "diagnosis_id": "database_migration_failure",
                    "reason_code": "no_direct_support",
                    "reason_summary": "No migration evidence.",
                    "relevant_evidence_references": ["E2"],
                    "confidence": 0.6,
                }
            ],
            "unsupported_signals_present": False,
            "unsupported_signals_material": False,
            "conflicting_supported_candidates": False,
            "insufficient_evidence": False,
            "reasoning_summary": "One supported cause.",
        }
    )


def test_scenario_loading_and_unknown_scenario_validation(
    fixture_directory: Path,
) -> None:
    scenarios = load_scenarios(
        fixture_directory / "evaluation_scenarios_m10.json"
    )
    model = FakeDiagnosticModel([])

    assert any(item.id == "supported-missing-environment-variable" for item in scenarios)
    try:
        run_diagnostic(scenarios, "does-not-exist", _collector(fixture_directory), model)
    except ValueError as error:
        assert "Unknown scenario ID" in str(error)
    else:
        raise AssertionError("Unknown scenario should fail before model execution.")
    assert model.calls == []


def test_all_variants_use_same_explicit_evidence_serialization(
    fixture_directory: Path,
) -> None:
    collected = _collected(fixture_directory)
    prompts = tuple(build_variant_prompt(variant, collected) for variant in VARIANTS)

    assert all('"id":"E1"' in prompt for prompt in prompts)
    assert all('"id":"E2"' in prompt for prompt in prompts)
    assert "SUPPORTED CANDIDATES" in prompts[0]
    assert "NOT RELEVANT" in prompts[1]
    assert "minimal JSON" in prompts[2]
    assert "llm-investigator-v4-uncertainty-candidate-semantics" in prompts[3]


def test_minimal_json_parsing_and_invalid_json_handling() -> None:
    parsed = parse_variant_result("scenario", "c", _minimal())
    invalid = parse_variant_result("scenario", "c", "not json")

    assert parsed.supported_diagnoses == ("missing_environment_variable",)
    assert parsed.rejected_diagnoses == ("database_migration_failure",)
    assert parsed.parse_status == "parsed"
    assert invalid.parse_status.startswith("parse_error:")


def test_fake_provider_runs_all_variants_without_policy_or_experiment_storage(
    fixture_directory: Path,
    tmp_path: Path,
    monkeypatch,
) -> None:
    import ai_investigation.uncertainty_investigator as uncertainty

    monkeypatch.setattr(
        uncertainty,
        "decide_investigation",
        lambda assessment: (_ for _ in ()).throw(AssertionError("policy invoked")),
    )
    monkeypatch.chdir(tmp_path)
    scenarios = load_scenarios(
        fixture_directory / "evaluation_scenarios_m10.json"
    )
    model = FakeDiagnosticModel([_plain_a(), _plain_b(), _minimal(), _production()])

    results = run_diagnostic(
        scenarios,
        "supported-missing-environment-variable",
        _collector(fixture_directory),
        model,
    )

    assert [item.variant for item in results] == list(VARIANTS)
    assert [json_mode for _, json_mode in model.calls] == [False, False, True, True]
    assert all(item.parse_status == "parsed" for item in results)
    assert not (tmp_path / "experiments").exists()


def test_output_contains_raw_parsed_and_compact_comparison() -> None:
    results = tuple(
        parse_variant_result("scenario", variant, response)
        for variant, response in zip(
            VARIANTS,
            (_plain_a(), _plain_b(), _minimal(), _production()),
            strict=True,
        )
    )

    text = render_diagnostic(results)

    assert "Scenario: scenario" in text
    assert "Prompt identifier: plain-classification" in text
    assert "Raw response:" in text
    assert "Parsed supported diagnoses: missing_environment_variable" in text
    assert "Parse status: parsed" in text
    assert "Variant B: supported=1, rejected=1, not-relevant=3" in text


def test_cli_variant_selection_uses_fake_provider(
    fixture_directory: Path,
    capsys,
) -> None:
    model = FakeDiagnosticModel([_minimal()])

    exit_code = main(
        [
            "--scenarios",
            str(fixture_directory / "evaluation_scenarios_m10.json"),
            "--fixtures",
            str(fixture_directory),
            "--scenario-id",
            "supported-missing-environment-variable",
            "--variant",
            "c",
        ],
        model=model,
    )

    assert exit_code == 0
    assert len(model.calls) == 1
    assert model.calls[0][1] is True
    assert "Variant: C" in capsys.readouterr().out
