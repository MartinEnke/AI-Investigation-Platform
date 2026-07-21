import json
from pathlib import Path

from ai_investigation import evaluate as evaluate_cli
from ai_investigation.diagnose_uncertainty_prompt import (
    VARIANT_IDENTIFIERS,
    build_variant_prompt,
    parse_variant_result,
)
from ai_investigation.decision_policy import DecisionOutcome, DecisionReason
from ai_investigation.evaluation.framework import (
    render_text_report,
    report_to_json,
    run_experiment,
)
from ai_investigation.evaluation.loader import load_scenarios
from ai_investigation.evaluation.tracking import load_experiment
from ai_investigation.evidence import EvidenceCollector
from ai_investigation.investigator import DeploymentFailureInvestigator, request_from_question
from ai_investigation.tools import JsonDeploymentTool, JsonLogTool, JsonServiceHealthTool
from ai_investigation.uncertainty_investigator import (
    CANDIDATE_SEMANTICS_PROMPT_SELECTION,
    CANDIDATE_SEMANTICS_PROMPT_VERSION,
    CANDIDATE_SEMANTICS_RESPONSE_SCHEMA_VERSION,
    LLMPolicyInvestigationSuccess,
    LLMPolicyInvestigator,
    UNCERTAINTY_PROMPT_SELECTION,
    build_uncertainty_prompt,
    parse_candidate_semantics_proposal,
    uncertainty_prompt_identifier,
    uncertainty_schema_identifier,
)


class FakeModel:
    provider_name = "fake"
    model_name = "parity-model"

    def __init__(self, response: str) -> None:
        self.response = response
        self.prompts: list[str] = []

    def generate(self, prompt: str) -> str:
        self.prompts.append(prompt)
        return self.response


def _dependencies(path: Path):
    tools = (
        JsonDeploymentTool(path / "deployments.json"),
        JsonLogTool(path / "logs.json"),
        JsonServiceHealthTool(path / "service_health.json"),
    )
    return EvidenceCollector(*tools), DeploymentFailureInvestigator(*tools)


def _collected(path: Path):
    collector, _ = _dependencies(path)
    return collector.collect(
        request_from_question("Why did deployment deploy-1047 fail?")
    )


def _diagnostic_shape_response() -> str:
    return json.dumps(
        {
            "conflicting_supported_candidates": False,
            "insufficient_evidence": False,
            "reasoning_summary": "A direct cause is supported.",
            "rejected_hypotheses": [
                {
                    "confidence": 0.1,
                    "diagnosis_id": "database_migration_failure",
                    "reason_code": "no_direct_support",
                    "reason_summary": "No direct evidence.",
                    "relevant_evidence_references": [],
                }
            ],
            "supported_candidates": [
                {
                    "confidence": 0.8,
                    "contradicting_evidence_references": [],
                    "diagnosis_id": "missing_environment_variable",
                    "evidence_strength": "strong",
                    "supporting_evidence_references": ["E2"],
                }
            ],
            "unsupported_signals_material": False,
            "unsupported_signals_present": False,
        },
        sort_keys=True,
    )


def test_diagnostic_and_production_resolve_same_explicit_prompt_and_schema(
    fixture_directory: Path,
) -> None:
    collected = _collected(fixture_directory)
    diagnostic_prompt = build_variant_prompt("d", collected)
    production_prompt = build_uncertainty_prompt(
        collected, CANDIDATE_SEMANTICS_PROMPT_SELECTION
    )

    assert diagnostic_prompt == production_prompt
    assert VARIANT_IDENTIFIERS["d"] == uncertainty_prompt_identifier(
        CANDIDATE_SEMANTICS_PROMPT_SELECTION
    )
    assert (
        uncertainty_prompt_identifier(CANDIDATE_SEMANTICS_PROMPT_SELECTION)
        == CANDIDATE_SEMANTICS_PROMPT_VERSION
    )
    assert (
        uncertainty_schema_identifier(CANDIDATE_SEMANTICS_PROMPT_SELECTION)
        == CANDIDATE_SEMANTICS_RESPONSE_SCHEMA_VERSION
    )
    assert uncertainty_prompt_identifier(UNCERTAINTY_PROMPT_SELECTION) != (
        CANDIDATE_SEMANTICS_PROMPT_VERSION
    )


def test_same_response_has_diagnostic_and_production_proposal_parity(
    fixture_directory: Path,
) -> None:
    raw = _diagnostic_shape_response()
    diagnostic = parse_variant_result("scenario", "d", raw)
    parsed = parse_candidate_semantics_proposal(raw)
    production = LLMPolicyInvestigator(
        FakeModel(raw), CANDIDATE_SEMANTICS_PROMPT_SELECTION
    ).investigate(_collected(fixture_directory))

    assert isinstance(production, LLMPolicyInvestigationSuccess)
    assert diagnostic.supported_diagnoses == ("missing_environment_variable",)
    assert diagnostic.rejected_diagnoses == ("database_migration_failure",)
    assert tuple(item.diagnosis_id for item in production.proposal.candidates) == (
        "missing_environment_variable",
    )
    assert tuple(
        item.diagnosis_id for item in production.proposal.rejected_hypotheses
    ) == ("database_migration_failure",)
    assert production.proposal.candidates[0].supporting_evidence_references == ("E2",)
    assert production.proposal.rejected_hypotheses[
        0
    ].relevant_evidence_references == ()
    assert production.proposal.conflicting_supported_candidates is False
    assert production.proposal.insufficient_evidence is False
    assert tuple(
        item.candidate.diagnosis for item in production.uncertainty.candidates
    ) == ("missing_environment_variable",)
    assert production.decision.outcome is DecisionOutcome.DIAGNOSIS
    assert production.decision.reason is DecisionReason.SINGLE_SUPPORTED_CANDIDATE


def test_exact_response_survives_complete_evaluator_projection(
    fixture_directory: Path,
) -> None:
    collector, deterministic = _dependencies(fixture_directory)
    scenario = next(
        item
        for item in load_scenarios(
            fixture_directory / "evaluation_scenarios_m10.json"
        )
        if item.id == "supported-missing-environment-variable"
    )

    report = run_experiment(
        (scenario,),
        collector,
        deterministic,
        investigator_mode="llm-policy",
        structured_model=FakeModel(_diagnostic_shape_response()),
        prompt_version=CANDIDATE_SEMANTICS_PROMPT_SELECTION,
        sleep=lambda seconds: None,
    )
    result = report.scenarios[0]

    assert result.candidate_diagnoses == ("missing_environment_variable",)
    assert result.rejected_hypothesis_diagnoses == ("database_migration_failure",)
    assert result.policy_outcome == "diagnosis"
    assert result.policy_reason == "single_supported_candidate"
    assert report.aggregate.single_candidate_assessments == 1
    assert report.aggregate.rejected_hypotheses_total == 1
    assert report.aggregate.assessments_with_rejected_hypotheses == 1
    assert "Rejected hypothesis diagnoses: database_migration_failure" in (
        render_text_report(report)
    )
    serialized = json.loads(report_to_json(report))
    assert serialized["scenarios"][0]["rejected_hypothesis_diagnoses"] == [
        "database_migration_failure"
    ]


def test_cli_saves_rejections_and_debug_is_strictly_opt_in(
    fixture_directory: Path,
    tmp_path: Path,
    capsys,
) -> None:
    all_scenarios = json.loads(
        (fixture_directory / "evaluation_scenarios_m10.json").read_text(encoding="utf-8")
    )
    # The M10 file includes earlier scenarios indirectly; use the base fixture entry directly.
    base_scenarios = json.loads(
        (fixture_directory / "evaluation_scenarios.json").read_text(encoding="utf-8")
    )
    scenario = next(
        item for item in base_scenarios if item["id"] == "supported-missing-environment-variable"
    )
    scenario_path = tmp_path / "scenario.json"
    scenario_path.write_text(json.dumps([scenario]), encoding="utf-8")
    experiment_root = tmp_path / "runs"

    common = [
        "--investigator",
        "llm-policy",
        "--prompt-version",
        CANDIDATE_SEMANTICS_PROMPT_SELECTION,
        "--scenarios",
        str(scenario_path),
        "--fixtures",
        str(fixture_directory),
    ]
    assert all_scenarios  # confirms the requested benchmark fixture remains readable
    assert evaluate_cli.main(
        common,
        structured_model=FakeModel(_diagnostic_shape_response()),
        sleep=lambda seconds: None,
    ) == 0
    ordinary = capsys.readouterr()
    assert "A direct cause is supported." not in ordinary.out
    assert "A direct cause is supported." not in ordinary.err
    assert "Resolved prompt: " + CANDIDATE_SEMANTICS_PROMPT_VERSION in ordinary.out

    assert evaluate_cli.main(
        [
            *common,
            "--debug-uncertainty",
            "--save-experiment",
            "--experiment-dir",
            str(experiment_root),
        ],
        structured_model=FakeModel(_diagnostic_shape_response()),
        sleep=lambda seconds: None,
    ) == 0
    debug = capsys.readouterr()
    assert "Raw provider response:" in debug.err
    assert "A direct cause is supported." in debug.err
    assert "Parsed rejected hypotheses: database_migration_failure" in debug.err
    record = load_experiment(next(experiment_root.iterdir()))
    assert dict(record.metadata.configuration)["prompt_version"] == (
        CANDIDATE_SEMANTICS_PROMPT_SELECTION
    )
    assert record.metadata.prompt_version == CANDIDATE_SEMANTICS_PROMPT_VERSION
    assert (
        record.metadata.response_schema_version
        == CANDIDATE_SEMANTICS_RESPONSE_SCHEMA_VERSION
    )
    assert record.report.scenarios[0].rejected_hypothesis_diagnoses == (
        "database_migration_failure",
    )
