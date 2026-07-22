import json
from pathlib import Path

import pytest

from ai_investigation import evaluate as evaluate_cli
from ai_investigation.decision_policy import DecisionOutcome, DecisionReason
from ai_investigation.evaluation.framework import render_text_report, run_experiment
from ai_investigation.evaluation.loader import load_scenarios
from ai_investigation.evaluation.tracking import load_experiment
from ai_investigation.evidence import EvidenceCollector
from ai_investigation.investigator import DeploymentFailureInvestigator, request_from_question
from ai_investigation.tools import JsonDeploymentTool, JsonLogTool, JsonServiceHealthTool
from ai_investigation.uncertainty_investigator import (
    CANDIDATE_SEMANTICS_PROMPT_SELECTION,
    CANDIDATE_SEMANTICS_PROMPT_VERSION,
    CANDIDATE_SEMANTICS_RESPONSE_SCHEMA_VERSION,
    LLMPolicyInvestigationFailure,
    LLMPolicyInvestigationSuccess,
    LLMPolicyInvestigator,
    RejectionReason,
    UNCERTAINTY_PROMPT_VERSION,
    UNCERTAINTY_RESPONSE_SCHEMA_VERSION,
    build_uncertainty_prompt,
    parse_candidate_semantics_proposal,
    proposal_to_uncertainty,
)


class FakeModel:
    provider_name = "fake"
    model_name = "candidate-semantics"

    def __init__(self, responses: list[str]) -> None:
        self.responses = responses
        self.calls = 0

    def generate(self, prompt: str) -> str:
        response = self.responses[self.calls]
        self.calls += 1
        return response


def _dependencies(path: Path):
    tools = (
        JsonDeploymentTool(path / "deployments.json"),
        JsonLogTool(path / "logs.json"),
        JsonServiceHealthTool(path / "service_health.json"),
    )
    return EvidenceCollector(*tools), DeploymentFailureInvestigator(*tools)


def _collected(path: Path, deployment_id: str = "deploy-1042"):
    collector, _ = _dependencies(path)
    return collector.collect(request_from_question(f"Why did {deployment_id} fail?"))


def _supported(
    diagnosis: str = "health_check_timeout",
    *,
    supporting: list[str] | None = None,
    contradicting: list[str] | None = None,
    strength: str = "strong",
    confidence: float = 0.8,
) -> dict[str, object]:
    return {
        "diagnosis_id": diagnosis,
        "supporting_evidence_references": supporting or ["E1", "E2", "E3"],
        "contradicting_evidence_references": contradicting or [],
        "evidence_strength": strength,
        "confidence": confidence,
    }


def _rejected(
    diagnosis: str = "missing_database_configuration",
    *,
    reason: str = "no_direct_support",
    references: list[str] | None = None,
    confidence: float = 0.9,
) -> dict[str, object]:
    return {
        "diagnosis_id": diagnosis,
        "reason_code": reason,
        "reason_summary": "The hypothesis was considered but lacks direct support.",
        "relevant_evidence_references": references or ["E2"],
        "confidence": confidence,
    }


def _response(
    supported: list[dict[str, object]] | None = None,
    rejected: list[dict[str, object]] | None = None,
    **overrides: object,
) -> str:
    value = {
        "supported_candidates": supported or [],
        "rejected_hypotheses": rejected or [],
        "unsupported_signals_present": False,
        "unsupported_signals_material": False,
        "conflicting_supported_candidates": len(supported or []) > 1,
        "insufficient_evidence": False,
        "reasoning_summary": "Candidate eligibility was assessed.",
    }
    value.update(overrides)
    return json.dumps(value)


def test_candidate_semantics_prompt_is_new_and_baseline_remains_identifiable(
    fixture_directory: Path,
) -> None:
    collected = _collected(fixture_directory)
    baseline = build_uncertainty_prompt(collected)
    prompt = build_uncertainty_prompt(
        collected, CANDIDATE_SEMANTICS_PROMPT_SELECTION
    )

    assert UNCERTAINTY_PROMPT_VERSION in baseline
    assert UNCERTAINTY_RESPONSE_SCHEMA_VERSION in baseline
    assert CANDIDATE_SEMANTICS_PROMPT_VERSION in prompt
    assert CANDIDATE_SEMANTICS_RESPONSE_SCHEMA_VERSION in prompt
    assert "Do not return every diagnosis you considered" in prompt
    assert "Absence of contradiction is not support" in prompt
    assert "Semantic similarity is not support" in prompt
    assert "rejected_hypotheses" in prompt
    assert "A diagnosis ID must appear in at most one collection" in prompt
    assert (
        "it may appear in supported_candidates or rejected_hypotheses, but never both"
        in prompt
    )


@pytest.mark.parametrize(
    ("supported", "rejected"),
    (
        ([], [_rejected()]),
        ([_supported()], [_rejected()]),
        (
            [
                _supported(),
                _supported("database_migration_failure", supporting=["E1", "E2"]),
            ],
            [],
        ),
    ),
)
def test_zero_one_and_multiple_supported_candidates_parse(
    supported: list[dict[str, object]],
    rejected: list[dict[str, object]],
) -> None:
    proposal = parse_candidate_semantics_proposal(_response(supported, rejected))

    assert len(proposal.candidates) == len(supported)
    assert len(proposal.rejected_hypotheses) == len(rejected)


@pytest.mark.parametrize(
    ("raw", "message"),
    (
        (_response([_supported(), _supported()]), "must be unique"),
        (_response([], [_rejected(), _rejected()]), "must be unique"),
        (_response([_supported()], [_rejected("health_check_timeout")]), "both supported"),
        (_response([_supported("unknown")]), "unsupported diagnosis"),
        (_response([], [_rejected("unknown")]), "unsupported diagnosis"),
        (
            _response(
                [
                    {
                        **_supported(),
                        "supporting_evidence_references": [],
                    }
                ]
            ),
            "requires direct supporting evidence",
        ),
        (_response([], [_rejected(reason="unknown")]), "invalid reason_code"),
    ),
)
def test_candidate_semantics_rejects_incoherent_structures(
    raw: str, message: str
) -> None:
    with pytest.raises(ValueError, match=message):
        parse_candidate_semantics_proposal(raw)


def test_rejected_hypothesis_is_typed_and_confidence_is_metadata() -> None:
    proposal = parse_candidate_semantics_proposal(
        _response([], [_rejected(confidence=0.99)])
    )

    rejected = proposal.rejected_hypotheses[0]
    assert rejected.reason_code is RejectionReason.NO_DIRECT_SUPPORT
    assert rejected.reported_confidence == 0.99


def test_invalid_rejected_reference_is_an_explicit_reference_failure(
    fixture_directory: Path,
) -> None:
    outcome = LLMPolicyInvestigator(
        FakeModel([_response([], [_rejected(references=["E99"])])]),
        CANDIDATE_SEMANTICS_PROMPT_SELECTION,
    ).investigate(_collected(fixture_directory))

    assert isinstance(outcome, LLMPolicyInvestigationFailure)
    assert outcome.status == "invalid_references"


@pytest.mark.parametrize(
    ("supported", "rejected", "outcome", "reason"),
    (
        (
            [_supported()],
            [
                _rejected(reason="symptom_not_root_cause"),
                _rejected("database_migration_failure", confidence=0.99),
            ],
            DecisionOutcome.DIAGNOSIS,
            DecisionReason.SINGLE_SUPPORTED_CANDIDATE,
        ),
        (
            [_supported(strength="moderate")],
            [_rejected(reason="only_shared_vocabulary")],
            DecisionOutcome.DIAGNOSIS,
            DecisionReason.SINGLE_SUPPORTED_CANDIDATE,
        ),
        (
            [],
            [_rejected(), _rejected("database_migration_failure")],
            DecisionOutcome.ABSTENTION,
            DecisionReason.NO_SUPPORTED_CANDIDATE,
        ),
        (
            [
                _supported(),
                _supported("database_migration_failure", supporting=["E1", "E2"]),
            ],
            [],
            DecisionOutcome.NEEDS_REVIEW,
            DecisionReason.CONFLICTING_SUPPORTED_CANDIDATES,
        ),
        (
            [
                _supported(
                    "missing_environment_variable",
                    supporting=["E1", "E2"],
                    contradicting=["E3"],
                )
            ],
            [_rejected()],
            DecisionOutcome.NEEDS_REVIEW,
            DecisionReason.UNRESOLVED_CONTRADICTORY_EVIDENCE,
        ),
    ),
)
def test_policy_receives_only_supported_candidates(
    fixture_directory: Path,
    supported: list[dict[str, object]],
    rejected: list[dict[str, object]],
    outcome: DecisionOutcome,
    reason: DecisionReason,
) -> None:
    result = LLMPolicyInvestigator(
        FakeModel([_response(supported, rejected)]),
        CANDIDATE_SEMANTICS_PROMPT_SELECTION,
    ).investigate(_collected(fixture_directory))

    assert isinstance(result, LLMPolicyInvestigationSuccess)
    assert result.decision.outcome is outcome
    assert result.decision.reason is reason
    assert len(result.uncertainty.candidates) == len(supported)
    assert len(result.proposal.rejected_hypotheses) == len(rejected)


def test_supported_and_rejected_order_do_not_affect_policy(
    fixture_directory: Path,
) -> None:
    supported = [
        _supported(),
        _supported("database_migration_failure", supporting=["E1", "E2"]),
    ]
    rejected = [
        _rejected(),
        _rejected("missing_environment_variable"),
    ]
    decisions = []
    for supported_order, rejected_order in (
        (supported, rejected),
        (list(reversed(supported)), list(reversed(rejected))),
    ):
        proposal = parse_candidate_semantics_proposal(
            _response(supported_order, rejected_order)
        )
        assessment = proposal_to_uncertainty(proposal, _collected(fixture_directory))
        from ai_investigation.decision_policy import decide_investigation

        decisions.append(decide_investigation(assessment))

    assert all(item.outcome is DecisionOutcome.NEEDS_REVIEW for item in decisions)
    assert all(item.diagnosis is None for item in decisions)


def test_evaluation_separates_supported_and_rejected_counts(
    fixture_directory: Path,
) -> None:
    collector, deterministic = _dependencies(fixture_directory)
    scenarios = load_scenarios(fixture_directory / "evaluation_scenarios.json")
    model = FakeModel(
        [
            _response([_supported()], [_rejected()]),
            _response([], [_rejected()]),
            _response(
                [
                    _supported(
                        "missing_environment_variable", supporting=["E1", "E2"]
                    ),
                    _supported(
                        "database_migration_failure", supporting=["E1", "E2"]
                    ),
                ],
                [_rejected("missing_database_configuration")],
            ),
        ]
    )

    report = run_experiment(
        (scenarios[0], scenarios[5], scenarios[6]),
        collector,
        deterministic,
        investigator_mode="llm-policy",
        structured_model=model,
        prompt_version=CANDIDATE_SEMANTICS_PROMPT_SELECTION,
        sleep=lambda seconds: None,
    )
    text = render_text_report(report)

    assert report.aggregate.policy_diagnoses == 1
    assert report.aggregate.policy_abstentions == 1
    assert report.aggregate.policy_needs_review == 1
    assert report.aggregate.zero_candidate_assessments == 1
    assert report.aggregate.single_candidate_assessments == 1
    assert report.aggregate.multi_candidate_assessments == 1
    assert report.aggregate.rejected_hypotheses_total == 3
    assert report.aggregate.assessments_with_rejected_hypotheses == 3
    assert report.scenarios[0].candidate_diagnoses == ("health_check_timeout",)
    assert report.scenarios[0].rejected_hypothesis_diagnoses == (
        "missing_database_configuration",
    )
    assert "Supported candidate diagnoses: health_check_timeout" in text
    assert "Rejected hypothesis diagnoses: missing_database_configuration" in text


def test_candidate_semantics_versions_are_saved_in_experiment_metadata(
    fixture_directory: Path,
    tmp_path: Path,
) -> None:
    scenarios = json.loads(
        (fixture_directory / "evaluation_scenarios.json").read_text(encoding="utf-8")
    )
    scenario_path = tmp_path / "scenario.json"
    scenario_path.write_text(json.dumps(scenarios[:1]), encoding="utf-8")
    experiment_root = tmp_path / "runs"

    exit_code = evaluate_cli.main(
        [
            "--investigator",
            "llm-policy",
            "--prompt-version",
            CANDIDATE_SEMANTICS_PROMPT_SELECTION,
            "--scenarios",
            str(scenario_path),
            "--fixtures",
            str(fixture_directory),
            "--save-experiment",
            "--experiment-dir",
            str(experiment_root),
        ],
        structured_model=FakeModel([_response([_supported()], [_rejected()])]),
        sleep=lambda seconds: None,
    )

    assert exit_code == 0
    record = load_experiment(next(experiment_root.iterdir()))
    assert record.metadata.prompt_version == CANDIDATE_SEMANTICS_PROMPT_VERSION
    assert (
        record.metadata.response_schema_version
        == CANDIDATE_SEMANTICS_RESPONSE_SCHEMA_VERSION
    )
    assert record.report.scenarios[0].rejected_hypothesis_diagnoses == (
        "missing_database_configuration",
    )
