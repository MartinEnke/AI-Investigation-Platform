import json
from pathlib import Path

import pytest

from ai_investigation import evaluate as evaluate_cli
from ai_investigation.decision_policy import DecisionOutcome, DecisionReason, EvidenceStrength
from ai_investigation.evaluation.framework import run_experiment
from ai_investigation.evaluation.loader import load_scenarios
from ai_investigation.evaluation.tracking import compare_experiments, load_experiment
from ai_investigation.evidence import EvidenceCollector
from ai_investigation.investigator import DeploymentFailureInvestigator, request_from_question
from ai_investigation.llm_investigator import ModelProviderError, build_prompt
from ai_investigation.tools import JsonDeploymentTool, JsonLogTool, JsonServiceHealthTool
from ai_investigation.uncertainty_investigator import (
    DECISION_POLICY_VERSION,
    LLMPolicyInvestigationFailure,
    LLMPolicyInvestigationSuccess,
    LLMPolicyInvestigator,
    UNCERTAINTY_PROMPT_VERSION,
    UncertaintyAdapterError,
    build_uncertainty_prompt,
    decision_to_result,
    parse_uncertainty_proposal,
    proposal_to_uncertainty,
    serialize_uncertainty_evidence,
)


class FakeModel:
    provider_name = "fake"
    model_name = "uncertainty-model"

    def __init__(self, responses: list[str]) -> None:
        self.responses = responses
        self.calls = 0
        self.prompts: list[str] = []

    def generate(self, prompt: str) -> str:
        self.prompts.append(prompt)
        response = self.responses[self.calls]
        self.calls += 1
        return response


class FailingModel:
    provider_name = "fake"
    model_name = "uncertainty-model"

    def generate(self, prompt: str) -> str:
        raise ModelProviderError("Provider unavailable.")


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


def _candidate(
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


def _proposal(
    candidates: list[dict[str, object]] | None = None,
    **overrides: object,
) -> str:
    value = {
        "candidates": candidates if candidates is not None else [_candidate()],
        "unsupported_signals_present": False,
        "unsupported_signals_material": False,
        "conflicting_supported_candidates": False,
        "insufficient_evidence": False,
        "reasoning_summary": "Evidence was assessed without selecting an outcome.",
    }
    value.update(overrides)
    return json.dumps(value)


def test_uncertainty_prompt_is_separate_and_does_not_change_v3(
    fixture_directory: Path,
) -> None:
    collected = _collected(fixture_directory)
    v3_before = build_prompt(collected, "v3")

    prompt = build_uncertainty_prompt(collected)

    assert f"Prompt version: {UNCERTAINTY_PROMPT_VERSION}." in prompt
    assert "do not choose a final diagnosis, abstention, or review outcome" in prompt
    assert "without hiding competing candidates" in prompt
    assert "Confidence is reported belief, not a substitute for evidence" in prompt
    assert "unsupported causes as unsupported signals" in prompt
    assert "Do not infer or return source-availability or missing-source metadata" in prompt
    assert "bare integer" in prompt
    assert '"missing_required_sources"' not in prompt
    assert build_prompt(collected, "v3") == v3_before


def test_uncertainty_evidence_uses_stable_explicit_ids(
    fixture_directory: Path,
) -> None:
    collected = _collected(fixture_directory)

    first = serialize_uncertainty_evidence(collected)
    second = serialize_uncertainty_evidence(collected)

    assert first == second
    assert [item["id"] for item in first["evidence"]] == ["E1", "E2", "E3"]
    assert [item["source"] for item in first["evidence"]] == [
        "deployment",
        "logs",
        "service_health",
    ]


@pytest.mark.parametrize(
    "candidates",
    (
        [],
        [_candidate()],
        [
            _candidate(),
            _candidate(
                "database_migration_failure", supporting=["E1", "E2"], confidence=0.6
            ),
        ],
    ),
)
def test_zero_one_and_multiple_candidates_parse(
    candidates: list[dict[str, object]],
) -> None:
    proposal = parse_uncertainty_proposal(
        _proposal(
            candidates,
            conflicting_supported_candidates=len(candidates) > 1,
        )
    )

    assert len(proposal.candidates) == len(candidates)


@pytest.mark.parametrize(
    ("raw", "message"),
    (
        (
            _proposal([_candidate(), _candidate()]),
            "must be unique",
        ),
        (_proposal([_candidate("unknown")]), "unsupported diagnosis"),
        (_proposal([_candidate(strength="certain")]), "invalid evidence_strength"),
        (
            _proposal([_candidate(supporting=["E1", "E2"], contradicting=["E2"])]),
            "both support and contradict",
        ),
        (_proposal([_candidate(supporting=[12])]), "E<number>"),
        (_proposal([_candidate(supporting=["E1", "E1"])]), "must be unique"),
        (
            _proposal(missing_required_sources=["deployment"]),
            "Invalid uncertainty response fields",
        ),
        (_proposal([_candidate(confidence=1.2)]), "between 0.0 and 1.0"),
    ),
)
def test_invalid_uncertainty_responses_fail_explicitly(raw: str, message: str) -> None:
    with pytest.raises(ValueError, match=message):
        parse_uncertainty_proposal(raw)


def test_single_candidate_adapter_preserves_confidence_and_evidence(
    fixture_directory: Path,
) -> None:
    proposal = parse_uncertainty_proposal(_proposal())

    assessment = proposal_to_uncertainty(proposal, _collected(fixture_directory))

    mapped = assessment.candidates[0]
    assert mapped.candidate.diagnosis == "health_check_timeout"
    assert mapped.candidate.evidence_strength is EvidenceStrength.STRONG
    assert mapped.candidate.supporting_evidence_references == (1, 2, 3)
    assert mapped.reported_confidence == 0.8
    assert mapped.required_sources_present is True


def test_multiple_candidates_and_unsupported_signals_are_preserved(
    fixture_directory: Path,
) -> None:
    proposal = parse_uncertainty_proposal(
        _proposal(
            [
                _candidate(),
                _candidate("database_migration_failure", supporting=["E1", "E2"]),
            ],
            conflicting_supported_candidates=True,
            unsupported_signals_present=True,
            unsupported_signals_material=True,
        )
    )

    assessment = proposal_to_uncertainty(proposal, _collected(fixture_directory))

    assert [item.candidate.diagnosis for item in assessment.candidates] == [
        "health_check_timeout",
        "database_migration_failure",
    ]
    assert assessment.conflicting_supported_candidates is True
    assert assessment.unsupported_signals_material is True


def test_adapter_derives_missing_required_sources(
    fixture_directory: Path,
) -> None:
    proposal = parse_uncertainty_proposal(
        _proposal([_candidate(supporting=["E1", "E2"])])
    )

    assessment = proposal_to_uncertainty(
        proposal, _collected(fixture_directory, "deploy-1045")
    )

    assert assessment.missing_required_sources == ("service_health",)
    assert assessment.candidates[0].required_sources_present is False


def test_present_required_sources_are_derived_without_model_declaration(
    fixture_directory: Path,
) -> None:
    proposal = parse_uncertainty_proposal(
        _proposal(
            [
                _candidate(
                    "missing_environment_variable",
                    supporting=["E1", "E2"],
                )
            ]
        )
    )

    assessment = proposal_to_uncertainty(proposal, _collected(fixture_directory))

    assert assessment.missing_required_sources == ()
    assert assessment.candidates[0].required_sources_present is True


def test_adapter_rejects_invalid_or_incoherent_references(
    fixture_directory: Path,
) -> None:
    invalid = parse_uncertainty_proposal(
        _proposal([_candidate(supporting=["E1", "E99"])])
    )
    with pytest.raises(UncertaintyAdapterError) as raised:
        proposal_to_uncertainty(invalid, _collected(fixture_directory))
    assert raised.value.invalid_references is True


def test_contradicting_references_remain_visible(fixture_directory: Path) -> None:
    proposal = parse_uncertainty_proposal(
        _proposal(
            [
                _candidate(
                    "missing_environment_variable",
                    supporting=["E1", "E2"],
                    contradicting=["E3"],
                )
            ]
        )
    )

    assessment = proposal_to_uncertainty(proposal, _collected(fixture_directory))

    assert assessment.candidates[0].candidate.contradicting_evidence_references == (3,)


def test_candidate_order_cannot_choose_a_winner(fixture_directory: Path) -> None:
    candidates = [
        _candidate(),
        _candidate("database_migration_failure", supporting=["E1", "E2"]),
    ]
    decisions = []
    for ordered in (candidates, list(reversed(candidates))):
        proposal = parse_uncertainty_proposal(
            _proposal(ordered, conflicting_supported_candidates=True)
        )
        assessment = proposal_to_uncertainty(proposal, _collected(fixture_directory))
        from ai_investigation.decision_policy import decide_investigation

        decisions.append(decide_investigation(assessment))

    assert all(decision.outcome is DecisionOutcome.NEEDS_REVIEW for decision in decisions)
    assert all(decision.diagnosis is None for decision in decisions)


@pytest.mark.parametrize(
    ("raw", "outcome", "reason"),
    (
        (_proposal(), DecisionOutcome.DIAGNOSIS, DecisionReason.SINGLE_SUPPORTED_CANDIDATE),
        (
            _proposal([_candidate(strength="weak", confidence=0.99)]),
            DecisionOutcome.ABSTENTION,
            DecisionReason.INSUFFICIENT_EVIDENCE,
        ),
        (
            _proposal([], insufficient_evidence=True),
            DecisionOutcome.ABSTENTION,
            DecisionReason.INSUFFICIENT_EVIDENCE,
        ),
        (
            _proposal([]),
            DecisionOutcome.ABSTENTION,
            DecisionReason.NO_SUPPORTED_CANDIDATE,
        ),
        (
            _proposal(
                [
                    _candidate(confidence=0.99),
                    _candidate(
                        "database_migration_failure",
                        supporting=["E1", "E2"],
                        confidence=0.99,
                    ),
                ],
                conflicting_supported_candidates=True,
            ),
            DecisionOutcome.NEEDS_REVIEW,
            DecisionReason.CONFLICTING_SUPPORTED_CANDIDATES,
        ),
        (
            _proposal(
                [
                    _candidate(
                        "missing_environment_variable",
                        supporting=["E1", "E2"],
                        contradicting=["E3"],
                        confidence=0.99,
                    )
                ]
            ),
            DecisionOutcome.NEEDS_REVIEW,
            DecisionReason.UNRESOLVED_CONTRADICTORY_EVIDENCE,
        ),
        (
            _proposal(
                [],
                unsupported_signals_present=True,
                unsupported_signals_material=True,
            ),
            DecisionOutcome.ABSTENTION,
            DecisionReason.UNSUPPORTED_SIGNALS_ONLY,
        ),
    ),
)
def test_policy_controls_each_integrated_outcome(
    fixture_directory: Path,
    raw: str,
    outcome: DecisionOutcome,
    reason: DecisionReason,
) -> None:
    result = LLMPolicyInvestigator(FakeModel([raw])).investigate(
        _collected(fixture_directory)
    )

    assert isinstance(result, LLMPolicyInvestigationSuccess)
    assert result.decision.outcome is outcome
    assert result.decision.reason is reason
    assert (result.result.root_cause is not None) is (
        outcome is DecisionOutcome.DIAGNOSIS
    )


def test_result_adapter_keeps_review_candidates_and_reason_inspectable(
    fixture_directory: Path,
) -> None:
    raw = _proposal(
        [
            _candidate(),
            _candidate("database_migration_failure", supporting=["E1", "E2"]),
        ],
        conflicting_supported_candidates=True,
    )
    outcome = LLMPolicyInvestigator(FakeModel([raw])).investigate(
        _collected(fixture_directory)
    )

    assert isinstance(outcome, LLMPolicyInvestigationSuccess)
    assert outcome.result.root_cause is None
    assert "Policy outcome: needs_review." in outcome.result.limitations
    assert "Policy reason: conflicting_supported_candidates." in outcome.result.limitations
    assert "Candidate diagnoses:" in outcome.result.limitations[2]


def test_missing_source_reaches_policy_abstention_instead_of_adapter_failure(
    fixture_directory: Path,
) -> None:
    outcome = LLMPolicyInvestigator(
        FakeModel([_proposal([_candidate(supporting=["E1", "E2"])])])
    ).investigate(_collected(fixture_directory, "deploy-1045"))

    assert isinstance(outcome, LLMPolicyInvestigationSuccess)
    assert outcome.uncertainty.missing_required_sources == ("service_health",)
    assert outcome.decision.outcome is DecisionOutcome.ABSTENTION
    assert outcome.decision.reason is DecisionReason.MISSING_REQUIRED_EVIDENCE


def test_invalid_reference_remains_distinct(
    fixture_directory: Path,
) -> None:
    invalid_reference = LLMPolicyInvestigator(
        FakeModel([_proposal([_candidate(supporting=["E1", "E99"])])])
    ).investigate(_collected(fixture_directory))

    assert isinstance(invalid_reference, LLMPolicyInvestigationFailure)
    assert invalid_reference.status == "invalid_references"


def test_invalid_response_and_provider_failure_remain_distinct(
    fixture_directory: Path,
) -> None:
    invalid_response = LLMPolicyInvestigator(FakeModel(["not json"])).investigate(
        _collected(fixture_directory)
    )
    provider_failure = LLMPolicyInvestigator(FailingModel()).investigate(
        _collected(fixture_directory)
    )

    assert isinstance(invalid_response, LLMPolicyInvestigationFailure)
    assert invalid_response.status == "invalid_response"
    assert isinstance(provider_failure, LLMPolicyInvestigationFailure)
    assert provider_failure.status == "provider_failure"
    assert provider_failure.errors == ("Provider unavailable.",)


def test_llm_policy_evaluation_reports_policy_dimensions_and_pacing(
    fixture_directory: Path,
) -> None:
    collector, deterministic = _dependencies(fixture_directory)
    scenarios = load_scenarios(fixture_directory / "evaluation_scenarios.json")
    model = FakeModel(
        [
            _proposal(),
            _proposal(
                [
                    _candidate(
                        "missing_environment_variable",
                        supporting=["E1", "E2"],
                        confidence=0.7,
                    )
                ]
            ),
        ]
    )
    sleeps: list[float] = []

    report = run_experiment(
        (scenarios[0], scenarios[1], scenarios[5]),
        collector,
        deterministic,
        investigator_mode="llm-policy",
        structured_model=model,
        prompt_version="v4-uncertainty",
        request_delay_seconds=3,
        sleep=sleeps.append,
    )

    assert model.calls == 2
    assert sleeps == [3]
    assert report.aggregate.policy_diagnoses == 2
    assert report.aggregate.single_candidate_assessments == 2
    assert report.scenarios[0].policy_outcome == "diagnosis"
    assert report.scenarios[0].candidate_diagnoses == ("health_check_timeout",)
    assert report.scenarios[1].execution_status == "not_evaluated"


def test_offline_evaluation_reaches_all_policy_outcomes(
    fixture_directory: Path,
) -> None:
    collector, deterministic = _dependencies(fixture_directory)
    scenarios = load_scenarios(fixture_directory / "evaluation_scenarios.json")
    model = FakeModel(
        [
            _proposal(),
            _proposal([], unsupported_signals_present=True),
            _proposal(
                [
                    _candidate(
                        "missing_environment_variable",
                        supporting=["E1", "E2"],
                    ),
                    _candidate(
                        "database_migration_failure",
                        supporting=["E1", "E2"],
                    ),
                ],
                conflicting_supported_candidates=True,
            ),
        ]
    )

    report = run_experiment(
        (scenarios[0], scenarios[5], scenarios[6]),
        collector,
        deterministic,
        investigator_mode="llm-policy",
        structured_model=model,
        prompt_version="v4-uncertainty",
        sleep=lambda seconds: None,
    )

    assert report.aggregate.policy_diagnoses == 1
    assert report.aggregate.policy_abstentions == 1
    assert report.aggregate.policy_needs_review == 1
    assert report.aggregate.single_candidate_assessments == 1
    assert report.aggregate.multi_candidate_assessments == 1


def test_cli_saves_llm_policy_identity_and_versions(
    fixture_directory: Path,
    tmp_path: Path,
) -> None:
    scenario_source = json.loads(
        (fixture_directory / "evaluation_scenarios.json").read_text(encoding="utf-8")
    )
    scenario_path = tmp_path / "scenario.json"
    scenario_path.write_text(json.dumps(scenario_source[:1]), encoding="utf-8")
    experiment_root = tmp_path / "runs"

    exit_code = evaluate_cli.main(
        [
            "--investigator",
            "llm-policy",
            "--prompt-version",
            "v4-uncertainty",
            "--scenarios",
            str(scenario_path),
            "--fixtures",
            str(fixture_directory),
            "--save-experiment",
            "--experiment-dir",
            str(experiment_root),
        ],
        structured_model=FakeModel([_proposal()]),
        sleep=lambda seconds: None,
    )

    assert exit_code == 0
    record = load_experiment(next(experiment_root.iterdir()))
    assert record.metadata.investigator_mode == "llm-policy"
    assert record.metadata.prompt_version == UNCERTAINTY_PROMPT_VERSION
    assert record.metadata.decision_policy_version == DECISION_POLICY_VERSION
    assert record.report.scenarios[0].policy_outcome == "diagnosis"
    assert compare_experiments(record, record).summary.regressed == 0
