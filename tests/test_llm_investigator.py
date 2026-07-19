import json
from pathlib import Path

import pytest

from ai_investigation.evidence import CollectedEvidence, EvidenceCollector
from ai_investigation.investigator import DeploymentFailureInvestigator, request_from_question
from ai_investigation.llm_investigator import (
    LLMDecision,
    LLM_RESPONSE_JSON_SCHEMA,
    LLMInvestigationFailure,
    LLMInvestigationSuccess,
    LLMInvestigator,
    ModelProviderError,
    ModelRefusalError,
    build_prompt,
    parse_decision,
    validate_evidence_references,
)
from ai_investigation.tools import JsonDeploymentTool, JsonLogTool, JsonServiceHealthTool


class FakeModel:
    def __init__(self, response: str = "", error: Exception | None = None) -> None:
        self.response = response
        self.error = error
        self.prompts: list[str] = []

    def generate(self, prompt: str) -> str:
        self.prompts.append(prompt)
        if self.error is not None:
            raise self.error
        return self.response


def collect(fixture_directory: Path, deployment_id: str = "deploy-1042") -> CollectedEvidence:
    collector = EvidenceCollector(
        JsonDeploymentTool(fixture_directory / "deployments.json"),
        JsonLogTool(fixture_directory / "logs.json"),
        JsonServiceHealthTool(fixture_directory / "service_health.json"),
    )
    return collector.collect(
        request_from_question(f"Why did deployment {deployment_id} fail?")
    )


def response(**overrides: object) -> str:
    value = {
        "outcome": "diagnosis",
        "diagnosis_id": "health_check_timeout",
        "confidence": 0.9,
        "evidence_references": [1, 2, 3],
        "abstention_reason": None,
    }
    value.update(overrides)
    return json.dumps(value)


def test_prompt_contains_complete_structured_and_numbered_evidence(
    fixture_directory: Path,
) -> None:
    collected = collect(fixture_directory)

    prompt = build_prompt(collected)

    assert prompt == build_prompt(collected)
    assert '"reason":"timeout"' in prompt
    assert '"reference":1,"source":"deployment"' in prompt
    assert '"reference":2,"source":"logs"' in prompt
    assert '"reference":3,"source":"service_health"' in prompt
    assert "Do not invent or request additional evidence." in prompt


def test_response_schema_matches_strict_application_contract() -> None:
    assert LLM_RESPONSE_JSON_SCHEMA["additionalProperties"] is False
    assert set(LLM_RESPONSE_JSON_SCHEMA["required"]) == {
        "outcome",
        "diagnosis_id",
        "confidence",
        "evidence_references",
        "abstention_reason",
    }


def test_parse_valid_diagnosis_preserves_low_confidence() -> None:
    decision = parse_decision(response(confidence=0.22))

    assert decision.outcome == "diagnosis"
    assert decision.diagnosis_id == "health_check_timeout"
    assert decision.confidence == 0.22
    assert decision.evidence_references == (1, 2, 3)


def test_parse_valid_abstention() -> None:
    decision = parse_decision(
        response(
            outcome="abstain",
            diagnosis_id=None,
            confidence=0.31,
            evidence_references=[1, 2],
            abstention_reason="conflicting_evidence",
        )
    )

    assert decision.outcome == "abstain"
    assert decision.abstention_reason == "conflicting_evidence"


@pytest.mark.parametrize(
    ("raw_response", "message"),
    (
        ("not json", "not valid JSON"),
        ("[]", "must be a JSON object"),
        (response(extra=True), "unexpected fields: extra"),
        (response(diagnosis_id="unknown"), "supported diagnosis"),
        (response(confidence=True), "non-boolean number"),
        (response(confidence=1.1), "between 0.0 and 1.0"),
        (response(evidence_references=[True]), "list of integers"),
        (response(abstention_reason="low_confidence"), "must be null"),
    ),
)
def test_invalid_structured_responses_are_rejected(
    raw_response: str,
    message: str,
) -> None:
    with pytest.raises(ValueError, match=message):
        parse_decision(raw_response)


def test_reference_validation_checks_integrity_and_source_coverage(
    fixture_directory: Path,
) -> None:
    collected = collect(fixture_directory)
    decision = LLMDecision(
        outcome="diagnosis",
        diagnosis_id="health_check_timeout",
        confidence=0.8,
        evidence_references=(2, 2, 4),
        abstention_reason=None,
    )

    errors = validate_evidence_references(decision, collected)

    assert errors == (
        "Evidence references must be unique.",
        "Evidence references do not exist: (4,).",
    )


def test_reference_validation_requires_diagnosis_sources(
    fixture_directory: Path,
) -> None:
    collected = collect(fixture_directory)
    decision = parse_decision(response(evidence_references=[1, 2]))

    assert validate_evidence_references(decision, collected) == (
        "Diagnosis references are missing required sources: service_health.",
    )


def test_valid_diagnosis_converts_to_result_in_collection_order(
    fixture_directory: Path,
) -> None:
    model = FakeModel(response(evidence_references=[3, 1, 2], confidence=0.22))

    outcome = LLMInvestigator(model).investigate(collect(fixture_directory))

    assert isinstance(outcome, LLMInvestigationSuccess)
    assert outcome.result.root_cause == (
        "The deployment health check timed out because the target service was unhealthy."
    )
    assert outcome.result.confidence == 0.22
    assert [item.source for item in outcome.result.evidence] == [
        "deployment",
        "logs",
        "service_health",
    ]
    assert outcome.result.decision_trace is None
    assert len(model.prompts) == 1


def test_valid_abstention_converts_to_inconclusive_result(
    fixture_directory: Path,
) -> None:
    model = FakeModel(
        response(
            outcome="abstain",
            diagnosis_id=None,
            confidence=0.31,
            evidence_references=[1, 2],
            abstention_reason="conflicting_evidence",
        )
    )

    outcome = LLMInvestigator(model).investigate(collect(fixture_directory))

    assert isinstance(outcome, LLMInvestigationSuccess)
    assert outcome.result.root_cause is None
    assert outcome.result.confidence == 0.31
    assert outcome.result.limitations == (
        "The model found conflicting evidence for supported diagnoses.",
    )


def test_invalid_references_do_not_produce_investigation_result(
    fixture_directory: Path,
) -> None:
    model = FakeModel(response(evidence_references=[1, 99]))

    outcome = LLMInvestigator(model).investigate(collect(fixture_directory))

    assert isinstance(outcome, LLMInvestigationFailure)
    assert outcome.status == "invalid_references"
    assert outcome.errors == ("Evidence references do not exist: (99,).",)


@pytest.mark.parametrize(
    ("error", "status"),
    (
        (ModelRefusalError("Blocked by model."), "refused"),
        (ModelProviderError("Provider unavailable."), "provider_failure"),
    ),
)
def test_model_execution_failures_remain_distinct(
    fixture_directory: Path,
    error: Exception,
    status: str,
) -> None:
    outcome = LLMInvestigator(FakeModel(error=error)).investigate(
        collect(fixture_directory)
    )

    assert isinstance(outcome, LLMInvestigationFailure)
    assert outcome.status == status
    assert outcome.errors == (str(error),)


def test_missing_deployment_is_not_evaluated_and_model_is_not_called(
    fixture_directory: Path,
) -> None:
    model = FakeModel(response())

    outcome = LLMInvestigator(model).investigate(
        collect(fixture_directory, "deploy-9999")
    )

    assert isinstance(outcome, LLMInvestigationFailure)
    assert outcome.status == "not_evaluated"
    assert model.prompts == []


def test_both_paths_accept_the_same_collected_evidence_instance(
    fixture_directory: Path,
) -> None:
    deployments = JsonDeploymentTool(fixture_directory / "deployments.json")
    logs = JsonLogTool(fixture_directory / "logs.json")
    health = JsonServiceHealthTool(fixture_directory / "service_health.json")
    collected = EvidenceCollector(deployments, logs, health).collect(
        request_from_question("Why did deployment deploy-1042 fail?")
    )
    deterministic = DeploymentFailureInvestigator(deployments, logs, health)

    deterministic_result = deterministic.investigate_evidence(collected)
    llm_outcome = LLMInvestigator(FakeModel(response())).investigate(collected)

    assert deterministic_result.root_cause is not None
    assert isinstance(llm_outcome, LLMInvestigationSuccess)
    assert llm_outcome.result.root_cause == deterministic_result.root_cause
