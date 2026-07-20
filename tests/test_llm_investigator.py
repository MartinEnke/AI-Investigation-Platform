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
    PROMPT_VERSION_V1,
    PROMPT_VERSION_V2,
    PROMPT_VERSION_V3,
    build_prompt,
    parse_decision,
    serialize_evidence_v2,
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


def test_prompt_v1_is_the_preserved_default(fixture_directory: Path) -> None:
    collected = collect(fixture_directory)

    assert build_prompt(collected) == build_prompt(collected, "v1")
    assert f"Prompt version: {PROMPT_VERSION_V1}." in build_prompt(collected)
    assert '"reference":1,"source":"deployment"' in build_prompt(collected)


@pytest.mark.parametrize("version", ("v1", "v2", "v3"))
def test_all_prompt_versions_are_selectable(
    fixture_directory: Path,
    version: str,
) -> None:
    assert f"Prompt version: llm-investigator-{version}." in build_prompt(
        collect(fixture_directory), version
    )


def test_prompt_v2_uses_self_contained_exact_evidence_ids(
    fixture_directory: Path,
) -> None:
    collected = collect(fixture_directory)

    payload = serialize_evidence_v2(collected)

    assert [item["id"] for item in payload["evidence"]] == [1, 2, 3]
    assert payload["evidence"][0] == {
        "id": 1,
        "type": "deployment",
        "source": "deployment",
        "observation": "deploy-1042 has status failed and failed stage health_check.",
        "content": {
            "id": "deploy-1042",
            "service": "checkout-api",
            "status": "failed",
            "failed_stage": "health_check",
        },
    }
    assert payload["evidence"][1]["type"] == "error_log"
    assert payload["evidence"][1]["source"] == "logs"
    assert payload["evidence"][1]["content"]["message"] == (
        "Health check timed out after 30 seconds."
    )


def test_prompt_v2_contains_grounding_abstention_and_diagnosis_boundaries(
    fixture_directory: Path,
) -> None:
    prompt = build_prompt(collect(fixture_directory), "v2")

    assert f"Prompt version: {PROMPT_VERSION_V2}." in prompt
    assert "The only valid evidence IDs are [1, 2, 3]" in prompt
    assert "never invent an evidence ID" in prompt
    assert "directly support the diagnosis" in prompt
    assert "incomplete, conflicting, unsupported" in prompt
    assert "multiple diagnoses equally plausible" in prompt
    assert "generic errors" in prompt
    assert "Prefer abstention over guessing" in prompt
    assert "Distinguish missing_environment_variable from missing_database_configuration" in prompt
    assert "specifically supports missing or invalid database configuration" in prompt


def test_llm_investigator_selects_prompt_v2(fixture_directory: Path) -> None:
    model = FakeModel(response())

    LLMInvestigator(model, "v2").investigate(collect(fixture_directory))

    assert len(model.prompts) == 1
    assert f"Prompt version: {PROMPT_VERSION_V2}." in model.prompts[0]


def test_prompt_v3_requires_complete_evidence_source_coverage(
    fixture_directory: Path,
) -> None:
    prompt = build_prompt(collect(fixture_directory), "v3")

    assert f"Prompt version: {PROMPT_VERSION_V3}." in prompt
    assert "Evidence references are part of the validity contract" in prompt
    assert "Every diagnosis must reference the deployment evidence" in prompt
    assert "all log evidence necessary to establish the claimed cause" in prompt
    assert "Reference service-health evidence whenever" in prompt
    assert "Do not return a diagnosis when a required evidence source is missing" in prompt
    assert "return only these exact integer IDs" in prompt


def test_prompt_v3_tightens_database_and_health_check_boundaries(
    fixture_directory: Path,
) -> None:
    prompt = build_prompt(collect(fixture_directory), "v3")

    assert "A generic database error does not establish database_migration_failure" in prompt
    assert "explicitly connects the failure to migration execution" in prompt
    assert "A timeout symptom does not establish health_check_timeout" in prompt
    assert "directly connects the timeout to a deployment health check" in prompt
    assert "Never choose the closest supported diagnosis" in prompt
    assert "similarity is not sufficient causal evidence" in prompt
    assert "outside the supported diagnosis set" in prompt


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
