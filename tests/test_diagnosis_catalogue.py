import json

from ai_investigation.decision_policy import SUPPORTED_DIAGNOSES
from ai_investigation.diagnosis_catalogue import (
    DIAGNOSIS_CATALOGUE,
    DIAGNOSIS_CATALOGUE_VERSION,
    render_diagnosis_catalogue,
)
from ai_investigation.evidence import EvidenceCollector
from ai_investigation.investigator import request_from_question
from ai_investigation.tools import (
    JsonDeploymentTool,
    JsonLogTool,
    JsonServiceHealthTool,
)
from ai_investigation.uncertainty_investigator import (
    CANDIDATE_SEMANTICS_PROMPT_SELECTION,
    CANDIDATE_SEMANTICS_PROMPT_VERSION,
    CANDIDATE_SEMANTICS_RESPONSE_SCHEMA_VERSION,
    build_uncertainty_prompt,
    uncertainty_prompt_identifier,
    uncertainty_schema_identifier,
)


def test_catalogue_defines_every_supported_diagnosis_once() -> None:
    diagnosis_ids = tuple(item.diagnosis_id for item in DIAGNOSIS_CATALOGUE)

    assert len(diagnosis_ids) == len(set(diagnosis_ids))
    assert set(diagnosis_ids) == SUPPORTED_DIAGNOSES


def test_catalogue_version_and_semantic_fields_are_explicit() -> None:
    assert DIAGNOSIS_CATALOGUE_VERSION == "diagnosis-catalogue-v1"
    for definition in DIAGNOSIS_CATALOGUE:
        assert definition.root_cause_description
        assert definition.qualifying_evidence
        assert definition.insufficient_evidence
        assert definition.negative_boundaries
        assert definition.causal_role
        assert definition.related_diagnosis_distinctions


def test_catalogue_rendering_is_deterministic_and_ordered() -> None:
    first = render_diagnosis_catalogue()
    second = render_diagnosis_catalogue()
    parsed = json.loads(first)

    assert first == second
    assert parsed["version"] == DIAGNOSIS_CATALOGUE_VERSION
    assert tuple(item["id"] for item in parsed["diagnoses"]) == tuple(
        item.diagnosis_id for item in DIAGNOSIS_CATALOGUE
    )


def test_candidate_prompt_renders_operational_boundaries(fixture_directory) -> None:
    collector = EvidenceCollector(
        JsonDeploymentTool(fixture_directory / "deployments.json"),
        JsonLogTool(fixture_directory / "logs.json"),
        JsonServiceHealthTool(fixture_directory / "service_health.json"),
    )
    collected = collector.collect(
        request_from_question("Why did deployment deploy-1042 fail?")
    )

    prompt = build_uncertainty_prompt(
        collected,
        CANDIDATE_SEMANTICS_PROMPT_SELECTION,
    )

    assert f"Diagnosis catalogue version: {DIAGNOSIS_CATALOGUE_VERSION}" in prompt
    assert "a missing DATABASE_URL environment key remains a missing environment variable" in prompt
    assert "relation users does not exist" in prompt
    assert "without migration or schema-update execution context" in prompt
    assert "health-check timeout is a downstream symptom" in prompt
    assert "Do not map generic evidence to the nearest available diagnosis" in prompt
    assert "A diagnosis ID must appear in at most one collection" in prompt


def test_prompt_alias_and_response_schema_remain_compatible() -> None:
    assert CANDIDATE_SEMANTICS_PROMPT_SELECTION == (
        "v4-uncertainty-candidate-semantics"
    )
    assert CANDIDATE_SEMANTICS_PROMPT_VERSION == (
        "llm-investigator-v4-uncertainty-candidate-semantics-"
        "contract-v3-catalogue-v1"
    )
    assert CANDIDATE_SEMANTICS_RESPONSE_SCHEMA_VERSION == (
        "llm-uncertainty-proposal-v4"
    )
    assert (
        uncertainty_prompt_identifier(CANDIDATE_SEMANTICS_PROMPT_SELECTION)
        == CANDIDATE_SEMANTICS_PROMPT_VERSION
    )
    assert (
        uncertainty_schema_identifier(CANDIDATE_SEMANTICS_PROMPT_SELECTION)
        == CANDIDATE_SEMANTICS_RESPONSE_SCHEMA_VERSION
    )
