import json

import pytest

from ai_investigation.evaluation.framework import _source_differences
from ai_investigation.uncertainty_investigator import (
    CANDIDATE_SEMANTICS_RESPONSE_JSON_SCHEMA,
    parse_candidate_semantics_proposal,
)


def _candidate(diagnosis: str = "missing_environment_variable") -> dict[str, object]:
    return {
        "diagnosis_id": diagnosis,
        "supporting_evidence_references": ["E1"],
        "contradicting_evidence_references": [],
        "evidence_strength": "strong",
        "confidence": 0.8,
    }


def _rejected(diagnosis: str) -> dict[str, object]:
    return {
        "diagnosis_id": diagnosis,
        "reason_code": "no_direct_support",
        "reason_summary": "No direct evidence supports this diagnosis.",
        "relevant_evidence_references": [],
        "confidence": 0.1,
    }


def _response(
    candidates: list[dict[str, object]],
    *,
    rejected: list[dict[str, object]] | None = None,
    legacy_conflict: bool | None = None,
) -> str:
    value: dict[str, object] = {
        "supported_candidates": candidates,
        "rejected_hypotheses": rejected or [],
        "unsupported_signals_present": False,
        "unsupported_signals_material": False,
        "insufficient_evidence": False,
        "reasoning_summary": "Candidate support was assessed.",
    }
    if legacy_conflict is not None:
        value["conflicting_supported_candidates"] = legacy_conflict
    return json.dumps(value)


def test_conflict_is_removed_from_current_provider_schema() -> None:
    assert "conflicting_supported_candidates" not in (
        CANDIDATE_SEMANTICS_RESPONSE_JSON_SCHEMA["required"]
    )
    assert "conflicting_supported_candidates" not in (
        CANDIDATE_SEMANTICS_RESPONSE_JSON_SCHEMA["properties"]
    )


def test_conflict_is_derived_from_supported_candidates() -> None:
    single = parse_candidate_semantics_proposal(_response([_candidate()]))
    multiple = parse_candidate_semantics_proposal(
        _response(
            [
                _candidate(),
                _candidate("database_migration_failure"),
            ]
        )
    )

    assert single.conflicting_supported_candidates is False
    assert multiple.conflicting_supported_candidates is True


def test_legacy_model_conflict_value_is_accepted_but_ignored() -> None:
    incorrectly_true = parse_candidate_semantics_proposal(
        _response([_candidate()], legacy_conflict=True)
    )
    incorrectly_false = parse_candidate_semantics_proposal(
        _response(
            [_candidate(), _candidate("database_migration_failure")],
            legacy_conflict=False,
        )
    )

    assert incorrectly_true.conflicting_supported_candidates is False
    assert incorrectly_false.conflicting_supported_candidates is True


def test_single_overlap_reports_the_diagnosis_id() -> None:
    with pytest.raises(
        ValueError,
        match=(
            "^Diagnoses cannot be both supported and rejected: "
            "missing_environment_variable$"
        ),
    ):
        parse_candidate_semantics_proposal(
            _response(
                [_candidate()],
                rejected=[_rejected("missing_environment_variable")],
            )
        )


def test_multiple_overlaps_are_reported_in_deterministic_sorted_order() -> None:
    with pytest.raises(
        ValueError,
        match=(
            "^Diagnoses cannot be both supported and rejected: "
            "database_migration_failure, missing_environment_variable$"
        ),
    ):
        parse_candidate_semantics_proposal(
            _response(
                [
                    _candidate("missing_environment_variable"),
                    _candidate("database_migration_failure"),
                ],
                rejected=[
                    _rejected("missing_environment_variable"),
                    _rejected("database_migration_failure"),
                ],
            )
        )


def test_disjoint_collections_remain_valid_with_legacy_conflict_field() -> None:
    proposal = parse_candidate_semantics_proposal(
        _response(
            [_candidate("missing_environment_variable")],
            rejected=[_rejected("database_migration_failure")],
            legacy_conflict=True,
        )
    )

    assert tuple(item.diagnosis_id for item in proposal.candidates) == (
        "missing_environment_variable",
    )
    assert tuple(item.diagnosis_id for item in proposal.rejected_hypotheses) == (
        "database_migration_failure",
    )
    assert proposal.conflicting_supported_candidates is False


def test_source_differences_are_unique_categories() -> None:
    missing, unexpected = _source_differences(
        ("deployment", "logs", "logs"),
        ("logs",),
    )

    assert missing == ("deployment",)
    assert unexpected == ()


def test_referenced_source_cannot_also_be_reported_missing() -> None:
    expected = ("deployment", "logs", "logs", "service_health")
    actual = ("logs", "logs", "service_health")

    missing, unexpected = _source_differences(expected, actual)

    assert missing == ("deployment",)
    assert len(missing) == len(set(missing))
    assert not set(missing) & set(actual)
    assert unexpected == ()
