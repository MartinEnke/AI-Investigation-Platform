import pytest

from ai_investigation.diagnosis import (
    DiagnosisContext,
    database_migration_failure_rule,
    evaluate_diagnoses,
    health_check_timeout_rule,
    missing_environment_variable_rule,
)


def context(*messages: str, stage: str = "startup") -> DiagnosisContext:
    return DiagnosisContext(
        deployment_id="deploy-2000",
        deployment={"status": "failed", "failed_stage": stage},
        error_logs=tuple({"level": "error", "message": message} for message in messages),
        service_health=None,
    )


@pytest.mark.parametrize(
    "message",
    (
        "Missing environment variable DATABASE_URL",
        "Required environment variable DATABASE_URL is not set",
        "KeyError: DATABASE_URL",
    ),
)
def test_accepted_missing_environment_variable_patterns(message: str) -> None:
    match = missing_environment_variable_rule(context(message))

    assert match is not None
    assert "DATABASE_URL" in match.root_cause


def test_missing_environment_variable_extracts_variable_name() -> None:
    match = missing_environment_variable_rule(context("KeyError: REDIS_ENDPOINT"))

    assert match is not None
    assert match.root_cause.endswith("REDIS_ENDPOINT was missing.")


def test_missing_environment_variable_is_case_insensitive() -> None:
    assert missing_environment_variable_rule(
        context("required ENVIRONMENT VARIABLE database_url IS NOT SET")
    ) is not None


@pytest.mark.parametrize(
    "message",
    (
        "Migration failed",
        "database migration failed",
        "failed to apply migrations",
        "alembic.util.exc.CommandError: revision missing",
    ),
)
def test_accepted_direct_migration_patterns(message: str) -> None:
    assert database_migration_failure_rule(context(message)) is not None


def test_migration_context_combines_with_missing_relation() -> None:
    assert database_migration_failure_rule(
        context("Applying migration revision 42", "relation users does not exist")
    ) is not None


def test_migration_matching_is_case_insensitive() -> None:
    assert database_migration_failure_rule(context("FAILED TO APPLY MIGRATIONS")) is not None


def test_unrelated_error_text_produces_no_match() -> None:
    assert evaluate_diagnoses(context("Connection reset by peer")).matches == ()


def test_later_relevant_error_is_inspected() -> None:
    result = evaluate_diagnoses(
        context("Connection reset by peer", "Missing environment variable DATABASE_URL")
    )

    assert result.match is not None
    assert "DATABASE_URL" in result.match.root_cause


def test_multiple_diagnosis_matches_are_retained_as_conflict() -> None:
    result = evaluate_diagnoses(
        context("Missing environment variable DATABASE_URL", "Migration failed")
    )

    assert result.match is None
    assert result.has_conflict
    assert len(result.matches) == 2


def test_health_check_timeout_matches_required_predicates() -> None:
    diagnosis_context = DiagnosisContext(
        deployment_id="deploy-1042",
        deployment={"status": "failed", "failed_stage": "health_check"},
        error_logs=({"level": "error", "reason": "timeout", "message": "timed out"},),
        service_health={"status": "unhealthy"},
    )

    match = health_check_timeout_rule(diagnosis_context)

    assert match is not None


@pytest.mark.parametrize(
    ("deployment", "error_logs", "service_health"),
    (
        (
            {"status": "running", "failed_stage": "health_check"},
            ({"reason": "timeout"},),
            {"status": "unhealthy"},
        ),
        (
            {"status": "failed", "failed_stage": "rollout"},
            ({"reason": "timeout"},),
            {"status": "unhealthy"},
        ),
        (
            {"status": "failed", "failed_stage": "health_check"},
            ({"reason": "connection_refused"},),
            {"status": "unhealthy"},
        ),
        (
            {"status": "failed", "failed_stage": "health_check"},
            ({"reason": "timeout"},),
            {"status": "healthy"},
        ),
    ),
)
def test_health_check_timeout_rejects_missing_required_predicate(
    deployment: dict[str, str],
    error_logs: tuple[dict[str, str], ...],
    service_health: dict[str, str],
) -> None:
    diagnosis_context = DiagnosisContext(
        deployment_id="deploy-1042",
        deployment=deployment,
        error_logs=error_logs,
        service_health=service_health,
    )

    assert health_check_timeout_rule(diagnosis_context) is None


def test_generic_database_error_without_migration_context_does_not_match() -> None:
    assert database_migration_failure_rule(context("relation users does not exist")) is None
