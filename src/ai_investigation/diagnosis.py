"""Small deterministic rule set for supported deployment failure diagnoses."""

import re
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import Any

JsonRecord = Mapping[str, Any]


@dataclass(frozen=True, slots=True)
class DiagnosisContext:
    """The evidence available to deterministic diagnosis rules."""

    deployment_id: str
    deployment: JsonRecord
    error_logs: tuple[JsonRecord, ...]
    service_health: JsonRecord | None


@dataclass(frozen=True, slots=True)
class DiagnosisMatch:
    """A supported root cause and its diagnosis-specific presentation."""

    root_cause: str
    answer: str
    confidence: float
    evidence_sources: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class DiagnosisResult:
    """All rule matches, retained so zero, one, and conflicts stay explicit."""

    matches: tuple[DiagnosisMatch, ...]

    @property
    def match(self) -> DiagnosisMatch | None:
        return self.matches[0] if len(self.matches) == 1 else None

    @property
    def has_conflict(self) -> bool:
        return len(self.matches) > 1


DiagnosisRule = Callable[[DiagnosisContext], DiagnosisMatch | None]

_MISSING_VARIABLE_PATTERNS = (
    re.compile(r"\bmissing environment variable\s+([A-Z_][A-Z0-9_]*)\b", re.IGNORECASE),
    re.compile(
        r"\brequired environment variable\s+([A-Z_][A-Z0-9_]*)\s+is not set\b",
        re.IGNORECASE,
    ),
    re.compile(r"\bkeyerror:\s*['\"]?([A-Z_][A-Z0-9_]*)['\"]?\b", re.IGNORECASE),
)
_DIRECT_MIGRATION_PATTERNS = (
    re.compile(r"\bmigration failed\b", re.IGNORECASE),
    re.compile(r"\bdatabase migration failed\b", re.IGNORECASE),
    re.compile(r"\bfailed to apply migrations\b", re.IGNORECASE),
    re.compile(r"\balembic\.util\.exc\.commanderror\b", re.IGNORECASE),
)
_MIGRATION_CONTEXT_PATTERN = re.compile(r"\b(?:migration|migrations|alembic)\b", re.IGNORECASE)
_MISSING_RELATION_PATTERN = re.compile(r"\brelation users does not exist\b", re.IGNORECASE)


def health_check_timeout_rule(context: DiagnosisContext) -> DiagnosisMatch | None:
    """Match only the exact predicates used by the original implementation."""

    timed_out = any(log.get("reason") == "timeout" for log in context.error_logs)
    health = context.service_health
    if not (
        context.deployment.get("status") == "failed"
        and context.deployment.get("failed_stage") == "health_check"
        and timed_out
        and health is not None
        and health.get("status") == "unhealthy"
    ):
        return None

    root_cause = "The deployment health check timed out because the target service was unhealthy."
    return DiagnosisMatch(
        root_cause=root_cause,
        answer=f"Deployment {context.deployment_id} failed during its health check. {root_cause}",
        confidence=1.0,
        evidence_sources=("deployment", "logs", "service_health"),
    )


def missing_environment_variable_rule(context: DiagnosisContext) -> DiagnosisMatch | None:
    """Match three explicit missing-variable message forms, regardless of stage."""

    variable_name = None
    for log in context.error_logs:
        message = str(log.get("message", ""))
        for pattern in _MISSING_VARIABLE_PATTERNS:
            match = pattern.search(message)
            if match:
                variable_name = match.group(1).upper()
                break
        if variable_name is not None:
            break
    if variable_name is None:
        return None

    root_cause = (
        f"The deployment failed because required environment variable {variable_name} was missing."
    )
    return DiagnosisMatch(
        root_cause=root_cause,
        answer=f"Deployment {context.deployment_id} failed. {root_cause}",
        confidence=1.0,
        evidence_sources=("deployment", "logs"),
    )


def database_migration_failure_rule(context: DiagnosisContext) -> DiagnosisMatch | None:
    """Match explicit migration failures or missing-relation errors with migration context."""

    messages = tuple(str(log.get("message", "")) for log in context.error_logs)
    direct_match = any(
        pattern.search(message)
        for message in messages
        for pattern in _DIRECT_MIGRATION_PATTERNS
    )
    contextual_match = (
        any(_MIGRATION_CONTEXT_PATTERN.search(message) for message in messages)
        and any(_MISSING_RELATION_PATTERN.search(message) for message in messages)
    )
    if not (direct_match or contextual_match):
        return None

    root_cause = "The deployment failed because a database migration could not be applied."
    return DiagnosisMatch(
        root_cause=root_cause,
        answer=f"Deployment {context.deployment_id} failed. {root_cause}",
        confidence=1.0,
        evidence_sources=("deployment", "logs"),
    )


DIAGNOSIS_RULES: tuple[DiagnosisRule, ...] = (
    health_check_timeout_rule,
    missing_environment_variable_rule,
    database_migration_failure_rule,
)


def evaluate_diagnoses(context: DiagnosisContext) -> DiagnosisResult:
    """Evaluate every rule and retain zero, exactly one, or multiple matches."""

    return DiagnosisResult(
        matches=tuple(match for rule in DIAGNOSIS_RULES if (match := rule(context)) is not None)
    )
