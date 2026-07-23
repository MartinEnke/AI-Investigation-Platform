"""Versioned operational boundaries for diagnoses available to LLM reasoning."""

import json
from dataclasses import dataclass

from ai_investigation.decision_policy import SUPPORTED_DIAGNOSES


DIAGNOSIS_CATALOGUE_VERSION = "diagnosis-catalogue-v1"


@dataclass(frozen=True, slots=True)
class DiagnosisDefinition:
    diagnosis_id: str
    root_cause_description: str
    qualifying_evidence: tuple[str, ...]
    insufficient_evidence: tuple[str, ...]
    negative_boundaries: tuple[str, ...]
    causal_role: str
    related_diagnosis_distinctions: tuple[str, ...]

    def __post_init__(self) -> None:
        if self.diagnosis_id not in SUPPORTED_DIAGNOSES:
            raise ValueError(f"Unsupported diagnosis: {self.diagnosis_id}.")
        scalar_fields = (self.root_cause_description, self.causal_role)
        collection_fields = (
            self.qualifying_evidence,
            self.insufficient_evidence,
            self.negative_boundaries,
            self.related_diagnosis_distinctions,
        )
        if any(not value.strip() for value in scalar_fields):
            raise ValueError("Diagnosis definition scalar fields must be non-empty.")
        if any(
            not values or any(not value.strip() for value in values)
            for values in collection_fields
        ):
            raise ValueError("Diagnosis definition collections must contain non-empty text.")


DIAGNOSIS_CATALOGUE = (
    DiagnosisDefinition(
        diagnosis_id="health_check_timeout",
        root_cause_description=(
            "The deployment failed because its health-check mechanism timed out while the target "
            "service remained unhealthy or unavailable."
        ),
        qualifying_evidence=(
            "Direct evidence connects the deployment failure to a readiness, liveness, or equivalent "
            "health check timing out or never succeeding.",
            "The health-check failure remains the primary supported failure mode because no supported "
            "upstream cause in the evidence explains it.",
        ),
        insufficient_evidence=(
            "A generic timeout, slow startup, service unavailability, or unhealthy status without a "
            "direct connection to the deployment health-check mechanism.",
        ),
        negative_boundaries=(
            "Do not select this diagnosis when a supported upstream cause directly explains why the "
            "later health check timed out.",
        ),
        causal_role="Primary failure mode when its cause remains unresolved; otherwise a downstream symptom.",
        related_diagnosis_distinctions=(
            "Database contention that blocks startup or migration takes causal precedence when it "
            "explains the subsequent health-check timeout.",
        ),
    ),
    DiagnosisDefinition(
        diagnosis_id="missing_environment_variable",
        root_cause_description=(
            "The deployment failed because a required environment key was absent or unavailable."
        ),
        qualifying_evidence=(
            "Direct evidence says a required environment variable or environment key is missing, "
            "unset, unavailable, not found, or raised a missing-key error such as KeyError.",
        ),
        insufficient_evidence=(
            "A configuration-related failure that does not identify an absent or unavailable "
            "environment key.",
        ),
        negative_boundaries=(
            "The business purpose of a variable does not change the failure mechanism: a missing "
            "DATABASE_URL environment key remains a missing environment variable.",
        ),
        causal_role="Upstream configuration-delivery root cause.",
        related_diagnosis_distinctions=(
            "Do not also select missing_database_configuration merely because the missing environment "
            "variable carries database settings.",
        ),
    ),
    DiagnosisDefinition(
        diagnosis_id="missing_database_configuration",
        root_cause_description=(
            "The deployment failed because required database connection configuration was missing, "
            "invalid, incomplete, or unusable."
        ),
        qualifying_evidence=(
            "Direct evidence identifies absent, invalid, incomplete, or unusable database connection "
            "configuration, such as an unavailable connection string or invalid connection settings.",
        ),
        insufficient_evidence=(
            "Generic SQL or database errors, missing relations or tables, migration failures, lock "
            "contention, or database-themed vocabulary without evidence about connection configuration.",
        ),
        negative_boundaries=(
            "A message such as 'relation users does not exist' does not by itself establish missing "
            "database configuration.",
            "Do not use this label for a missing environment key when the evidence directly establishes "
            "missing_environment_variable.",
        ),
        causal_role="Upstream database connection-configuration root cause.",
        related_diagnosis_distinctions=(
            "A migration execution failure is database_migration_failure unless separate evidence "
            "directly establishes missing or invalid connection configuration.",
            "Database contention is not missing configuration.",
        ),
    ),
    DiagnosisDefinition(
        diagnosis_id="database_migration_failure",
        root_cause_description=(
            "The deployment failed because a database migration or schema update could not be applied."
        ),
        qualifying_evidence=(
            "Direct evidence states that migration execution, migration application, a migration "
            "command, or a schema-update operation failed.",
        ),
        insufficient_evidence=(
            "A generic schema, relation, table, SQL, or database error without migration or schema-update "
            "execution context.",
        ),
        negative_boundaries=(
            "A failed migration does not by itself establish missing database connection configuration.",
            "Do not infer migration failure solely from database vocabulary or an application crash.",
        ),
        causal_role="Upstream migration-execution root cause.",
        related_diagnosis_distinctions=(
            "When evidence shows contention or locking prevented migration progress, "
            "database_contention_blocked_migration is the more direct upstream cause.",
        ),
    ),
    DiagnosisDefinition(
        diagnosis_id="database_contention_blocked_migration",
        root_cause_description=(
            "The deployment migration or startup was blocked by database contention."
        ),
        qualifying_evidence=(
            "Evidence establishes a causal chain in which locking, pool exhaustion, long-running "
            "queries, or blocked database access prevents migration or startup progress.",
        ),
        insufficient_evidence=(
            "A slow deployment, timeout, unavailable service, isolated database error, or lock-related "
            "wording without evidence that contention blocked migration or startup progress.",
        ),
        negative_boundaries=(
            "Do not select this diagnosis when contention is merely possible but the causal link to "
            "blocked migration or startup is absent.",
        ),
        causal_role="Upstream resource-contention root cause that can produce downstream failures.",
        related_diagnosis_distinctions=(
            "A later startup delay, service outage, or health-check timeout is a downstream symptom "
            "when database contention already explains it.",
            "A migration that fails for a non-contention reason remains database_migration_failure.",
        ),
    ),
)


def render_diagnosis_catalogue() -> str:
    """Return a deterministic JSON representation for prompt construction."""

    value = {
        "version": DIAGNOSIS_CATALOGUE_VERSION,
        "diagnoses": [
            {
                "id": definition.diagnosis_id,
                "root_cause_description": definition.root_cause_description,
                "qualifying_evidence": definition.qualifying_evidence,
                "insufficient_evidence": definition.insufficient_evidence,
                "negative_boundaries": definition.negative_boundaries,
                "causal_role": definition.causal_role,
                "related_diagnosis_distinctions": (
                    definition.related_diagnosis_distinctions
                ),
            }
            for definition in DIAGNOSIS_CATALOGUE
        ],
    }
    return json.dumps(value, sort_keys=True, separators=(",", ":"))


_catalogue_ids = tuple(item.diagnosis_id for item in DIAGNOSIS_CATALOGUE)
if len(_catalogue_ids) != len(set(_catalogue_ids)):
    raise ValueError("Diagnosis catalogue IDs must be unique.")
if set(_catalogue_ids) != SUPPORTED_DIAGNOSES:
    raise ValueError("Diagnosis catalogue must define every supported diagnosis exactly once.")
