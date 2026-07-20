"""Local experiment metadata, event recording, persistence, and comparison."""

from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import platform
import secrets
import subprocess
import tempfile
from typing import Callable

from ai_investigation.evaluation.framework import InvestigatorMode
from ai_investigation.evaluation.models import (
    AggregateMetrics,
    EvaluationReport,
    ScenarioRunResult,
)
from ai_investigation.evaluation.comparison import (
    ComparisonReport as ExperimentComparison,
    MetricDelta as MetricComparison,
    compare_experiments,
    render_comparison,
)

EXPERIMENT_SCHEMA_VERSION = 1
EVENT_TYPES = {
    "experiment_started",
    "scenario_started",
    "evidence_collection_started",
    "evidence_collection_completed",
    "deterministic_investigation_started",
    "deterministic_investigation_completed",
    "model_investigation_started",
    "model_investigation_completed",
    "validation_completed",
    "scenario_evaluated",
    "scenario_failed",
    "scenario_completed",
    "experiment_completed",
}
EVENT_STAGES = {
    "experiment",
    "scenario",
    "evidence_collection",
    "investigation",
    "validation",
    "evaluation",
}


@dataclass(frozen=True, slots=True)
class ExperimentMetadata:
    experiment_id: str
    created_at: str
    investigator_mode: InvestigatorMode
    scenario_source: str
    scenario_count: int
    scenario_ids: tuple[str, ...]
    provider: str | None
    model: str | None
    repository_revision: str | None
    python_version: str
    platform: str
    configuration: tuple[tuple[str, str], ...]
    tags: tuple[str, ...]
    notes: str | None
    prompt_version: str | None = None
    response_schema_version: str | None = None


@dataclass(frozen=True, slots=True)
class ExecutionEvent:
    sequence: int
    timestamp: str
    event_type: str
    experiment_id: str
    scenario_id: str | None
    investigator: str | None
    stage: str
    status: str
    duration_ms: float | None
    details: tuple[tuple[str, str], ...]


@dataclass(frozen=True, slots=True)
class TimingSummary:
    evidence_collection_ms: float
    deterministic_investigation_ms: float | None
    model_investigation_ms: float | None
    validation_ms: float | None
    evaluation_ms: float
    scenario_total_ms: float
    experiment_total_ms: float


@dataclass(frozen=True, slots=True)
class ExperimentRecord:
    schema_version: int
    metadata: ExperimentMetadata
    report: EvaluationReport
    timing: TimingSummary
    events: tuple[ExecutionEvent, ...]


class ExperimentPersistenceError(Exception):
    """Raised when a local experiment artifact cannot be stored."""


class EventRecorder:
    """Collect finite execution events for one local experiment."""

    def __init__(
        self,
        experiment_id: str,
        wall_clock: Callable[[], datetime] = lambda: datetime.now(timezone.utc),
    ) -> None:
        self._experiment_id = experiment_id
        self._wall_clock = wall_clock
        self._events: list[ExecutionEvent] = []

    @property
    def events(self) -> tuple[ExecutionEvent, ...]:
        return tuple(self._events)

    def __call__(
        self,
        event_type: str,
        scenario_id: str | None,
        investigator: str | None,
        stage: str,
        status: str,
        duration_ms: float | None,
        details: tuple[tuple[str, str], ...],
    ) -> None:
        if event_type not in EVENT_TYPES:
            raise ValueError(f"Unknown experiment event type: {event_type}.")
        if stage not in EVENT_STAGES:
            raise ValueError(f"Unknown experiment stage: {stage}.")
        timestamp = _utc(self._wall_clock()).isoformat().replace("+00:00", "Z")
        self._events.append(
            ExecutionEvent(
                sequence=len(self._events) + 1,
                timestamp=timestamp,
                event_type=event_type,
                experiment_id=self._experiment_id,
                scenario_id=scenario_id,
                investigator=investigator,
                stage=stage,
                status=status,
                duration_ms=duration_ms,
                details=details,
            )
        )


def create_metadata(
    *,
    investigator_mode: InvestigatorMode,
    scenario_source: str,
    scenario_ids: tuple[str, ...],
    provider: str | None = None,
    model: str | None = None,
    configuration: tuple[tuple[str, str], ...] = (),
    tags: tuple[str, ...] = (),
    notes: str | None = None,
    prompt_version: str | None = None,
    response_schema_version: str | None = None,
    now: Callable[[], datetime] = lambda: datetime.now(timezone.utc),
    revision_resolver: Callable[[], str | None] = lambda: resolve_git_revision(),
    token_factory: Callable[[], str] = lambda: secrets.token_hex(3),
) -> ExperimentMetadata:
    created = _utc(now())
    revision = revision_resolver()
    revision_part = revision[:7].lower() if revision else "nogit"
    suffix = f"{revision_part}-{token_factory().lower()}"
    timestamp = created.strftime("%Y%m%dT%H%M%SZ")
    experiment_id = f"{timestamp}-{investigator_mode}-{suffix}"
    safe_configuration = tuple(
        (key, value)
        for key, value in configuration
        if "key" not in key.lower() and "secret" not in key.lower() and "token" not in key.lower()
    )
    return ExperimentMetadata(
        experiment_id=experiment_id,
        created_at=created.isoformat().replace("+00:00", "Z"),
        investigator_mode=investigator_mode,
        scenario_source=scenario_source,
        scenario_count=len(scenario_ids),
        scenario_ids=scenario_ids,
        provider=provider if investigator_mode != "deterministic" else None,
        model=model if investigator_mode != "deterministic" else None,
        repository_revision=revision,
        python_version=platform.python_version(),
        platform=f"{platform.system()} {platform.machine()}",
        configuration=safe_configuration,
        tags=tags,
        notes=notes,
        prompt_version=(prompt_version if investigator_mode != "deterministic" else None),
        response_schema_version=(
            response_schema_version if investigator_mode != "deterministic" else None
        ),
    )


def build_record(
    metadata: ExperimentMetadata,
    report: EvaluationReport,
    events: tuple[ExecutionEvent, ...],
) -> ExperimentRecord:
    return ExperimentRecord(
        schema_version=EXPERIMENT_SCHEMA_VERSION,
        metadata=metadata,
        report=report,
        timing=_timing_summary(events),
        events=events,
    )


def experiment_to_json(record: ExperimentRecord) -> str:
    return json.dumps(asdict(record), indent=2, sort_keys=True) + "\n"


def events_to_jsonl(events: tuple[ExecutionEvent, ...]) -> str:
    return "".join(json.dumps(asdict(event), sort_keys=True) + "\n" for event in events)


def save_experiment(
    record: ExperimentRecord,
    report_text: str,
    root: Path,
) -> Path:
    target = root / record.metadata.experiment_id
    try:
        root.mkdir(parents=True, exist_ok=True)
        target.mkdir()
        _atomic_write(target / "experiment.json", experiment_to_json(record))
        _atomic_write(target / "report.txt", report_text)
        _atomic_write(target / "events.jsonl", events_to_jsonl(record.events))
    except (OSError, ValueError) as error:
        raise ExperimentPersistenceError(
            f"Could not persist experiment {record.metadata.experiment_id}: {error}"
        ) from error
    return target


def load_experiment(path: Path) -> ExperimentRecord:
    artifact = path / "experiment.json" if path.is_dir() else path
    try:
        value = json.loads(artifact.read_text(encoding="utf-8"))
        return _record_from_dict(value)
    except (OSError, ValueError, KeyError, TypeError) as error:
        raise ExperimentPersistenceError(f"Malformed experiment artifact {artifact}: {error}") from error


def list_experiments(root: Path) -> tuple[tuple[ExperimentRecord, ...], tuple[str, ...]]:
    if not root.exists():
        return (), ()
    records: list[ExperimentRecord] = []
    errors: list[str] = []
    for path in sorted(item for item in root.iterdir() if item.is_dir()):
        try:
            records.append(load_experiment(path))
        except ExperimentPersistenceError as error:
            errors.append(str(error))
    return tuple(records), tuple(errors)


def resolve_git_revision(cwd: Path | None = None) -> str | None:
    try:
        result = subprocess.run(
            ("git", "rev-parse", "HEAD"),
            cwd=cwd,
            check=True,
            capture_output=True,
            text=True,
            timeout=2,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    revision = result.stdout.strip()
    return revision or None


def _utc(value: datetime) -> datetime:
    if value.tzinfo is None:
        raise ValueError("Experiment timestamps must include a timezone.")
    return value.astimezone(timezone.utc)


def _atomic_write(path: Path, content: str) -> None:
    descriptor, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            stream.write(content)
        os.replace(temporary, path)
    except BaseException:
        try:
            os.unlink(temporary)
        except OSError:
            pass
        raise


def _timing_summary(events: tuple[ExecutionEvent, ...]) -> TimingSummary:
    def total(event_type: str, *, required: bool = False) -> float | None:
        durations = tuple(
            event.duration_ms
            for event in events
            if event.event_type == event_type and event.duration_ms is not None
        )
        if not durations and not required:
            return None
        return sum(durations)

    return TimingSummary(
        evidence_collection_ms=total("evidence_collection_completed", required=True) or 0.0,
        deterministic_investigation_ms=total("deterministic_investigation_completed"),
        model_investigation_ms=total("model_investigation_completed"),
        validation_ms=total("validation_completed"),
        evaluation_ms=(total("scenario_evaluated", required=True) or 0.0)
        + (total("scenario_failed", required=True) or 0.0),
        scenario_total_ms=total("scenario_completed", required=True) or 0.0,
        experiment_total_ms=total("experiment_completed", required=True) or 0.0,
    )


def _record_from_dict(value: object) -> ExperimentRecord:
    if not isinstance(value, dict) or value.get("schema_version") != EXPERIMENT_SCHEMA_VERSION:
        raise ValueError("unsupported or missing schema_version")
    metadata_value = value["metadata"]
    report_value = value["report"]
    timing_value = value["timing"]
    events_value = value["events"]
    metadata = ExperimentMetadata(
        **{
            **metadata_value,
            "scenario_ids": tuple(metadata_value["scenario_ids"]),
            "configuration": tuple(tuple(item) for item in metadata_value["configuration"]),
            "tags": tuple(metadata_value["tags"]),
        }
    )
    aggregate = AggregateMetrics(**report_value["aggregate"])
    scenario_results = tuple(
        ScenarioRunResult(
            **{
                **item,
                "expected_sources": tuple(item["expected_sources"]),
                "referenced_sources": tuple(item["referenced_sources"]),
                "missing_sources": tuple(item["missing_sources"]),
                "unexpected_sources": tuple(item["unexpected_sources"]),
            }
        )
        for item in report_value["scenarios"]
    )
    report = EvaluationReport(
        investigator_mode=report_value["investigator_mode"],
        scenarios=scenario_results,
        aggregate=aggregate,
        confidence_disclaimer=report_value["confidence_disclaimer"],
    )
    events = tuple(
        ExecutionEvent(
            **{
                **item,
                "details": tuple(tuple(detail) for detail in item["details"]),
            }
        )
        for item in events_value
    )
    return ExperimentRecord(
        schema_version=value["schema_version"],
        metadata=metadata,
        report=report,
        timing=TimingSummary(**timing_value),
        events=events,
    )
