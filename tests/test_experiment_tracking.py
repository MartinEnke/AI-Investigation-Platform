import json
from dataclasses import replace
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

import ai_investigation.evaluate as evaluate_cli
from ai_investigation.evaluation.framework import render_text_report, run_experiment
from ai_investigation.evaluation.loader import load_scenarios
from ai_investigation.evaluation.tracking import (
    EventRecorder,
    ExperimentPersistenceError,
    build_record,
    compare_experiments,
    create_metadata,
    events_to_jsonl,
    list_experiments,
    load_experiment,
    render_comparison,
    save_experiment,
)
from ai_investigation.evidence import EvidenceCollector
from ai_investigation.experiments import main as experiments_main
from ai_investigation.investigator import DeploymentFailureInvestigator
from ai_investigation.llm_investigator import ModelProviderError
from ai_investigation.tools import JsonDeploymentTool, JsonLogTool, JsonServiceHealthTool


FIXED_TIME = datetime(2026, 7, 19, 15, 15, tzinfo=timezone.utc)


class IncrementingClock:
    def __init__(self) -> None:
        self.value = 0.0

    def __call__(self) -> float:
        self.value += 0.001
        return self.value


class FakeModel:
    provider_name = "fake-provider"
    model_name = "fake-model-v1"

    def __init__(self, response: str = "", error: Exception | None = None) -> None:
        self.response = response
        self.error = error

    def generate(self, prompt: str) -> str:
        if self.error is not None:
            raise self.error
        return self.response


def dependencies(fixture_directory: Path):
    tools = (
        JsonDeploymentTool(fixture_directory / "deployments.json"),
        JsonLogTool(fixture_directory / "logs.json"),
        JsonServiceHealthTool(fixture_directory / "service_health.json"),
    )
    return EvidenceCollector(*tools), DeploymentFailureInvestigator(*tools)


def tracked_record(fixture_directory: Path, *, scenario_indexes=(0,), token="abc123"):
    selected = tuple(
        load_scenarios(fixture_directory / "evaluation_scenarios.json")[index]
        for index in scenario_indexes
    )
    metadata = create_metadata(
        investigator_mode="deterministic",
        scenario_source="tests/fixtures/evaluation_scenarios.json",
        scenario_ids=tuple(item.id for item in selected),
        configuration=(("format", "text"),),
        tags=("baseline",),
        now=lambda: FIXED_TIME,
        revision_resolver=lambda: "a1b2c3d4e5",
        token_factory=lambda: token,
    )
    recorder = EventRecorder(metadata.experiment_id, lambda: FIXED_TIME)
    collector, investigator = dependencies(fixture_directory)
    report = run_experiment(
        selected,
        collector,
        investigator,
        clock=IncrementingClock(),
        observer=recorder,
    )
    return build_record(metadata, report, recorder.events)


def test_metadata_is_utc_unique_and_omits_deterministic_provider() -> None:
    tokens = iter(("one", "two"))
    kwargs = {
        "investigator_mode": "deterministic",
        "scenario_source": "scenarios.json",
        "scenario_ids": ("one",),
        "provider": "must-not-persist",
        "model": "must-not-persist",
        "now": lambda: FIXED_TIME.astimezone(timezone(timedelta(hours=2))),
        "revision_resolver": lambda: None,
        "token_factory": lambda: next(tokens),
    }

    first = create_metadata(**kwargs)
    second = create_metadata(**kwargs)

    assert first.created_at == "2026-07-19T15:15:00Z"
    assert first.experiment_id != second.experiment_id
    assert first.provider is None
    assert first.model is None
    assert first.repository_revision is None


def test_metadata_rejects_naive_time_and_filters_secret_configuration() -> None:
    with pytest.raises(ValueError, match="timezone"):
        create_metadata(
            investigator_mode="deterministic",
            scenario_source="scenarios.json",
            scenario_ids=(),
            now=lambda: datetime(2026, 1, 1),
        )

    metadata = create_metadata(
        investigator_mode="gemini",
        scenario_source="scenarios.json",
        scenario_ids=("one",),
        provider="fake-provider",
        model="fake-model",
        configuration=(("format", "json"), ("GEMINI_API_KEY", "secret")),
        now=lambda: FIXED_TIME,
        revision_resolver=lambda: None,
        token_factory=lambda: "safe",
    )

    assert metadata.provider == "fake-provider"
    assert metadata.model == "fake-model"
    assert metadata.configuration == (("format", "json"),)
    assert "secret" not in repr(metadata)


def test_events_are_sequenced_and_capture_stage_timings(fixture_directory: Path) -> None:
    record = tracked_record(fixture_directory)

    assert [event.sequence for event in record.events] == list(
        range(1, len(record.events) + 1)
    )
    assert record.events[0].event_type == "experiment_started"
    assert record.events[-1].event_type == "experiment_completed"
    assert record.timing.evidence_collection_ms > 0
    assert record.timing.deterministic_investigation_ms > 0
    assert record.timing.evaluation_ms > 0
    assert record.timing.scenario_total_ms > 0
    assert record.timing.experiment_total_ms > 0
    parsed = [json.loads(line) for line in events_to_jsonl(record.events).splitlines()]
    assert parsed[0]["sequence"] == 1


def test_persistence_creates_artifacts_round_trips_and_refuses_overwrite(
    fixture_directory: Path,
    tmp_path: Path,
) -> None:
    record = tracked_record(fixture_directory)
    root = tmp_path / "nested" / "runs"

    path = save_experiment(record, render_text_report(record.report), root)

    assert (path / "experiment.json").exists()
    assert (path / "report.txt").exists()
    assert (path / "events.jsonl").exists()
    assert load_experiment(path) == record
    with pytest.raises(ExperimentPersistenceError, match="Could not persist"):
        save_experiment(record, "again", root)


def test_listing_skips_malformed_experiments(fixture_directory: Path, tmp_path: Path) -> None:
    record = tracked_record(fixture_directory)
    save_experiment(record, "report", tmp_path)
    malformed = tmp_path / "malformed"
    malformed.mkdir()
    (malformed / "experiment.json").write_text("not json")

    records, errors = list_experiments(tmp_path)

    assert records == (record,)
    assert len(errors) == 1
    assert "Malformed experiment artifact" in errors[0]


def test_listing_and_inspection_cli(fixture_directory: Path, tmp_path: Path, capsys) -> None:
    record = tracked_record(fixture_directory)
    save_experiment(record, "report", tmp_path)

    assert experiments_main(["--experiment-dir", str(tmp_path), "list"]) == 0
    listed = capsys.readouterr().out
    assert record.metadata.experiment_id in listed
    assert "| deterministic | scenarios=1 |" in listed
    assert "| deterministic | deterministic |" not in listed
    assert "tags=baseline" in listed

    assert experiments_main(
        ["--experiment-dir", str(tmp_path), "show", record.metadata.experiment_id]
    ) == 0
    shown = capsys.readouterr().out
    assert "Diagnosis accuracy: 1/1" in shown
    assert "Experiment total:" in shown
    assert "Events:" not in shown

    assert experiments_main(
        [
            "--experiment-dir",
            str(tmp_path),
            "show",
            record.metadata.experiment_id,
            "--events",
        ]
    ) == 0
    events = capsys.readouterr().out
    assert "1. experiment_started [started]" in events
    assert "1. experiment_started [started] —" not in events
    assert (
        "2. scenario_started [started] — supported-health-check-timeout"
        in events
    )
    assert (
        "4. evidence_collection_completed [completed] — "
        "supported-health-check-timeout — 1.000 ms"
        in events
    )


def test_model_listing_displays_provider_and_model(
    fixture_directory: Path,
    tmp_path: Path,
    capsys,
) -> None:
    record = tracked_record(fixture_directory)
    metadata = replace(
        record.metadata,
        experiment_id="model-experiment",
        investigator_mode="gemini",
        provider="google-gemini",
        model="gemini-2.5-flash",
    )
    save_experiment(replace(record, metadata=metadata), "report", tmp_path)

    assert experiments_main(["--experiment-dir", str(tmp_path), "list"]) == 0

    assert (
        "| gemini | google-gemini/gemini-2.5-flash | scenarios=1 |"
        in capsys.readouterr().out
    )


def test_comparison_handles_metrics_regressions_and_scenario_sets(
    fixture_directory: Path,
) -> None:
    before = tracked_record(fixture_directory, scenario_indexes=(0, 1), token="before")
    after_base = tracked_record(fixture_directory, scenario_indexes=(0,), token="after")
    failed_scenario = replace(
        after_base.report.scenarios[0], semantic_correctness_status="incorrect"
    )
    aggregate = replace(after_base.report.aggregate, correct_diagnoses=0, semantic_failures=1)
    after = replace(
        after_base,
        report=replace(after_base.report, scenarios=(failed_scenario,), aggregate=aggregate),
    )

    comparison = compare_experiments(before, after)

    assert "supported-health-check-timeout" in comparison.regressions
    assert "unknown-deployment" in comparison.only_before
    assert any(
        metric.metric == "structured_response_validity" and not metric.comparable
        for metric in comparison.metrics
    )
    assert "not comparable" in render_comparison(comparison)
    reverse = compare_experiments(after, before)
    assert "supported-health-check-timeout" in reverse.improvements
    assert "unknown-deployment" in reverse.only_after


def test_provider_failure_produces_trackable_events(fixture_directory: Path) -> None:
    scenario = (load_scenarios(fixture_directory / "evaluation_scenarios.json")[0],)
    metadata = create_metadata(
        investigator_mode="gemini",
        scenario_source="scenarios.json",
        scenario_ids=(scenario[0].id,),
        provider="fake-provider",
        model="fake-model-v1",
        now=lambda: FIXED_TIME,
        revision_resolver=lambda: None,
        token_factory=lambda: "failure",
    )
    recorder = EventRecorder(metadata.experiment_id, lambda: FIXED_TIME)
    collector, investigator = dependencies(fixture_directory)

    report = run_experiment(
        scenario,
        collector,
        investigator,
        investigator_mode="gemini",
        structured_model=FakeModel(error=ModelProviderError("offline")),
        observer=recorder,
    )

    assert report.scenarios[0].execution_status == "provider_failure"
    assert any(event.event_type == "scenario_failed" for event in recorder.events)


def test_evaluation_cli_saves_fake_model_experiment(
    fixture_directory: Path,
    tmp_path: Path,
    capsys,
) -> None:
    source = json.loads(
        (fixture_directory / "evaluation_scenarios.json").read_text(encoding="utf-8")
    )
    scenario_path = tmp_path / "scenario.json"
    scenario_path.write_text(json.dumps(source[:1]), encoding="utf-8")
    response = json.dumps(
        {
            "outcome": "diagnosis",
            "diagnosis_id": "health_check_timeout",
            "confidence": 0.9,
            "evidence_references": [1, 2, 3],
            "abstention_reason": None,
        }
    )
    root = tmp_path / "runs"

    exit_code = evaluate_cli.main(
        [
            "--investigator",
            "gemini",
            "--save-experiment",
            "--experiment-dir",
            str(root),
            "--scenarios",
            str(scenario_path),
            "--fixtures",
            str(fixture_directory),
        ],
        structured_model=FakeModel(response),
    )

    assert exit_code == 0
    stored = tuple(root.iterdir())
    assert len(stored) == 1
    record = load_experiment(stored[0])
    assert record.metadata.provider == "fake-provider"
    assert record.metadata.model == "fake-model-v1"
    assert "Experiment ID:" in capsys.readouterr().out
