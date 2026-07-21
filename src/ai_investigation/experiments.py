"""List, inspect, and compare locally persisted experiments."""

import argparse
from pathlib import Path
import sys
from typing import Sequence

from ai_investigation.evaluation.comparison import (
    ScenarioChange,
    compare_experiments,
    comparison_to_json,
    render_comparison,
)
from ai_investigation.evaluation.tracking import (
    ExperimentPersistenceError,
    list_experiments,
    load_experiment,
)


def _default_root() -> Path:
    return Path(__file__).resolve().parents[2] / "experiments" / "runs"


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Inspect local evaluation experiments.")
    parser.add_argument("--experiment-dir", type=Path, default=_default_root())
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("list")
    show = commands.add_parser("show")
    show.add_argument("experiment")
    show.add_argument("--events", action="store_true")
    compare = commands.add_parser("compare")
    compare.add_argument("before")
    compare.add_argument("after")
    compare.add_argument("--json", action="store_true")
    compare.add_argument("--fail-on-regression", action="store_true")
    args = parser.parse_args(argv)

    if args.command == "list":
        records, errors = list_experiments(args.experiment_dir)
        for record in records:
            metadata = record.metadata
            aggregate = record.report.aggregate
            provider = "/".join(
                value for value in (metadata.provider, metadata.model) if value
            )
            fields = [
                f"{metadata.experiment_id} | {metadata.created_at} | "
                f"{metadata.investigator_mode}",
                f"scenarios={metadata.scenario_count} | "
                f"diagnosis={aggregate.correct_diagnoses}/{aggregate.diagnosis_cases} | "
                f"abstention={aggregate.correct_abstentions}/{aggregate.abstention_cases} | "
                f"provider_failures={aggregate.provider_failures} | "
                f"tags={','.join(metadata.tags) or '-'}",
            ]
            if provider:
                fields.insert(1, provider)
            print(" | ".join(fields))
        for error in errors:
            print(f"Warning: {error}", file=sys.stderr)
        return 0

    try:
        if args.command == "show":
            path = _resolve(args.experiment, args.experiment_dir)
            record = load_experiment(path)
            print(_render_record(record, path, args.events), end="")
        else:
            before = load_experiment(_resolve(args.before, args.experiment_dir))
            after = load_experiment(_resolve(args.after, args.experiment_dir))
            comparison = compare_experiments(before, after)
            rendered = (
                comparison_to_json(comparison)
                if args.json
                else render_comparison(comparison)
            )
            print(rendered, end="")
            if args.fail_on_regression and any(
                scenario.change is ScenarioChange.REGRESSED
                for scenario in comparison.scenarios
            ):
                return 1
    except ExperimentPersistenceError as error:
        print(f"Error: {error}", file=sys.stderr)
        return 2
    return 0


def _resolve(value: str, root: Path) -> Path:
    path = Path(value)
    return path if path.exists() else root / value


def _render_record(record, path: Path, show_events: bool) -> str:
    metadata = record.metadata
    aggregate = record.report.aggregate
    failed = tuple(
        result.scenario_id
        for result in record.report.scenarios
        if result.semantic_correctness_status != "correct"
        or result.execution_status not in ("completed", "not_evaluated")
    )
    lines = [
        f"Experiment: {metadata.experiment_id}",
        f"Created: {metadata.created_at}",
        f"Investigator: {metadata.investigator_mode}",
        f"Provider: {metadata.provider or 'none'}",
        f"Model: {metadata.model or 'none'}",
        f"Prompt version: {metadata.prompt_version or 'none'}",
        f"Decision policy: {metadata.decision_policy_version or 'none'}",
        f"Scenario source: {metadata.scenario_source}",
        f"Scenarios: {metadata.scenario_count}",
        f"Revision: {metadata.repository_revision or 'unavailable'}",
        f"Tags: {', '.join(metadata.tags) or 'none'}",
        f"Notes: {metadata.notes or 'none'}",
        f"Diagnosis accuracy: {aggregate.correct_diagnoses}/{aggregate.diagnosis_cases}",
        f"Abstention accuracy: {aggregate.correct_abstentions}/{aggregate.abstention_cases}",
        f"Provider failures: {aggregate.provider_failures}",
        f"Semantic failures: {aggregate.semantic_failures}",
        f"Experiment total: {record.timing.experiment_total_ms:.2f} ms",
        f"Failed scenarios: {', '.join(failed) or 'none'}",
        f"Artifacts: {path}",
    ]
    request_delay = dict(metadata.configuration).get("request_delay_seconds")
    if request_delay is not None and float(request_delay) > 0:
        lines.insert(7, f"Request delay: {request_delay} seconds")
    if show_events:
        lines.append("Events:")
        lines.extend(_event_line(event) for event in record.events)
    return "\n".join(lines) + "\n"


def _event_line(event) -> str:
    parts = [f"  {event.sequence}. {event.event_type} [{event.status}]"]
    if event.scenario_id is not None:
        parts.append(event.scenario_id)
    if event.duration_ms is not None:
        parts.append(f"{event.duration_ms:.3f} ms")
    return " — ".join(parts)


if __name__ == "__main__":
    raise SystemExit(main())
