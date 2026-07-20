"""Command-line entry point for reusable investigation evaluation."""

import argparse
from pathlib import Path
from typing import Sequence
import os
import sys

from ai_investigation.evaluation.framework import (
    render_text_report,
    report_to_json,
    run_experiment,
)
from ai_investigation.evaluation.loader import load_scenarios
from ai_investigation.evaluation.tracking import (
    EventRecorder,
    ExperimentPersistenceError,
    build_record,
    compare_experiments,
    create_metadata,
    load_experiment,
    render_comparison,
    save_experiment,
)
from ai_investigation.evidence import EvidenceCollector
from ai_investigation.investigator import DeploymentFailureInvestigator
from ai_investigation.llm_investigator import (
    ModelProviderError,
    PROMPT_VERSION,
    RESPONSE_SCHEMA_VERSION,
    StructuredModel,
)
from ai_investigation.tools import JsonDeploymentTool, JsonLogTool, JsonServiceHealthTool


def _repository_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _dependencies(
    fixtures: Path,
) -> tuple[EvidenceCollector, DeploymentFailureInvestigator]:
    tools = (
        JsonDeploymentTool(fixtures / "deployments.json"),
        JsonLogTool(fixtures / "logs.json"),
        JsonServiceHealthTool(fixtures / "service_health.json"),
    )
    return EvidenceCollector(*tools), DeploymentFailureInvestigator(*tools)


def _gemini_model() -> StructuredModel:
    from ai_investigation.gemini_model import DEFAULT_GEMINI_MODEL, GeminiStructuredModel

    return GeminiStructuredModel(
        os.environ.get("GEMINI_API_KEY", ""),
        os.environ.get("GEMINI_MODEL", DEFAULT_GEMINI_MODEL),
    )


def _groq_model(
    model_name: str | None = None,
    provider_name: str | None = None,
) -> StructuredModel:
    from ai_investigation.groq_model import DEFAULT_GROQ_MODEL, GroqStructuredModel

    provider = provider_name or os.environ.get("AI_INVESTIGATION_PROVIDER", "groq")
    if provider != "groq":
        raise ValueError(f"Unsupported LLM provider: {provider}.")
    selected_model = (
        model_name
        or os.environ.get("AI_INVESTIGATION_MODEL")
        or DEFAULT_GROQ_MODEL
    )
    return GroqStructuredModel(os.environ.get("GROQ_API_KEY", ""), selected_model)


def main(
    argv: Sequence[str] | None = None,
    *,
    structured_model: StructuredModel | None = None,
) -> int:
    root = _repository_root()
    parser = argparse.ArgumentParser(description="Evaluate investigation behavior.")
    parser.add_argument(
        "--investigator",
        choices=("deterministic", "gemini", "llm", "both"),
        default="deterministic",
    )
    parser.add_argument("--format", choices=("text", "json"), default="text")
    parser.add_argument("--provider", choices=("groq",))
    parser.add_argument("--model")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--save-experiment", action="store_true")
    parser.add_argument("--experiment-dir", type=Path, default=root / "experiments" / "runs")
    parser.add_argument("--tag", action="append", default=[])
    parser.add_argument("--notes")
    parser.add_argument("--compare-to", type=Path)
    parser.add_argument(
        "--scenarios",
        type=Path,
        default=root / "tests" / "fixtures" / "evaluation_scenarios.json",
    )
    parser.add_argument(
        "--fixtures",
        type=Path,
        default=root / "tests" / "fixtures",
    )
    args = parser.parse_args(argv)

    try:
        model = structured_model
        if args.investigator in ("gemini", "both") and model is None:
            model = _gemini_model()
        elif args.investigator == "llm" and model is None:
            if args.provider not in (None, "groq"):
                raise ValueError(f"Unsupported LLM provider: {args.provider}.")
            model = _groq_model(args.model, args.provider)
        collector, deterministic = _dependencies(args.fixtures)
        scenarios = load_scenarios(args.scenarios)
        tracking_enabled = args.save_experiment or args.compare_to is not None
        metadata = (
            create_metadata(
                investigator_mode=args.investigator,
                scenario_source=_portable_path(args.scenarios, root),
                scenario_ids=tuple(scenario.id for scenario in scenarios),
                provider=getattr(model, "provider_name", None),
                model=getattr(model, "model_name", None),
                configuration=tuple(
                    (key, value)
                    for key, value in (
                        ("format", args.format),
                        ("provider", args.provider),
                        ("model", args.model),
                    )
                    if value is not None
                ),
                tags=tuple(args.tag),
                notes=args.notes,
                prompt_version=(
                    PROMPT_VERSION if args.investigator in ("gemini", "both", "llm") else None
                ),
                response_schema_version=(
                    RESPONSE_SCHEMA_VERSION
                    if args.investigator in ("gemini", "both", "llm")
                    else None
                ),
            )
            if tracking_enabled
            else None
        )
        recorder = EventRecorder(metadata.experiment_id) if metadata is not None else None
        report = run_experiment(
            scenarios,
            collector,
            deterministic,
            investigator_mode=args.investigator,
            structured_model=model,
            observer=recorder,
        )
        record = (
            build_record(metadata, report, recorder.events)
            if metadata is not None and recorder is not None
            else None
        )
    except (ModelProviderError, ValueError, ExperimentPersistenceError) as error:
        print(f"Error: {error}", file=sys.stderr)
        return 2

    rendered = report_to_json(report) if args.format == "json" else render_text_report(report)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered, encoding="utf-8")
    print(rendered, end="")
    if args.save_experiment and record is not None:
        try:
            stored = save_experiment(record, render_text_report(report), args.experiment_dir)
        except ExperimentPersistenceError as error:
            print(f"Error: {error}", file=sys.stderr)
            return 3
        print(f"Experiment ID: {record.metadata.experiment_id}")
        print(f"Stored at: {stored}")
    if args.compare_to is not None and record is not None:
        try:
            previous = load_experiment(_resolve_experiment(args.compare_to, args.experiment_dir))
        except ExperimentPersistenceError as error:
            print(f"Error: {error}", file=sys.stderr)
            return 3
        print(render_comparison(compare_experiments(previous, record)), end="")
    return 0


def _portable_path(path: Path, root: Path) -> str:
    try:
        return str(path.resolve().relative_to(root.resolve()))
    except ValueError:
        return path.name


def _resolve_experiment(value: Path, root: Path) -> Path:
    return value if value.exists() else root / value


if __name__ == "__main__":
    raise SystemExit(main())
