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
from ai_investigation.evidence import EvidenceCollector
from ai_investigation.investigator import DeploymentFailureInvestigator
from ai_investigation.llm_investigator import ModelProviderError, StructuredModel
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


def main(
    argv: Sequence[str] | None = None,
    *,
    structured_model: StructuredModel | None = None,
) -> int:
    root = _repository_root()
    parser = argparse.ArgumentParser(description="Evaluate investigation behavior.")
    parser.add_argument(
        "--investigator",
        choices=("deterministic", "gemini", "both"),
        default="deterministic",
    )
    parser.add_argument("--format", choices=("text", "json"), default="text")
    parser.add_argument("--output", type=Path)
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
        collector, deterministic = _dependencies(args.fixtures)
        report = run_experiment(
            load_scenarios(args.scenarios),
            collector,
            deterministic,
            investigator_mode=args.investigator,
            structured_model=model,
        )
    except (ModelProviderError, ValueError) as error:
        print(f"Error: {error}", file=sys.stderr)
        return 2

    rendered = report_to_json(report) if args.format == "json" else render_text_report(report)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered, encoding="utf-8")
    print(rendered, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
