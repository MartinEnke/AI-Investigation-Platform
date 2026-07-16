"""Command-line entry point for local investigations."""

import argparse
from pathlib import Path

from ai_investigation.investigator import DeploymentFailureInvestigator, request_from_question
from ai_investigation.models import InvestigationResult
from ai_investigation.tools import JsonDeploymentTool, JsonLogTool, JsonServiceHealthTool


def _fixture_directory() -> Path:
    return Path(__file__).resolve().parents[2] / "tests" / "fixtures"


def build_investigator(fixtures: Path) -> DeploymentFailureInvestigator:
    return DeploymentFailureInvestigator(
        JsonDeploymentTool(fixtures / "deployments.json"),
        JsonLogTool(fixtures / "logs.json"),
        JsonServiceHealthTool(fixtures / "service_health.json"),
    )


def render(result: InvestigationResult) -> str:
    lines = [result.answer, f"Root cause: {result.root_cause or 'Undetermined'}"]
    lines.append(f"Confidence: {result.confidence:.0%}")
    if result.evidence:
        lines.append("Evidence:")
        lines.extend(f"  {index}. [{item.source}] {item.summary}" for index, item in enumerate(result.evidence, 1))
    if result.limitations:
        lines.append("Limitations:")
        lines.extend(f"  - {item}" for item in result.limitations)
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description="Investigate a deployment failure.")
    parser.add_argument("question", help="Question containing a deployment ID")
    args = parser.parse_args()
    investigator = build_investigator(_fixture_directory())
    print(render(investigator.investigate(request_from_question(args.question))))


if __name__ == "__main__":
    main()

