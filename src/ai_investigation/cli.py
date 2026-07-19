"""Command-line entry point for local investigations."""

import argparse
import os
from pathlib import Path
from typing import Sequence

from ai_investigation.evidence import CollectedEvidence, EvidenceCollector
from ai_investigation.investigator import DeploymentFailureInvestigator, request_from_question
from ai_investigation.llm_investigator import (
    LLMInvestigationOutcome,
    LLMInvestigationSuccess,
    LLMInvestigator,
    ModelProviderError,
    StructuredModel,
)
from ai_investigation.models import InvestigationResult
from ai_investigation.tools import JsonDeploymentTool, JsonLogTool, JsonServiceHealthTool

INVESTIGATORS = ("deterministic", "gemini", "both")


def _fixture_directory() -> Path:
    return Path(__file__).resolve().parents[2] / "tests" / "fixtures"


def _tools(fixtures: Path) -> tuple[JsonDeploymentTool, JsonLogTool, JsonServiceHealthTool]:
    return (
        JsonDeploymentTool(fixtures / "deployments.json"),
        JsonLogTool(fixtures / "logs.json"),
        JsonServiceHealthTool(fixtures / "service_health.json"),
    )


def build_investigator(fixtures: Path) -> DeploymentFailureInvestigator:
    """Build the existing deterministic investigator from local fixtures."""

    return DeploymentFailureInvestigator(*_tools(fixtures))


def build_dependencies(
    fixtures: Path,
) -> tuple[EvidenceCollector, DeploymentFailureInvestigator]:
    """Build the shared collector and deterministic interpreter."""

    tools = _tools(fixtures)
    return EvidenceCollector(*tools), DeploymentFailureInvestigator(*tools)


def render(result: InvestigationResult) -> str:
    """Preserve the original concise result rendering API."""

    lines = [result.answer, f"Root cause: {result.root_cause or 'Undetermined'}"]
    lines.append(f"Confidence: {result.confidence:.0%}")
    if result.evidence:
        lines.append("Evidence:")
        lines.extend(
            f"  {index}. [{item.source}] {item.summary}"
            for index, item in enumerate(result.evidence, 1)
        )
    if result.limitations:
        lines.append("Limitations:")
        lines.extend(f"  - {item}" for item in result.limitations)
    return "\n".join(lines)


def run_investigation(
    question: str,
    investigator_name: str,
    collector: EvidenceCollector,
    deterministic_investigator: DeploymentFailureInvestigator,
    *,
    structured_model: StructuredModel | None = None,
) -> str:
    """Run one investigation from shared evidence and return terminal text."""

    if investigator_name not in INVESTIGATORS:
        raise ValueError(f"Unknown investigator: {investigator_name}.")

    collected = collector.collect(request_from_question(question))
    sections = [
        _section(
            "Investigation",
            (f"Question: {question}", f"Investigator: {investigator_name}"),
        ),
        _format_collected_evidence(collected),
    ]

    if investigator_name in ("deterministic", "both"):
        result = deterministic_investigator.investigate_evidence(collected)
        sections.extend(_format_deterministic_result(result, collected))
    if investigator_name in ("gemini", "both"):
        if structured_model is None:
            raise ValueError("A structured model is required for the Gemini investigator.")
        outcome = LLMInvestigator(structured_model).investigate(collected)
        sections.extend(_format_llm_outcome(outcome))

    return "\n\n".join(sections)


def _format_collected_evidence(collected: CollectedEvidence) -> str:
    lines = [
        f"{index}. [{item.source}] {item.summary}"
        for index, item in enumerate(collected.evidence, start=1)
    ]
    if not lines:
        lines.append("No evidence was collected.")
    if collected.limitations:
        lines.append("Collection limitations:")
        lines.extend(f"- {limitation}" for limitation in collected.limitations)
    return _section("Collected Evidence", lines)


def _format_deterministic_result(
    result: InvestigationResult,
    collected: CollectedEvidence,
) -> tuple[str, ...]:
    result_lines = _result_lines(result)
    trace = result.decision_trace
    if trace is not None:
        result_lines.extend(
            (
                f"Decision outcome: {trace.outcome}",
                "Matched rules: " + (", ".join(trace.matched_rule_ids) or "none"),
                "Rule evaluations:",
            )
        )
        for evaluation in trace.evaluated_rules:
            conditions = ", ".join(
                f"{condition.condition}={str(condition.matched).lower()}"
                for condition in evaluation.conditions
            )
            result_lines.append(
                f"- {evaluation.rule_id}: matched={str(evaluation.matched).lower()} ({conditions})"
            )

    references = _result_evidence_references(result, collected)
    return (
        _section("Deterministic Conclusion", result_lines),
        _section(
            "Deterministic Evidence References",
            (_references_text(references),),
        ),
        _section(
            "Deterministic Validation",
            ("Deterministic rule evaluation completed.",),
        ),
    )


def _format_llm_outcome(outcome: LLMInvestigationOutcome) -> tuple[str, ...]:
    if isinstance(outcome, LLMInvestigationSuccess):
        decision = outcome.decision
        result_lines = _result_lines(outcome.result)
        result_lines.extend(
            (
                f"Abstained: {'yes' if decision.outcome == 'abstain' else 'no'}",
                f"Abstention reason: {decision.abstention_reason or 'none'}",
                "Confidence note: model confidence is uncalibrated and should not be treated as correctness.",
            )
        )
        return (
            _section("Model Interpretation", result_lines),
            _section(
                "Model Evidence References",
                (_references_text(decision.evidence_references),),
            ),
            _section(
                "Model Validation",
                (
                    "Structured response: valid",
                    "Evidence references: valid",
                    "Semantic correctness: not evaluated",
                ),
            ),
        )

    validation = ["Execution status: " + outcome.status]
    if outcome.status == "invalid_response":
        validation.append("Structured response: invalid")
    elif outcome.status == "invalid_references":
        validation.extend(("Structured response: valid", "Evidence references: invalid"))
    elif outcome.status == "not_evaluated":
        validation.append("Model evaluation was not attempted.")
    return (
        _section("Model Validation", validation),
        _section("Error", outcome.errors),
    )


def _result_lines(result: InvestigationResult) -> list[str]:
    lines = [
        f"Answer: {result.answer}",
        f"Diagnosis: {result.root_cause or 'Undetermined'}",
        f"Confidence: {result.confidence:.2f}",
    ]
    if result.limitations:
        lines.append("Limitations:")
        lines.extend(f"- {limitation}" for limitation in result.limitations)
    return lines


def _result_evidence_references(
    result: InvestigationResult,
    collected: CollectedEvidence,
) -> tuple[int, ...]:
    selected_ids = {id(item) for item in result.evidence}
    return tuple(
        index
        for index, item in enumerate(collected.evidence, start=1)
        if id(item) in selected_ids
    )


def _references_text(references: Sequence[int]) -> str:
    return ", ".join(str(reference) for reference in references) or "none"


def _section(title: str, lines: Sequence[str]) -> str:
    return "\n".join((title, "-" * len(title), *lines))


def _interactive_inputs() -> tuple[str, str]:
    print("AI Investigation Platform")
    while True:
        selection = input("Investigator (deterministic/gemini/both): ").strip().lower()
        if selection in INVESTIGATORS:
            break
        print("Choose 'deterministic', 'gemini', or 'both'.")
    return selection, input("Investigation question: ").strip()


def _gemini_model() -> StructuredModel:
    from ai_investigation.gemini_model import DEFAULT_GEMINI_MODEL, GeminiStructuredModel

    api_key = os.environ.get("GEMINI_API_KEY", "")
    model_name = os.environ.get("GEMINI_MODEL", DEFAULT_GEMINI_MODEL)
    return GeminiStructuredModel(api_key, model_name)


class _LazyGeminiModel:
    """Construct Gemini only if the LLM investigator requests generation."""

    def generate(self, prompt: str) -> str:
        return _gemini_model().generate(prompt)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Investigate a deployment failure.")
    parser.add_argument(
        "--investigator",
        choices=INVESTIGATORS,
        default=None,
        help="Reasoning path to use (default: deterministic)",
    )
    parser.add_argument("question", nargs="?", help="Question containing a deployment ID")
    args = parser.parse_args(argv)

    if args.question is None:
        investigator_name, question = _interactive_inputs()
    else:
        investigator_name = args.investigator or "deterministic"
        question = args.question

    collector, deterministic = build_dependencies(_fixture_directory())
    try:
        model = _LazyGeminiModel() if investigator_name in ("gemini", "both") else None
        print(
            run_investigation(
                question,
                investigator_name,
                collector,
                deterministic,
                structured_model=model,
            )
        )
    except (ModelProviderError, ValueError) as error:
        print(_section("Error", (str(error),)))
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
