"""Execute deterministic scenarios and report exact expectation matches."""

import argparse
from collections.abc import Iterable
from pathlib import Path
from typing import TextIO
import sys

from ai_investigation.evaluation.loader import load_scenarios
from ai_investigation.evaluation.models import EvaluationResult, EvaluationScenario
from ai_investigation.investigator import DeploymentFailureInvestigator, request_from_question
from ai_investigation.tools import JsonDeploymentTool, JsonLogTool, JsonServiceHealthTool


def evaluate_scenario(
    scenario: EvaluationScenario,
    investigator: DeploymentFailureInvestigator,
) -> EvaluationResult:
    actual = investigator.investigate(request_from_question(scenario.question))
    actual_inconclusive = actual.root_cause is None
    actual_sources = tuple(item.source for item in actual.evidence)
    actual_decision_outcome = (
        actual.decision_trace.outcome if actual.decision_trace is not None else None
    )
    actual_matched_rule_ids = (
        actual.decision_trace.matched_rule_ids if actual.decision_trace is not None else None
    )

    root_cause_matches = actual.root_cause == scenario.expected_root_cause
    inconclusive_matches = actual_inconclusive == scenario.expected_inconclusive
    evidence_sources_match = actual_sources == scenario.expected_evidence_sources
    confidence_matches = (
        None
        if scenario.expected_confidence is None
        else actual.confidence == scenario.expected_confidence
    )
    limitations_match = (
        None
        if scenario.expected_limitations is None
        else actual.limitations == scenario.expected_limitations
    )
    decision_outcome_matches = (
        None
        if scenario.expected_decision_outcome is None
        else actual_decision_outcome == scenario.expected_decision_outcome
    )
    matched_rule_ids_match = (
        None
        if scenario.expected_matched_rule_ids is None
        else actual_matched_rule_ids == scenario.expected_matched_rule_ids
    )

    failures: list[str] = []
    if not root_cause_matches:
        failures.append(
            f"root cause: expected {scenario.expected_root_cause!r}, got {actual.root_cause!r}"
        )
    if not inconclusive_matches:
        failures.append(
            f"inconclusive: expected {scenario.expected_inconclusive}, got {actual_inconclusive}"
        )
    if not evidence_sources_match:
        failures.append(
            f"evidence sources: expected {scenario.expected_evidence_sources!r}, got {actual_sources!r}"
        )
    if confidence_matches is False:
        failures.append(
            f"confidence: expected {scenario.expected_confidence!r}, got {actual.confidence!r}"
        )
    if limitations_match is False:
        failures.append(
            f"limitations: expected {scenario.expected_limitations!r}, got {actual.limitations!r}"
        )
    if decision_outcome_matches is False:
        failures.append(
            "decision outcome: "
            f"expected {scenario.expected_decision_outcome!r}, got {actual_decision_outcome!r}"
        )
    if matched_rule_ids_match is False:
        failures.append(
            "matched rule IDs: "
            f"expected {scenario.expected_matched_rule_ids!r}, got {actual_matched_rule_ids!r}"
        )

    return EvaluationResult(
        scenario_id=scenario.id,
        passed=not failures,
        root_cause_matches=root_cause_matches,
        inconclusive_matches=inconclusive_matches,
        evidence_sources_match=evidence_sources_match,
        confidence_matches=confidence_matches,
        limitations_match=limitations_match,
        decision_outcome_matches=decision_outcome_matches,
        matched_rule_ids_match=matched_rule_ids_match,
        actual_root_cause=actual.root_cause,
        actual_inconclusive=actual_inconclusive,
        actual_evidence_sources=actual_sources,
        expected_confidence=scenario.expected_confidence,
        actual_confidence=actual.confidence,
        expected_limitations=scenario.expected_limitations,
        actual_limitations=actual.limitations,
        expected_decision_outcome=scenario.expected_decision_outcome,
        actual_decision_outcome=actual_decision_outcome,
        expected_matched_rule_ids=scenario.expected_matched_rule_ids,
        actual_matched_rule_ids=actual_matched_rule_ids,
        failures=tuple(failures),
    )


def run_evaluation(
    scenarios: Iterable[EvaluationScenario],
    investigator: DeploymentFailureInvestigator,
) -> tuple[EvaluationResult, ...]:
    return tuple(evaluate_scenario(scenario, investigator) for scenario in scenarios)


def print_report(results: Iterable[EvaluationResult], stream: TextIO = sys.stdout) -> None:
    collected = tuple(results)
    for result in collected:
        status = "PASS" if result.passed else "FAIL"
        print(f"{status} {result.scenario_id}", file=stream)
        for failure in result.failures:
            print(f"  - {failure}", file=stream)

    passed = sum(result.passed for result in collected)
    failed = len(collected) - passed
    print(f"Summary: {passed} passed, {failed} failed", file=stream)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run deterministic investigation evaluations.")
    parser.add_argument("scenarios", type=Path, help="Path to evaluation_scenarios.json")
    parser.add_argument("--fixtures", type=Path, required=True, help="Directory containing evidence fixtures")
    args = parser.parse_args()

    investigator = DeploymentFailureInvestigator(
        JsonDeploymentTool(args.fixtures / "deployments.json"),
        JsonLogTool(args.fixtures / "logs.json"),
        JsonServiceHealthTool(args.fixtures / "service_health.json"),
    )
    results = run_evaluation(load_scenarios(args.scenarios), investigator)
    print_report(results)


if __name__ == "__main__":
    main()
