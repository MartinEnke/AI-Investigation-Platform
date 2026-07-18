"""Load transparent local evaluation scenarios from JSON."""

import json
from pathlib import Path

from ai_investigation.evaluation.models import EvaluationScenario


def load_scenarios(path: Path) -> tuple[EvaluationScenario, ...]:
    with path.open(encoding="utf-8") as fixture:
        data = json.load(fixture)
    if not isinstance(data, list):
        raise ValueError(f"Expected a list of evaluation scenarios in {path}")

    scenarios = tuple(_parse_scenario(item, index, path) for index, item in enumerate(data))
    ids = [scenario.id for scenario in scenarios]
    if len(ids) != len(set(ids)):
        raise ValueError(f"Evaluation scenario IDs must be unique in {path}")
    return scenarios


def _parse_scenario(item: object, index: int, path: Path) -> EvaluationScenario:
    if not isinstance(item, dict):
        raise ValueError(f"Scenario {index} in {path} must be an object")

    scenario_id = item.get("id")
    question = item.get("question")
    expected_root_cause = item.get("expected_root_cause")
    expected_inconclusive = item.get("expected_inconclusive")
    expected_sources = item.get("expected_evidence_sources")
    expected_confidence = item.get("expected_confidence")
    expected_limitations = item.get("expected_limitations")

    if not isinstance(scenario_id, str) or not scenario_id:
        raise ValueError(f"Scenario {index} in {path} requires a non-empty string id")
    if not isinstance(question, str) or not question:
        raise ValueError(f"Scenario {scenario_id} requires a non-empty string question")
    if expected_root_cause is not None and not isinstance(expected_root_cause, str):
        raise ValueError(f"Scenario {scenario_id} has an invalid expected_root_cause")
    if not isinstance(expected_inconclusive, bool):
        raise ValueError(f"Scenario {scenario_id} requires a boolean expected_inconclusive")
    if not isinstance(expected_sources, list) or not all(
        isinstance(source, str) for source in expected_sources
    ):
        raise ValueError(f"Scenario {scenario_id} requires a list of evidence source strings")
    if expected_confidence is not None and (
        isinstance(expected_confidence, bool)
        or not isinstance(expected_confidence, (int, float))
    ):
        raise ValueError(f"Scenario {scenario_id} has an invalid expected_confidence")
    if expected_limitations is not None and (
        not isinstance(expected_limitations, list)
        or not all(isinstance(limitation, str) for limitation in expected_limitations)
    ):
        raise ValueError(f"Scenario {scenario_id} has invalid expected_limitations")

    return EvaluationScenario(
        id=scenario_id,
        question=question,
        expected_root_cause=expected_root_cause,
        expected_inconclusive=expected_inconclusive,
        expected_evidence_sources=tuple(expected_sources),
        expected_confidence=(
            float(expected_confidence) if expected_confidence is not None else None
        ),
        expected_limitations=(
            tuple(expected_limitations) if expected_limitations is not None else None
        ),
    )
