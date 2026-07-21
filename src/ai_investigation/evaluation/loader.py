"""Load transparent local evaluation scenarios from JSON."""

import json
from pathlib import Path

from ai_investigation.evaluation.models import EvaluationScenario

DIAGNOSIS_IDS = {
    "health_check_timeout",
    "missing_environment_variable",
    "database_migration_failure",
    "missing_database_configuration",
    "database_contention_blocked_migration",
}
EXECUTION_STATUSES = {
    "completed",
    "not_evaluated",
    "invalid_response",
    "invalid_references",
    "refused",
    "provider_failure",
    "adapter_failure",
}
ROBUSTNESS_CATEGORIES = {
    "wording_variation",
    "evidence_reordering",
    "distractor",
    "missing_evidence",
    "conflicting_evidence",
    "unsupported_cause",
    "supported_generalization",
    "duplicate_evidence",
}


def load_scenarios(path: Path) -> tuple[EvaluationScenario, ...]:
    with path.open(encoding="utf-8") as fixture:
        data = json.load(fixture)
    included: tuple[EvaluationScenario, ...] = ()
    if isinstance(data, dict):
        include = data.get("include")
        data = data.get("scenarios")
        if not isinstance(include, str) or not include:
            raise ValueError(f"Scenario collection in {path} requires a non-empty include")
        included = load_scenarios(path.parent / include)
    if not isinstance(data, list):
        raise ValueError(f"Expected a list of evaluation scenarios in {path}")

    scenarios = included + tuple(
        _parse_scenario(item, index, path) for index, item in enumerate(data)
    )
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
    expected_decision_outcome = item.get("expected_decision_outcome")
    expected_matched_rule_ids = item.get("expected_matched_rule_ids")
    expected_diagnosis_id = item.get("expected_diagnosis_id")
    expected_should_abstain = item.get("expected_should_abstain")
    expected_execution_status = item.get("expected_execution_status")
    robustness_categories = item.get("robustness_categories", [])

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
    if expected_decision_outcome is not None and expected_decision_outcome not in (
        "single_match",
        "no_match",
        "multiple_matches",
    ):
        raise ValueError(f"Scenario {scenario_id} has an invalid expected_decision_outcome")
    if expected_matched_rule_ids is not None and (
        not isinstance(expected_matched_rule_ids, list)
        or not all(isinstance(rule_id, str) for rule_id in expected_matched_rule_ids)
    ):
        raise ValueError(f"Scenario {scenario_id} has invalid expected_matched_rule_ids")
    if expected_diagnosis_id is not None and expected_diagnosis_id not in DIAGNOSIS_IDS:
        raise ValueError(f"Scenario {scenario_id} has invalid expected_diagnosis_id")
    if expected_should_abstain is not None and not isinstance(
        expected_should_abstain, bool
    ):
        raise ValueError(f"Scenario {scenario_id} has invalid expected_should_abstain")
    if (
        expected_execution_status is not None
        and expected_execution_status not in EXECUTION_STATUSES
    ):
        raise ValueError(f"Scenario {scenario_id} has invalid expected_execution_status")
    if not isinstance(robustness_categories, list) or not all(
        isinstance(category, str) and category in ROBUSTNESS_CATEGORIES
        for category in robustness_categories
    ):
        raise ValueError(f"Scenario {scenario_id} has invalid robustness_categories")
    if len(robustness_categories) != len(set(robustness_categories)):
        raise ValueError(f"Scenario {scenario_id} has duplicate robustness_categories")

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
        expected_decision_outcome=expected_decision_outcome,
        expected_matched_rule_ids=(
            tuple(expected_matched_rule_ids)
            if expected_matched_rule_ids is not None
            else None
        ),
        expected_diagnosis_id=expected_diagnosis_id,
        expected_should_abstain=expected_should_abstain,
        expected_execution_status=expected_execution_status,
        robustness_categories=tuple(robustness_categories),
    )
