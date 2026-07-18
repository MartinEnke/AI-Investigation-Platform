"""Domain models shared by the investigation workflow."""

from dataclasses import dataclass
from typing import Literal


@dataclass(frozen=True, slots=True)
class RuleConditionResult:
    """The deterministic result of one named rule condition."""

    condition: str
    matched: bool


@dataclass(frozen=True, slots=True)
class RuleEvaluation:
    """The ordered condition results and outcome for one diagnosis rule."""

    rule_id: str
    matched: bool
    conditions: tuple[RuleConditionResult, ...]


@dataclass(frozen=True, slots=True)
class DecisionTrace:
    """Structured facts explaining the aggregate diagnosis decision."""

    evaluated_rules: tuple[RuleEvaluation, ...]
    matched_rule_ids: tuple[str, ...]
    outcome: Literal["single_match", "no_match", "multiple_matches"]


@dataclass(frozen=True, slots=True)
class InvestigationRequest:
    """A user's question and the deployment identifier extracted from it."""

    question: str
    deployment_id: str | None


@dataclass(frozen=True, slots=True)
class Evidence:
    """One ordered fact used to reach an investigation conclusion."""

    source: str
    summary: str


@dataclass(frozen=True, slots=True)
class InvestigationResult:
    """The deterministic outcome of an investigation."""

    answer: str
    root_cause: str | None
    evidence: tuple[Evidence, ...]
    confidence: float
    limitations: tuple[str, ...]
    decision_trace: DecisionTrace | None = None
