"""Domain models shared by the investigation workflow."""

from dataclasses import dataclass


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

