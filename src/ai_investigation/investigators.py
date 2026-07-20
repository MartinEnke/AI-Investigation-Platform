"""Common investigator contract over deterministically collected evidence."""

from dataclasses import dataclass
from typing import Protocol

from ai_investigation.evidence import CollectedEvidence
from ai_investigation.investigator import DeploymentFailureInvestigator
from ai_investigation.llm_investigator import (
    LLMInvestigationFailure,
    LLMInvestigationSuccess,
    LLMInvestigator,
    DEFAULT_PROMPT_VERSION,
    PromptVersion,
    StructuredModel,
    prompt_version_identifier,
)
from ai_investigation.models import InvestigationResult


@dataclass(frozen=True, slots=True)
class InvestigatorIdentity:
    mode: str
    provider: str | None = None
    model: str | None = None
    prompt_version: str | None = None


@dataclass(frozen=True, slots=True)
class InvestigatorExecution:
    status: str
    result: InvestigationResult | None
    diagnosis_id: str | None
    structured_response_valid: bool | None
    evidence_references_valid: bool | None
    errors: tuple[str, ...] = ()


class Investigator(Protocol):
    @property
    def identity(self) -> InvestigatorIdentity:
        """Return stable implementation metadata."""

    def investigate(self, collected: CollectedEvidence) -> InvestigatorExecution:
        """Interpret already-collected evidence without accessing tools."""


class DeterministicInvestigatorAdapter:
    def __init__(self, investigator: DeploymentFailureInvestigator) -> None:
        self._investigator = investigator

    @property
    def identity(self) -> InvestigatorIdentity:
        return InvestigatorIdentity(mode="deterministic")

    def investigate(self, collected: CollectedEvidence) -> InvestigatorExecution:
        result = self._investigator.investigate_evidence(collected)
        trace = result.decision_trace
        diagnosis_id = (
            trace.matched_rule_ids[0]
            if trace is not None
            and trace.outcome == "single_match"
            and len(trace.matched_rule_ids) == 1
            else None
        )
        return InvestigatorExecution(
            status="completed",
            result=result,
            diagnosis_id=diagnosis_id,
            structured_response_valid=None,
            evidence_references_valid=True,
        )


class LLMInvestigatorAdapter:
    def __init__(
        self,
        model: StructuredModel,
        investigator: LLMInvestigator | None = None,
        prompt_version: PromptVersion = DEFAULT_PROMPT_VERSION,
    ) -> None:
        self._model = model
        self._investigator = investigator or LLMInvestigator(model, prompt_version)
        self._prompt_version = prompt_version

    @property
    def identity(self) -> InvestigatorIdentity:
        return InvestigatorIdentity(
            mode="llm",
            provider=self._model.provider_name,
            model=self._model.model_name,
            prompt_version=prompt_version_identifier(self._prompt_version),
        )

    def investigate(self, collected: CollectedEvidence) -> InvestigatorExecution:
        outcome = self._investigator.investigate(collected)
        if isinstance(outcome, LLMInvestigationSuccess):
            return InvestigatorExecution(
                status="completed",
                result=outcome.result,
                diagnosis_id=outcome.decision.diagnosis_id,
                structured_response_valid=True,
                evidence_references_valid=True,
            )
        assert isinstance(outcome, LLMInvestigationFailure)
        return InvestigatorExecution(
            status=outcome.status,
            result=None,
            diagnosis_id=None,
            structured_response_valid=(
                False
                if outcome.status == "invalid_response"
                else True if outcome.status == "invalid_references" else None
            ),
            evidence_references_valid=(
                False if outcome.status == "invalid_references" else None
            ),
            errors=outcome.errors,
        )
