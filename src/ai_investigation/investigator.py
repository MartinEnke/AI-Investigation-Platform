"""Sequential deployment-failure investigation workflow."""

import re

from ai_investigation.diagnosis import DiagnosisContext, evaluate_diagnoses
from ai_investigation.evidence import CollectedEvidence, EvidenceCollector
from ai_investigation.models import InvestigationRequest, InvestigationResult
from ai_investigation.tools.protocols import DeploymentTool, LogTool, ServiceHealthTool

DEPLOYMENT_ID_PATTERN = re.compile(r"\bdeploy-\d+\b", re.IGNORECASE)


def request_from_question(question: str) -> InvestigationRequest:
    match = DEPLOYMENT_ID_PATTERN.search(question)
    return InvestigationRequest(
        question=question,
        deployment_id=match.group(0).lower() if match else None,
    )


class DeploymentFailureInvestigator:
    def __init__(
        self,
        deployments: DeploymentTool,
        logs: LogTool,
        service_health: ServiceHealthTool,
    ) -> None:
        self._evidence_collector = EvidenceCollector(deployments, logs, service_health)

    def investigate(self, request: InvestigationRequest) -> InvestigationResult:
        return self.investigate_evidence(self._evidence_collector.collect(request))

    def investigate_evidence(self, collected: CollectedEvidence) -> InvestigationResult:
        request = collected.request
        if request.deployment_id is None:
            return InvestigationResult(
                answer="I could not investigate because no deployment ID was provided.",
                root_cause=None,
                evidence=(),
                confidence=0.0,
                limitations=collected.limitations,
            )

        deployment = collected.deployment
        if deployment is None:
            return InvestigationResult(
                answer=f"Deployment {request.deployment_id} was not found.",
                root_cause=None,
                evidence=(),
                confidence=0.0,
                limitations=collected.limitations,
            )

        diagnosis = evaluate_diagnoses(
            DiagnosisContext(
                request.deployment_id,
                deployment,
                collected.error_logs,
                collected.service_health,
            )
        )
        if diagnosis.match is not None:
            match = diagnosis.match
            selected_evidence = tuple(
                item for item in collected.evidence if item.source in match.evidence_sources
            )
            return InvestigationResult(
                answer=match.answer,
                root_cause=match.root_cause,
                evidence=selected_evidence,
                confidence=match.confidence,
                limitations=(),
                decision_trace=diagnosis.decision_trace,
            )

        if diagnosis.has_conflict:
            return InvestigationResult(
                answer=f"The cause of deployment {request.deployment_id}'s failure is inconclusive.",
                root_cause=None,
                evidence=collected.evidence,
                confidence=0.25,
                limitations=("Conflicting supported failure patterns matched the available evidence.",),
                decision_trace=diagnosis.decision_trace,
            )

        limitations = collected.limitations or (
            "The available evidence does not match a known failure pattern.",
        )
        return InvestigationResult(
            answer=f"The cause of deployment {request.deployment_id}'s failure is inconclusive.",
            root_cause=None,
            evidence=collected.evidence,
            confidence=0.25,
            limitations=limitations,
            decision_trace=diagnosis.decision_trace,
        )
