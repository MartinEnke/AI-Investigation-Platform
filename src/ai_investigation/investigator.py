"""Sequential deployment-failure investigation workflow."""

import re

from ai_investigation.diagnosis import DiagnosisContext, evaluate_diagnoses
from ai_investigation.models import Evidence, InvestigationRequest, InvestigationResult
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
        self._deployments = deployments
        self._logs = logs
        self._service_health = service_health

    def investigate(self, request: InvestigationRequest) -> InvestigationResult:
        if request.deployment_id is None:
            return InvestigationResult(
                answer="I could not investigate because no deployment ID was provided.",
                root_cause=None,
                evidence=(),
                confidence=0.0,
                limitations=("Include an ID such as deploy-1042 in the question.",),
            )

        deployment = self._deployments.get_deployment(request.deployment_id)
        if deployment is None:
            return InvestigationResult(
                answer=f"Deployment {request.deployment_id} was not found.",
                root_cause=None,
                evidence=(),
                confidence=0.0,
                limitations=("No deployment record is available in the local fixtures.",),
            )

        evidence = [
            Evidence(
                source="deployment",
                summary=(
                    f"{request.deployment_id} has status {deployment.get('status', 'unknown')} "
                    f"and failed stage {deployment.get('failed_stage', 'unknown')}."
                ),
            )
        ]
        limitations: list[str] = []

        logs = self._logs.get_logs(request.deployment_id)
        error_logs = tuple(item for item in logs if item.get("level") == "error")
        if error_logs:
            evidence.extend(
                Evidence(source="logs", summary=str(item.get("message", "Error recorded.")))
                for item in error_logs
            )
        else:
            limitations.append("No error log is available for the deployment.")

        service = deployment.get("service")
        health = self._service_health.get_service_health(service) if isinstance(service, str) else None
        if health is not None:
            evidence.append(
                Evidence(
                    source="service_health",
                    summary=(
                        f"{service} was {health.get('status', 'unknown')}: "
                        f"{health.get('detail', 'no detail recorded')}"
                    ),
                )
            )
        else:
            limitations.append("No service-health record is available for the target service.")

        diagnosis = evaluate_diagnoses(
            DiagnosisContext(request.deployment_id, deployment, error_logs, health)
        )
        if diagnosis.match is not None:
            match = diagnosis.match
            selected_evidence = tuple(
                item for item in evidence if item.source in match.evidence_sources
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
                evidence=tuple(evidence),
                confidence=0.25,
                limitations=("Conflicting supported failure patterns matched the available evidence.",),
                decision_trace=diagnosis.decision_trace,
            )

        if not limitations:
            limitations.append("The available evidence does not match a known failure pattern.")
        return InvestigationResult(
            answer=f"The cause of deployment {request.deployment_id}'s failure is inconclusive.",
            root_cause=None,
            evidence=tuple(evidence),
            confidence=0.25,
            limitations=tuple(limitations),
            decision_trace=diagnosis.decision_trace,
        )
