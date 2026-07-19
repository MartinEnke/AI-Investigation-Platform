"""Deterministic collection of evidence used by investigation workflows."""

from dataclasses import dataclass

from ai_investigation.models import Evidence, InvestigationRequest
from ai_investigation.tools.protocols import (
    DeploymentTool,
    JsonRecord,
    LogTool,
    ServiceHealthTool,
)


@dataclass(frozen=True, slots=True)
class CollectedEvidence:
    """The immutable input produced before any diagnosis is attempted."""

    request: InvestigationRequest
    deployment: JsonRecord | None
    error_logs: tuple[JsonRecord, ...]
    service_health: JsonRecord | None
    evidence: tuple[Evidence, ...]
    limitations: tuple[str, ...]


class EvidenceCollector:
    """Collect and normalize deployment evidence without interpreting it."""

    def __init__(
        self,
        deployments: DeploymentTool,
        logs: LogTool,
        service_health: ServiceHealthTool,
    ) -> None:
        self._deployments = deployments
        self._logs = logs
        self._service_health = service_health

    def collect(self, request: InvestigationRequest) -> CollectedEvidence:
        if request.deployment_id is None:
            return CollectedEvidence(
                request=request,
                deployment=None,
                error_logs=(),
                service_health=None,
                evidence=(),
                limitations=("Include an ID such as deploy-1042 in the question.",),
            )

        deployment = self._deployments.get_deployment(request.deployment_id)
        if deployment is None:
            return CollectedEvidence(
                request=request,
                deployment=None,
                error_logs=(),
                service_health=None,
                evidence=(),
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
        service_health = (
            self._service_health.get_service_health(service)
            if isinstance(service, str)
            else None
        )
        if service_health is not None:
            evidence.append(
                Evidence(
                    source="service_health",
                    summary=(
                        f"{service} was {service_health.get('status', 'unknown')}: "
                        f"{service_health.get('detail', 'no detail recorded')}"
                    ),
                )
            )
        else:
            limitations.append("No service-health record is available for the target service.")

        return CollectedEvidence(
            request=request,
            deployment=deployment,
            error_logs=error_logs,
            service_health=service_health,
            evidence=tuple(evidence),
            limitations=tuple(limitations),
        )
