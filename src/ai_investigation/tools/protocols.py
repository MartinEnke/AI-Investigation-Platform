"""Small contracts for evidence sources used by the investigator."""

from collections.abc import Mapping, Sequence
from typing import Protocol

JsonRecord = Mapping[str, object]


class DeploymentTool(Protocol):
    def get_deployment(self, deployment_id: str) -> JsonRecord | None:
        """Return one deployment, or None when it is unknown."""


class LogTool(Protocol):
    def get_logs(self, deployment_id: str) -> Sequence[JsonRecord]:
        """Return logs for a deployment in fixture order."""


class ServiceHealthTool(Protocol):
    def get_service_health(self, service: str) -> JsonRecord | None:
        """Return the recorded health of a service, if available."""

