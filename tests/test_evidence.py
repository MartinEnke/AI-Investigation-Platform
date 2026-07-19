from dataclasses import FrozenInstanceError
from pathlib import Path

import pytest

from ai_investigation.evidence import EvidenceCollector
from ai_investigation.investigator import DeploymentFailureInvestigator, request_from_question
from ai_investigation.models import Evidence
from ai_investigation.tools import JsonDeploymentTool, JsonLogTool, JsonServiceHealthTool


def collector(fixture_directory: Path) -> EvidenceCollector:
    return EvidenceCollector(
        JsonDeploymentTool(fixture_directory / "deployments.json"),
        JsonLogTool(fixture_directory / "logs.json"),
        JsonServiceHealthTool(fixture_directory / "service_health.json"),
    )


def test_collected_evidence_preserves_existing_values_and_order(
    fixture_directory: Path,
) -> None:
    collected = collector(fixture_directory).collect(
        request_from_question("Why did deployment deploy-1042 fail?")
    )

    assert collected.deployment is not None
    assert collected.deployment["id"] == "deploy-1042"
    assert collected.error_logs == (
        {
            "deployment_id": "deploy-1042",
            "level": "error",
            "reason": "timeout",
            "message": "Health check timed out after 30 seconds.",
        },
    )
    assert collected.service_health is not None
    assert collected.service_health["service"] == "checkout-api"
    assert collected.evidence == (
        Evidence(
            source="deployment",
            summary="deploy-1042 has status failed and failed stage health_check.",
        ),
        Evidence(source="logs", summary="Health check timed out after 30 seconds."),
        Evidence(
            source="service_health",
            summary="checkout-api was unhealthy: Readiness endpoint returned HTTP 503.",
        ),
    )
    assert collected.limitations == ()


def test_precollected_and_direct_investigation_results_are_identical(
    fixture_directory: Path,
    investigator: DeploymentFailureInvestigator,
) -> None:
    request = request_from_question("Why did deployment deploy-1042 fail?")
    collected = collector(fixture_directory).collect(request)

    assert investigator.investigate_evidence(collected) == investigator.investigate(request)


def test_collection_calls_each_applicable_tool_once() -> None:
    calls = {"deployments": 0, "logs": 0, "health": 0}

    class Deployments:
        def get_deployment(self, deployment_id: str):
            calls["deployments"] += 1
            return {
                "id": deployment_id,
                "service": "api",
                "status": "failed",
                "failed_stage": "startup",
            }

    class Logs:
        def get_logs(self, deployment_id: str):
            calls["logs"] += 1
            return [
                {"level": "info", "message": "Started."},
                {"level": "error", "message": "Failed."},
            ]

    class Health:
        def get_service_health(self, service: str):
            calls["health"] += 1
            return {"status": "healthy", "detail": "HTTP 200."}

    collected = EvidenceCollector(Deployments(), Logs(), Health()).collect(
        request_from_question("Why did deployment deploy-2000 fail?")
    )

    assert calls == {"deployments": 1, "logs": 1, "health": 1}
    assert tuple(item["message"] for item in collected.error_logs) == ("Failed.",)


def test_missing_deployment_id_skips_all_tools_and_collection_is_immutable() -> None:
    class UnusedTool:
        def __getattr__(self, name: str):
            raise AssertionError(f"Tool should not be used: {name}")

    collected = EvidenceCollector(UnusedTool(), UnusedTool(), UnusedTool()).collect(
        request_from_question("Why did the deployment fail?")
    )

    assert collected.evidence == ()
    assert collected.limitations == ("Include an ID such as deploy-1042 in the question.",)
    with pytest.raises(FrozenInstanceError):
        collected.evidence = (Evidence("logs", "unexpected"),)  # type: ignore[misc]
