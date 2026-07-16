from pathlib import Path

from ai_investigation.tools import JsonDeploymentTool, JsonLogTool, JsonServiceHealthTool


def test_fixture_loading(fixture_directory: Path) -> None:
    deployment = JsonDeploymentTool(fixture_directory / "deployments.json").get_deployment("deploy-1042")
    logs = JsonLogTool(fixture_directory / "logs.json").get_logs("deploy-1042")
    health = JsonServiceHealthTool(fixture_directory / "service_health.json").get_service_health("checkout-api")

    assert deployment is not None and deployment["status"] == "failed"
    assert [record["level"] for record in logs] == ["info", "error"]
    assert health is not None and health["status"] == "unhealthy"

