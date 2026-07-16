from pathlib import Path

import pytest

from ai_investigation.investigator import DeploymentFailureInvestigator
from ai_investigation.tools import JsonDeploymentTool, JsonLogTool, JsonServiceHealthTool


@pytest.fixture
def fixture_directory() -> Path:
    return Path(__file__).parent / "fixtures"


@pytest.fixture
def investigator(fixture_directory: Path) -> DeploymentFailureInvestigator:
    return DeploymentFailureInvestigator(
        JsonDeploymentTool(fixture_directory / "deployments.json"),
        JsonLogTool(fixture_directory / "logs.json"),
        JsonServiceHealthTool(fixture_directory / "service_health.json"),
    )

