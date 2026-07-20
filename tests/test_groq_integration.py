import os
from pathlib import Path

import pytest

from ai_investigation.evidence import EvidenceCollector
from ai_investigation.groq_model import DEFAULT_GROQ_MODEL, GroqStructuredModel
from ai_investigation.investigator import request_from_question
from ai_investigation.investigators import LLMInvestigatorAdapter
from ai_investigation.tools import JsonDeploymentTool, JsonLogTool, JsonServiceHealthTool


pytestmark = pytest.mark.llm_integration


@pytest.mark.skipif(
    os.environ.get("RUN_LLM_INTEGRATION_TESTS") != "1"
    or not os.environ.get("GROQ_API_KEY"),
    reason="real Groq integration requires explicit opt-in and GROQ_API_KEY",
)
def test_real_groq_returns_a_structurally_valid_referenced_result(
    fixture_directory: Path,
) -> None:
    tools = (
        JsonDeploymentTool(fixture_directory / "deployments.json"),
        JsonLogTool(fixture_directory / "logs.json"),
        JsonServiceHealthTool(fixture_directory / "service_health.json"),
    )
    collected = EvidenceCollector(*tools).collect(
        request_from_question("Why did deployment deploy-1042 fail?")
    )
    model = GroqStructuredModel(
        os.environ.get("GROQ_API_KEY", ""),
        os.environ.get("AI_INVESTIGATION_MODEL", DEFAULT_GROQ_MODEL),
    )

    execution = LLMInvestigatorAdapter(model).investigate(collected)

    assert execution.status == "completed"
    assert execution.structured_response_valid is True
    assert execution.evidence_references_valid is True
    assert execution.result is not None
