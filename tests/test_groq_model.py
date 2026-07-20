import json
from io import BytesIO
from pathlib import Path
import socket
from urllib.error import HTTPError

import pytest

from ai_investigation.groq_model import (
    GROQ_USER_AGENT,
    GroqAuthenticationError,
    GroqRateLimitError,
    GroqStructuredModel,
    GroqTimeoutError,
)
from ai_investigation.llm_investigator import ModelProviderError
from ai_investigation.evaluation.framework import run_experiment
from ai_investigation.evaluation.loader import load_scenarios
from ai_investigation.evidence import EvidenceCollector
from ai_investigation.investigator import DeploymentFailureInvestigator
from ai_investigation.tools import JsonDeploymentTool, JsonLogTool, JsonServiceHealthTool


def test_groq_adapter_makes_one_structured_request() -> None:
    requests = []

    def transport(request, timeout):
        requests.append((request, timeout))
        return json.dumps(
            {"choices": [{"message": {"content": '{"outcome":"abstain"}'}}]}
        ).encode()

    model = GroqStructuredModel("secret", "test-model", transport=transport)

    assert model.generate("prompt") == '{"outcome":"abstain"}'
    assert model.provider_name == "groq"
    assert model.model_name == "test-model"
    assert len(requests) == 1
    payload = json.loads(requests[0][0].data)
    assert payload["model"] == "test-model"
    assert payload["messages"] == [{"role": "user", "content": "prompt"}]
    assert payload["response_format"] == {"type": "json_object"}
    assert payload["temperature"] == 0
    headers = dict(requests[0][0].header_items())
    assert headers["Authorization"] == "Bearer secret"
    assert headers["Content-type"] == "application/json"
    assert headers["Accept"] == "application/json"
    assert headers["User-agent"] == GROQ_USER_AGENT


@pytest.mark.parametrize(
    ("status", "error_type"),
    (
        (401, GroqAuthenticationError),
        (403, GroqAuthenticationError),
        (429, GroqRateLimitError),
        (500, ModelProviderError),
    ),
)
def test_groq_http_errors_are_normalized(status: int, error_type: type[Exception]) -> None:
    def transport(request, timeout):
        raise HTTPError(request.full_url, status, "failed", {}, None)

    with pytest.raises(error_type):
        GroqStructuredModel("secret", "model", transport=transport).generate("prompt")


def _http_error(status: int, body: bytes | object) -> HTTPError:
    return HTTPError("https://api.groq.com", status, "failed", {}, body)


def test_authentication_error_includes_groq_json_message() -> None:
    def transport(request, timeout):
        raise _http_error(
            401,
            BytesIO(json.dumps({"error": {"message": "Invalid API key"}}).encode()),
        )

    with pytest.raises(GroqAuthenticationError) as raised:
        GroqStructuredModel("secret", "model", transport=transport).generate("prompt")

    assert str(raised.value) == (
        "Groq authentication failed. Provider message: Invalid API key"
    )


def test_validation_error_includes_groq_json_message() -> None:
    def transport(request, timeout):
        raise _http_error(
            400,
            BytesIO(
                json.dumps({"error": {"message": "response_format is invalid"}}).encode()
            ),
        )

    with pytest.raises(ModelProviderError) as raised:
        GroqStructuredModel("secret", "model", transport=transport).generate("prompt")

    assert str(raised.value) == (
        "Groq request failed with HTTP 400. Provider message: response_format is invalid"
    )


def test_cloudflare_1010_is_not_reported_as_authentication_failure() -> None:
    body = b"<html><body>Access denied. error code: 1010</body></html>"

    def transport(request, timeout):
        raise _http_error(403, BytesIO(body))

    with pytest.raises(ModelProviderError) as raised:
        GroqStructuredModel("secret", "model", transport=transport).generate("prompt")

    assert not isinstance(raised.value, GroqAuthenticationError)
    assert str(raised.value) == (
        "Groq request was blocked by the upstream HTTP service. "
        "Provider message: <html><body>Access denied. error code: 1010</body></html>"
    )


def test_non_json_http_response_is_included_raw() -> None:
    def transport(request, timeout):
        raise _http_error(502, BytesIO(b"upstream gateway unavailable"))

    with pytest.raises(ModelProviderError) as raised:
        GroqStructuredModel("secret", "model", transport=transport).generate("prompt")

    assert "Provider message: upstream gateway unavailable" in str(raised.value)


def test_unreadable_http_response_uses_generic_message() -> None:
    class UnreadableBody:
        def read(self):
            raise OSError("stream closed")

        def close(self):
            pass

    def transport(request, timeout):
        raise _http_error(500, UnreadableBody())

    with pytest.raises(ModelProviderError) as raised:
        GroqStructuredModel("secret", "model", transport=transport).generate("prompt")

    assert str(raised.value) == "Groq request failed with HTTP 500."


def test_evaluation_report_preserves_groq_provider_message(
    fixture_directory: Path,
) -> None:
    def transport(request, timeout):
        raise _http_error(
            401,
            BytesIO(json.dumps({"error": {"message": "Invalid API key"}}).encode()),
        )

    tools = (
        JsonDeploymentTool(fixture_directory / "deployments.json"),
        JsonLogTool(fixture_directory / "logs.json"),
        JsonServiceHealthTool(fixture_directory / "service_health.json"),
    )
    scenario = load_scenarios(fixture_directory / "evaluation_scenarios.json")[:1]

    report = run_experiment(
        scenario,
        EvidenceCollector(*tools),
        DeploymentFailureInvestigator(*tools),
        investigator_mode="llm",
        structured_model=GroqStructuredModel("secret", "model", transport=transport),
    )

    assert report.scenarios[0].execution_status == "provider_failure"
    assert report.scenarios[0].error == (
        "Groq authentication failed. Provider message: Invalid API key"
    )


def test_groq_timeout_is_normalized() -> None:
    def transport(request, timeout):
        raise socket.timeout()

    with pytest.raises(GroqTimeoutError):
        GroqStructuredModel("secret", "model", transport=transport).generate("prompt")


def test_groq_malformed_envelope_is_a_provider_failure() -> None:
    with pytest.raises(ModelProviderError, match="invalid API response"):
        GroqStructuredModel(
            "secret", "model", transport=lambda request, timeout: b"{}"
        ).generate("prompt")


def test_groq_configuration_requires_key_and_model() -> None:
    with pytest.raises(ValueError, match="GROQ_API_KEY"):
        GroqStructuredModel("", "model")
    with pytest.raises(ValueError, match="model name"):
        GroqStructuredModel("secret", "")
