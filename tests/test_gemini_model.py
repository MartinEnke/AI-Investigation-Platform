from enum import Enum
from types import SimpleNamespace

import pytest

from ai_investigation.gemini_model import (
    DEFAULT_GEMINI_MODEL,
    GeminiStructuredModel,
)
from ai_investigation.llm_investigator import (
    LLM_RESPONSE_JSON_SCHEMA,
    ModelProviderError,
    ModelRefusalError,
)


class FakeModels:
    def __init__(self, response: object = None, error: Exception | None = None) -> None:
        self.response = response
        self.error = error
        self.calls: list[dict[str, object]] = []

    def generate_content(self, **kwargs: object) -> object:
        self.calls.append(kwargs)
        if self.error is not None:
            raise self.error
        return self.response


class FakeClient:
    def __init__(self, models: FakeModels) -> None:
        self.models = models


def gemini_response(
    *,
    text: str | None = '{"outcome":"abstain"}',
    block_reason: object = None,
    finish_reason: object = "STOP",
    candidates: bool = True,
) -> object:
    return SimpleNamespace(
        text=text,
        prompt_feedback=SimpleNamespace(block_reason=block_reason),
        candidates=(SimpleNamespace(finish_reason=finish_reason),) if candidates else (),
    )


def test_gemini_request_uses_model_and_provider_independent_schema() -> None:
    models = FakeModels(gemini_response())
    adapter = GeminiStructuredModel(
        "test-key",
        DEFAULT_GEMINI_MODEL,
        client=FakeClient(models),
    )

    result = adapter.generate("investigate this evidence")

    assert result == '{"outcome":"abstain"}'
    assert models.calls == [
        {
            "model": "gemini-2.5-flash",
            "contents": "investigate this evidence",
            "config": {
                "response_mime_type": "application/json",
                "response_json_schema": LLM_RESPONSE_JSON_SCHEMA,
            },
        }
    ]


@pytest.mark.parametrize("field", ("api_key", "model"))
def test_gemini_configuration_is_explicit(field: str) -> None:
    arguments = {"api_key": "test-key", "model": DEFAULT_GEMINI_MODEL}
    arguments[field] = ""

    with pytest.raises(ValueError, match="required"):
        GeminiStructuredModel(**arguments, client=FakeClient(FakeModels()))


def test_prompt_block_is_normalized_as_refusal() -> None:
    adapter = GeminiStructuredModel(
        "test-key",
        DEFAULT_GEMINI_MODEL,
        client=FakeClient(FakeModels(gemini_response(block_reason="SAFETY", candidates=False))),
    )

    with pytest.raises(ModelRefusalError, match=r"blocked the prompt \(SAFETY\)"):
        adapter.generate("prompt")


class FinishReason(Enum):
    SAFETY = "SAFETY"


def test_filtered_candidate_is_normalized_as_refusal() -> None:
    adapter = GeminiStructuredModel(
        "test-key",
        DEFAULT_GEMINI_MODEL,
        client=FakeClient(
            FakeModels(gemini_response(finish_reason=FinishReason.SAFETY, text=None))
        ),
    )

    with pytest.raises(ModelRefusalError, match=r"filtered the response \(SAFETY\)"):
        adapter.generate("prompt")


def test_no_candidate_is_normalized_as_provider_failure() -> None:
    adapter = GeminiStructuredModel(
        "test-key",
        DEFAULT_GEMINI_MODEL,
        client=FakeClient(FakeModels(gemini_response(candidates=False))),
    )

    with pytest.raises(ModelProviderError, match="no response candidate"):
        adapter.generate("prompt")


def test_sdk_exception_is_normalized_without_retry() -> None:
    models = FakeModels(error=RuntimeError("network unavailable"))
    adapter = GeminiStructuredModel(
        "test-key",
        DEFAULT_GEMINI_MODEL,
        client=FakeClient(models),
    )

    with pytest.raises(ModelProviderError, match="Gemini request failed"):
        adapter.generate("prompt")

    assert len(models.calls) == 1


def test_candidate_without_text_is_left_for_application_validation() -> None:
    adapter = GeminiStructuredModel(
        "test-key",
        DEFAULT_GEMINI_MODEL,
        client=FakeClient(FakeModels(gemini_response(text=None))),
    )

    assert adapter.generate("prompt") == ""
