"""Concrete Google Gemini adapter for structured investigation responses."""

from typing import Any

from ai_investigation.llm_investigator import (
    LLM_RESPONSE_JSON_SCHEMA,
    ModelProviderError,
    ModelRefusalError,
)

DEFAULT_GEMINI_MODEL = "gemini-2.5-flash"
_BLOCKED_REASONS = {
    "BLOCKLIST",
    "IMAGE_SAFETY",
    "MODEL_ARMOR",
    "PROHIBITED_CONTENT",
    "RECITATION",
    "SAFETY",
}


class GeminiStructuredModel:
    """Make one synchronous Gemini request constrained by the approved JSON schema."""

    def __init__(self, api_key: str, model: str, *, client: object | None = None) -> None:
        if not api_key:
            raise ValueError("A Gemini API key is required.")
        if not model:
            raise ValueError("A Gemini model name is required.")
        self._model = model
        self._client = client if client is not None else _create_client(api_key)

    def generate(self, prompt: str) -> str:
        try:
            response = self._client.models.generate_content(  # type: ignore[attr-defined]
                model=self._model,
                contents=prompt,
                config={
                    "response_mime_type": "application/json",
                    "response_json_schema": LLM_RESPONSE_JSON_SCHEMA,
                },
            )
            _raise_for_blocked_response(response)
            candidates = getattr(response, "candidates", None)
            if not candidates:
                raise ModelProviderError("Gemini returned no response candidate.")
            text = getattr(response, "text", None)
            return text if isinstance(text, str) else ""
        except (ModelProviderError, ModelRefusalError):
            raise
        except Exception as error:
            raise ModelProviderError("Gemini request failed.") from error


def _create_client(api_key: str) -> Any:
    try:
        from google import genai
    except ImportError as error:
        raise ModelProviderError(
            "Gemini support requires the optional 'experiment' dependency."
        ) from error

    try:
        return genai.Client(api_key=api_key)
    except Exception as error:
        raise ModelProviderError("Gemini client initialization failed.") from error


def _raise_for_blocked_response(response: object) -> None:
    prompt_feedback = getattr(response, "prompt_feedback", None)
    block_reason = _enum_value(getattr(prompt_feedback, "block_reason", None))
    if block_reason and block_reason not in {
        "BLOCK_REASON_UNSPECIFIED",
        "UNSPECIFIED",
    }:
        raise ModelRefusalError(f"Gemini blocked the prompt ({block_reason}).")

    for candidate in getattr(response, "candidates", None) or ():
        finish_reason = _enum_value(getattr(candidate, "finish_reason", None))
        if finish_reason in _BLOCKED_REASONS:
            raise ModelRefusalError(f"Gemini filtered the response ({finish_reason}).")


def _enum_value(value: object) -> str | None:
    if value is None:
        return None
    raw_value = getattr(value, "value", value)
    text = str(raw_value)
    return text.rsplit(".", 1)[-1]
