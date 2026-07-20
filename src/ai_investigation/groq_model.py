"""Small Groq chat-completions adapter for structured investigation output."""

import json
import socket
from typing import Callable
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from ai_investigation.llm_investigator import ModelProviderError

DEFAULT_GROQ_MODEL = "llama-3.3-70b-versatile"
GROQ_CHAT_COMPLETIONS_URL = "https://api.groq.com/openai/v1/chat/completions"
GROQ_USER_AGENT = "AI-Investigation-Platform/1.0"


class GroqAuthenticationError(ModelProviderError):
    pass


class GroqRateLimitError(ModelProviderError):
    pass


class GroqTimeoutError(ModelProviderError):
    pass


def _http_transport(request: Request, timeout: float) -> bytes:
    with urlopen(request, timeout=timeout) as response:  # noqa: S310 - fixed HTTPS endpoint
        return response.read()


class GroqStructuredModel:
    """Make one synchronous Groq request; application validation remains authoritative."""

    def __init__(
        self,
        api_key: str,
        model: str,
        *,
        timeout_seconds: float = 30.0,
        transport: Callable[[Request, float], bytes] = _http_transport,
    ) -> None:
        if not api_key:
            raise ValueError("GROQ_API_KEY is required for Groq evaluation.")
        if not model:
            raise ValueError("A Groq model name is required.")
        self._api_key = api_key
        self._model = model
        self._timeout_seconds = timeout_seconds
        self._transport = transport

    @property
    def provider_name(self) -> str:
        return "groq"

    @property
    def model_name(self) -> str:
        return self._model

    def generate(self, prompt: str) -> str:
        payload = json.dumps(
            {
                "model": self._model,
                "messages": [{"role": "user", "content": prompt}],
                "response_format": {"type": "json_object"},
                "temperature": 0,
            },
            separators=(",", ":"),
        ).encode("utf-8")
        request = Request(
            GROQ_CHAT_COMPLETIONS_URL,
            data=payload,
            headers={
                "Authorization": f"Bearer {self._api_key}",
                "Content-Type": "application/json",
                "Accept": "application/json",
                "User-Agent": GROQ_USER_AGENT,
            },
            method="POST",
        )
        try:
            raw = self._transport(request, self._timeout_seconds)
        except HTTPError as error:
            detail = _http_error_detail(error)
            if error.code == 403 and _is_cloudflare_1010(detail):
                message = _with_detail(
                    "Groq request was blocked by the upstream HTTP service.", detail
                )
                raise ModelProviderError(message) from error
            if error.code in (401, 403):
                message = _with_detail("Groq authentication failed.", detail)
                raise GroqAuthenticationError(message) from error
            if error.code == 429:
                message = _with_detail(
                    "Groq rate limit or quota was exceeded.", detail
                )
                raise GroqRateLimitError(message) from error
            message = _with_detail(
                f"Groq request failed with HTTP {error.code}.", detail
            )
            raise ModelProviderError(message) from error
        except (TimeoutError, socket.timeout) as error:
            raise GroqTimeoutError("Groq request timed out.") from error
        except URLError as error:
            if isinstance(error.reason, (TimeoutError, socket.timeout)):
                raise GroqTimeoutError("Groq request timed out.") from error
            raise ModelProviderError("Groq request failed.") from error
        except OSError as error:
            raise ModelProviderError("Groq request failed.") from error

        try:
            response = json.loads(raw.decode("utf-8"))
            content = response["choices"][0]["message"]["content"]
        except (UnicodeDecodeError, json.JSONDecodeError, KeyError, IndexError, TypeError) as error:
            raise ModelProviderError("Groq returned an invalid API response.") from error
        if not isinstance(content, str) or not content.strip():
            return ""
        return content


def _http_error_detail(error: HTTPError) -> str | None:
    """Read an HTTP error body once and retain its useful provider detail."""

    try:
        body = error.read()
    except Exception:  # HTTP response streams may raise provider-specific I/O errors.
        return None
    if not body:
        return None
    if isinstance(body, bytes):
        text = body.decode("utf-8", errors="replace").strip()
    else:
        text = str(body).strip()
    if not text:
        return None

    try:
        payload = json.loads(text)
    except json.JSONDecodeError:
        return text

    if isinstance(payload, dict):
        provider_error = payload.get("error")
        if isinstance(provider_error, dict):
            message = provider_error.get("message")
            if isinstance(message, str) and message.strip():
                return message.strip()
        message = payload.get("message")
        if isinstance(message, str) and message.strip():
            return message.strip()
    return text


def _with_detail(generic_message: str, detail: str | None) -> str:
    if detail is None:
        return generic_message
    return f"{generic_message} Provider message: {detail}"


def _is_cloudflare_1010(detail: str | None) -> bool:
    return detail is not None and "error code: 1010" in detail.casefold()
