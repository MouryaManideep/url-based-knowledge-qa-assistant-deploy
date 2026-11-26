"""Thin wrapper over the official google-genai SDK with retries, model fallback and
user-friendly error messages."""
from __future__ import annotations

import json
import random
import re
import time
from typing import Any, Callable, Dict, Iterator, List, Optional, Sequence, Union

from utils.logger import setup_logger

logger = setup_logger("llm")

# Overloaded / transient server errors: retry the same model once, then fall back.
TRANSIENT_CODES = {500, 502, 503, 504}
# Quota, missing model, or still overloaded after a retry: move on to the next model.
FALLBACK_CODES = TRANSIENT_CODES | {429, 404}


class LLMError(Exception):
    """A model call failed. The message is safe to show to users."""


def error_code(exc: Exception) -> Optional[int]:
    code = getattr(exc, "code", None) or getattr(exc, "status_code", None)
    try:
        return int(code) if code is not None else None
    except (TypeError, ValueError):
        return None


def friendly_error(exc: Optional[Exception]) -> str:
    if exc is None:
        return "The model request failed. Please try again."
    code = error_code(exc)
    text = str(exc).lower()
    if code in (401, 403) or "api key not valid" in text or "api_key_invalid" in text or "permission" in text:
        return "Your Gemini API key was rejected. Check the key in the sidebar."
    if code == 429 or "resource_exhausted" in text or "quota" in text:
        return "Gemini rate limit or quota reached. Wait a minute and try again."
    if code == 404 or "not found" in text:
        return "The configured Gemini models are not available for this key. Set GEMINI_MODEL to a model you can use."
    if code in TRANSIENT_CODES or "unavailable" in text or "overloaded" in text:
        return "Gemini is temporarily overloaded (fallback models too). Please try again in a minute."
    if "safety" in text or "blocked" in text:
        return "Gemini declined to answer this request."
    return "The model request failed. Please try again."


def _log_failure(model: str, exc: Exception) -> None:
    message = re.sub(r"\s+", " ", str(exc))[:240]
    logger.warning("Gemini call failed: model=%s code=%s type=%s message=%s", model, error_code(exc), type(exc).__name__, message)


class GeminiClient:
    def __init__(
        self,
        api_key: str,
        models: Union[str, Sequence[str]],
        client: Optional[Any] = None,
        sleep: Callable[[float], None] = time.sleep,
    ):
        if not api_key:
            raise LLMError("Add a Gemini API key in the sidebar to ask questions.")
        if isinstance(models, str):  # A single name gets the configured fallbacks appended.
            from config import GEMINI_FALLBACK_MODELS

            models = [models, *GEMINI_FALLBACK_MODELS]
        self.models: List[str] = list(dict.fromkeys(m for m in models if m))
        if not self.models:
            raise LLMError("No Gemini model is configured.")
        self._sleep = sleep
        if client is None:
            from google import genai  # Lazy import keeps startup fast.

            client = genai.Client(api_key=api_key)
        self._client = client
        self.last_model: Optional[str] = None

    @property
    def model(self) -> str:
        return self.models[0]

    def _attempts(self):
        """Yield (model, attempt) pairs: each model gets one retry for transient errors."""
        for model in self.models:
            for attempt in range(2):
                yield model, attempt

    def _should_retry(self, exc: Exception, attempt: int) -> str:
        """Return 'retry' (same model), 'next' (next model) or 'raise'."""
        code = error_code(exc)
        if code in TRANSIENT_CODES and attempt == 0:
            return "retry"
        if code in FALLBACK_CODES:
            return "next"
        return "raise"

    def stream(self, system: str, contents: List[Dict[str, Any]]) -> Iterator[str]:
        """Yield text deltas. Raises LLMError with a friendly message on failure."""
        last_exc: Optional[Exception] = None
        skip_model: Optional[str] = None
        for model, attempt in self._attempts():
            if model == skip_model:
                continue
            started = False
            try:
                response = self._client.models.generate_content_stream(
                    model=model, contents=contents, config={"system_instruction": system}
                )
                for chunk in response:
                    text = getattr(chunk, "text", None)
                    if text:
                        started = True
                        self.last_model = model
                        yield text
                return
            except Exception as exc:
                _log_failure(model, exc)
                last_exc = exc
                if started:  # Can't transparently restart an answer the user is already reading.
                    raise LLMError("The answer was interrupted by a Gemini error. Please ask again.") from exc
                decision = self._should_retry(exc, attempt)
                if decision == "raise":
                    raise LLMError(friendly_error(exc)) from exc
                if decision == "retry":
                    self._sleep(1.0 + random.random())
                else:
                    skip_model = model
        raise LLMError(friendly_error(last_exc)) from last_exc

    def generate_json(self, system: str, prompt: str) -> Dict[str, Any]:
        last_exc: Optional[Exception] = None
        skip_model: Optional[str] = None
        for model, attempt in self._attempts():
            if model == skip_model:
                continue
            try:
                response = self._client.models.generate_content(
                    model=model,
                    contents=[{"role": "user", "parts": [{"text": prompt}]}],
                    config={"system_instruction": system, "response_mime_type": "application/json"},
                )
                self.last_model = model
                return parse_json_object(getattr(response, "text", "") or "")
            except LLMError:
                raise
            except Exception as exc:
                _log_failure(model, exc)
                last_exc = exc
                decision = self._should_retry(exc, attempt)
                if decision == "raise":
                    raise LLMError(friendly_error(exc)) from exc
                if decision == "retry":
                    self._sleep(1.0 + random.random())
                else:
                    skip_model = model
        raise LLMError(friendly_error(last_exc)) from last_exc


def parse_json_object(text: str) -> Dict[str, Any]:
    cleaned = text.strip()
    fenced = re.match(r"^```(?:json)?\s*(.*?)\s*```$", cleaned, re.S)
    if fenced:
        cleaned = fenced.group(1)
    try:
        value = json.loads(cleaned)
    except json.JSONDecodeError:
        start, end = cleaned.find("{"), cleaned.rfind("}")
        if start == -1 or end <= start:
            raise LLMError("The model returned an unexpected response. Please try again.")
        try:
            value = json.loads(cleaned[start:end + 1])
        except json.JSONDecodeError as exc:
            raise LLMError("The model returned an unexpected response. Please try again.") from exc
    if not isinstance(value, dict):
        raise LLMError("The model returned an unexpected response. Please try again.")
    return value
