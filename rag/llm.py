"""Thin wrapper over the official google-genai SDK, with user-friendly error messages."""
from __future__ import annotations

import json
import re
from typing import Any, Dict, Iterator, List, Optional

from utils.logger import setup_logger

logger = setup_logger("llm")


class LLMError(Exception):
    """A model call failed. The message is safe to show to users."""


def friendly_error(exc: Exception) -> str:
    code = getattr(exc, "code", None) or getattr(exc, "status_code", None)
    text = str(exc).lower()
    if code in (401, 403) or "api key not valid" in text or "api_key_invalid" in text or "permission" in text:
        return "Your Gemini API key was rejected. Check the key in the sidebar."
    if code == 429 or "resource_exhausted" in text or "quota" in text:
        return "Gemini rate limit or quota reached. Wait a minute and try again."
    if code == 404 or "not found" in text:
        return "The configured Gemini model is not available for this key. Set GEMINI_MODEL to a model you can use."
    if code in (500, 502, 503, 504) or "unavailable" in text or "overloaded" in text:
        return "Gemini is temporarily unavailable. Please try again shortly."
    if "safety" in text or "blocked" in text:
        return "Gemini declined to answer this request."
    return "The model request failed. Please try again."


class GeminiClient:
    def __init__(self, api_key: str, model: str, client: Optional[Any] = None):
        if not api_key:
            raise LLMError("Add a Gemini API key in the sidebar to ask questions.")
        self.model = model
        if client is None:
            from google import genai  # Lazy import keeps startup fast.

            client = genai.Client(api_key=api_key)
        self._client = client

    def stream(self, system: str, contents: List[Dict[str, Any]]) -> Iterator[str]:
        """Yield text deltas. Raises LLMError with a friendly message on failure."""
        try:
            response = self._client.models.generate_content_stream(
                model=self.model, contents=contents, config={"system_instruction": system}
            )
            for chunk in response:
                text = getattr(chunk, "text", None)
                if text:
                    yield text
        except LLMError:
            raise
        except Exception as exc:
            logger.error("Gemini streaming failed (%s).", type(exc).__name__)
            raise LLMError(friendly_error(exc)) from exc

    def generate_json(self, system: str, prompt: str) -> Dict[str, Any]:
        try:
            response = self._client.models.generate_content(
                model=self.model,
                contents=[{"role": "user", "parts": [{"text": prompt}]}],
                config={"system_instruction": system, "response_mime_type": "application/json"},
            )
        except Exception as exc:
            logger.error("Gemini JSON request failed (%s).", type(exc).__name__)
            raise LLMError(friendly_error(exc)) from exc
        return parse_json_object(getattr(response, "text", "") or "")


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
