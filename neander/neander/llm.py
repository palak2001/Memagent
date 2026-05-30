"""LLM provider adapter — OpenAI and Gemini, switchable via ``LLM_PROVIDER``.

Design choice: a thin Protocol + two concrete providers keeps the rest of the
codebase provider-agnostic. The agent and extractor call ``get_provider(settings)``
once and use the returned object. Swapping providers is a one-line env-var change
(``LLM_PROVIDER=gemini``) with zero code changes elsewhere.

OpenAI  → uses the ``openai`` SDK, ``OPENAI_API_KEY``.
Gemini  → uses ``google-generativeai``, ``GEMINI_API_KEY`` (or ``GOOGLE_API_KEY``).
"""

from __future__ import annotations

import json
from typing import Iterator, Protocol, runtime_checkable

from .config import Settings


@runtime_checkable
class LLMProvider(Protocol):
    """Minimal interface required by the agent and extractor."""

    def chat_stream(self, messages: list[dict]) -> Iterator[str]:
        """Stream the assistant response token by token."""
        ...

    def extract_json(self, prompt: str) -> dict:
        """Call the extraction model and return a parsed JSON dict."""
        ...


# ---------------------------------------------------------------------------
# OpenAI provider
# ---------------------------------------------------------------------------

class OpenAIProvider:
    """Direct OpenAI SDK wrapper — streaming chat and JSON extraction."""

    def __init__(self, settings: Settings) -> None:
        import openai  # lazy — only the LLM path needs this

        self._client = openai.OpenAI(api_key=settings.api_key)
        self._chat_model = settings.chat_model
        self._extract_model = settings.extract_model

    def chat_stream(self, messages: list[dict]) -> Iterator[str]:
        stream = self._client.chat.completions.create(
            model=self._chat_model,
            messages=messages,
            stream=True,
        )
        for chunk in stream:
            delta = chunk.choices[0].delta.content
            if delta:
                yield delta

    def extract_json(self, prompt: str) -> dict:
        response = self._client.chat.completions.create(
            model=self._extract_model,
            messages=[{"role": "user", "content": prompt}],
            response_format={"type": "json_object"},
        )
        text = response.choices[0].message.content or "{}"
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            return {}


# ---------------------------------------------------------------------------
# Gemini provider
# ---------------------------------------------------------------------------

class GeminiProvider:
    """Google Gemini adapter via the ``google-generativeai`` SDK."""

    def __init__(self, settings: Settings) -> None:
        import google.generativeai as genai  # lazy import

        genai.configure(api_key=settings.api_key)
        self._chat_model = genai.GenerativeModel(settings.chat_model)
        self._extract_model_name = settings.extract_model
        self._genai = genai

    def _to_gemini_messages(self, messages: list[dict]) -> tuple[str, list[dict]]:
        """Split off the system message and convert OpenAI-style messages to Gemini format."""
        system_parts = []
        history = []
        for msg in messages:
            role = msg["role"]
            content = msg["content"]
            if role == "system":
                system_parts.append(content)
            elif role == "user":
                history.append({"role": "user", "parts": [content]})
            elif role == "assistant":
                history.append({"role": "model", "parts": [content]})
        system_instruction = "\n\n".join(system_parts)
        return system_instruction, history

    def chat_stream(self, messages: list[dict]) -> Iterator[str]:
        import google.generativeai as genai

        system_instruction, history = self._to_gemini_messages(messages)

        model = genai.GenerativeModel(
            self._chat_model._model_name,
            system_instruction=system_instruction if system_instruction else None,
        )

        # Last message must be from the user
        if not history or history[-1]["role"] != "user":
            return
        user_message = history[-1]["parts"][0]
        prior = history[:-1]

        chat = model.start_chat(history=prior)
        response = chat.send_message(user_message, stream=True)
        for chunk in response:
            if chunk.text:
                yield chunk.text

    def extract_json(self, prompt: str) -> dict:
        import google.generativeai as genai

        model = genai.GenerativeModel(self._extract_model_name)
        response = model.generate_content(
            prompt,
            generation_config=genai.types.GenerationConfig(
                response_mime_type="application/json"
            ),
        )
        try:
            return json.loads(response.text)
        except (json.JSONDecodeError, AttributeError):
            return {}


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------

def get_provider(settings: Settings) -> LLMProvider:
    """Instantiate the correct LLM provider from settings."""
    if settings.provider == "gemini":
        return GeminiProvider(settings)
    return OpenAIProvider(settings)
