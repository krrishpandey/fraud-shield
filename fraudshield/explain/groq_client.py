"""Groq (OpenAI-compatible) client exposing the small part of the Anthropic SDK interface that
explain/llm.py uses: client.messages.create(...) -> obj with .content[*].type/.text and .stop_reason.
Lets the explanation layer run on Groq-hosted open models with no change to prompt, validator or audit.
"""
from __future__ import annotations

import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import httpx

GROQ_URL = "https://api.groq.com/openai/v1/chat/completions"
DEFAULT_GROQ_MODEL = "openai/gpt-oss-120b"
REASONING_PREFIXES = ("openai/gpt-oss",)
RATE_LIMIT_RETRIES = 3
MAX_RETRY_WAIT_S = 30.0


@dataclass
class _Block:
    text: str
    type: str = "text"


@dataclass
class _Response:
    content: list[_Block]
    stop_reason: str
    usage: dict[str, Any] | None = None  # Groq token counts (lets batch jobs pace themselves under the TPM limit)


class _Messages:
    def __init__(self, owner: "GroqClient"):
        self._o = owner

    def create(self, *, model: str, max_tokens: int, temperature: float, system: str,
               messages: list[dict[str, Any]], response_format: dict[str, Any] | None = None) -> _Response:
        body = {"model": model, "max_tokens": max_tokens, "temperature": temperature,
                "messages": [{"role": "system", "content": system}, *messages]}
        if response_format is not None:
            # Structured outputs (explain/claims.py): {"type": "json_schema", ...} or {"type": "json_object"}.
            # A schema Groq cannot serve comes back as HTTP 400, which the caller treats as "try the next mode".
            body["response_format"] = response_format
        if model.startswith(REASONING_PREFIXES):
            # Hidden reasoning tokens count against max_tokens; without room the answer is cut off.
            body["reasoning_effort"] = "low"
            body["max_tokens"] = max(max_tokens, 2000)
        # Rate limited (429): wait as long as Groq asks (Retry-After) and try again. Explanations run in the
        # background, so a short wait is fine. A Retry-After beyond the cap means the daily quota is spent: give up
        # at once rather than delay the template. After the last try the error reaches the caller (template shown).
        for attempt in range(RATE_LIMIT_RETRIES + 1):
            r = self._o.http.post(GROQ_URL, json=body, headers={"Authorization": f"Bearer {self._o.api_key}"},
                                  timeout=self._o.timeout)
            if r.status_code != 429 or attempt == RATE_LIMIT_RETRIES:
                break
            try:
                wait = float(r.headers.get("retry-after", 2 ** attempt))
            except ValueError:
                wait = float(2 ** attempt)
            if wait > MAX_RETRY_WAIT_S:
                break
            time.sleep(max(wait, 0.5))
        r.raise_for_status()
        j = r.json()
        choice = j["choices"][0]
        text = (choice.get("message") or {}).get("content") or ""
        stop = "end_turn" if choice.get("finish_reason") == "stop" else str(choice.get("finish_reason"))
        return _Response([_Block(text.strip())], stop, j.get("usage"))


class GroqClient:
    supports_response_format = True  # explain/claims.py asks for strict JSON claims only from clients that say so

    def __init__(self, api_key: str, http: httpx.Client | None = None, timeout: float = 15.0):
        self.api_key = api_key
        self.http = http or httpx.Client()
        self.timeout = timeout
        self.messages = _Messages(self)


def pick_llm(explain_cfg: dict[str, Any]) -> tuple[Any, str | None, str | None]:
    """(client, model_id, provider). Anthropic if its key is set, else Groq, else no LLM (template)."""
    if os.environ.get("ANTHROPIC_API_KEY"):
        import anthropic  # noqa: PLC0415

        from fraudshield.explain.llm import DEFAULT_MODEL_ID
        return (anthropic.Anthropic(timeout=15.0, max_retries=1),
                explain_cfg.get("model_id", DEFAULT_MODEL_ID), "anthropic")
    if os.environ.get("GROQ_API_KEY"):
        return (GroqClient(os.environ["GROQ_API_KEY"]),
                explain_cfg.get("groq_model_id", DEFAULT_GROQ_MODEL), "groq")
    return None, None, None


def load_dotenv(path: str | Path) -> None:
    """Minimal .env loader: KEY=VALUE lines; existing environment variables win."""
    p = Path(path)
    if not p.exists():
        return
    for line in p.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        k, v = (s.strip() for s in line.split("=", 1))
        os.environ.setdefault(k, v.strip('"').strip("'"))
