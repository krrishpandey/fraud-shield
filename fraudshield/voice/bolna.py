"""Thin Bolna REST client. Endpoints and fields as documented at https://www.bolna.ai/docs (read 31 Oct 2026):

POST /v2/agent                 {agent_config, agent_prompts} -> {agent_id, state, version_id}
PUT  /v2/agent/{agent_id}      same body
POST /dispositions/bulk        {agent_id, dispositions: [...]} -> {message, ids}
POST /call                     {agent_id, recipient_phone_number, from_phone_number?, user_data?, agent_data?}
                               -> {message, status: "queued", execution_id}
GET  /executions/{id}          -> AgentExecution (status, transcript, extracted_data, telephony_data, ...)
Auth: Authorization: Bearer <BOLNA_API_KEY>.
"""
from __future__ import annotations

from typing import Any

import httpx

BASE_URL = "https://api.bolna.ai"


class BolnaError(RuntimeError):
    pass


class BolnaClient:
    def __init__(self, api_key: str, base_url: str = BASE_URL, timeout_s: float = 15.0,
                 transport: httpx.BaseTransport | None = None):
        if not api_key:
            raise BolnaError("BOLNA_API_KEY is not set")
        self._http = httpx.Client(base_url=base_url, timeout=timeout_s, transport=transport,
                                  headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"})

    def _req(self, method: str, path: str, body: dict[str, Any] | None = None) -> dict[str, Any]:
        try:
            r = self._http.request(method, path, json=body)
        except httpx.HTTPError as e:
            raise BolnaError(f"{method} {path}: {type(e).__name__}: {e}") from e
        if r.status_code >= 400:
            raise BolnaError(f"{method} {path}: HTTP {r.status_code}: {r.text[:300]}")
        try:
            return r.json()
        except ValueError as e:
            raise BolnaError(f"{method} {path}: response is not JSON") from e

    def create_agent(self, body: dict[str, Any]) -> dict[str, Any]:
        return self._req("POST", "/v2/agent", body)

    def update_agent(self, agent_id: str, body: dict[str, Any]) -> dict[str, Any]:
        return self._req("PUT", f"/v2/agent/{agent_id}", body)

    def create_dispositions(self, agent_id: str, dispositions: list[dict[str, Any]]) -> dict[str, Any]:
        return self._req("POST", "/dispositions/bulk", {"agent_id": agent_id, "dispositions": dispositions})

    def place_call(self, agent_id: str, phone: str, user_data: dict[str, Any], language: str | None = None,
                   from_phone: str | None = None, bypass_guardrails: bool = False) -> dict[str, Any]:
        # bypass_guardrails skips the agent's calling-hours window (demo recording at night); off by default
        body: dict[str, Any] = {"agent_id": agent_id, "recipient_phone_number": phone, "user_data": user_data,
                                "bypass_call_guardrails": bool(bypass_guardrails)}
        if language:
            body["agent_data"] = {"language": language}  # "start call in customer's language"
        if from_phone:
            body["from_phone_number"] = from_phone
        out = self._req("POST", "/call", body)
        if not out.get("execution_id"):
            raise BolnaError(f"POST /call returned no execution_id: {out}")
        return out

    def get_execution(self, execution_id: str) -> dict[str, Any]:
        return self._req("GET", f"/executions/{execution_id}")
