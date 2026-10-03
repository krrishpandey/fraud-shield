import json

import httpx
import pytest

from fraudshield.explain.groq_client import GroqClient, load_dotenv, pick_llm


def _client(handler):
    return GroqClient(api_key="k", http=httpx.Client(transport=httpx.MockTransport(handler)))


def test_messages_create_maps_to_openai_chat_and_back():
    seen = {}

    def handler(req: httpx.Request):
        seen["url"] = str(req.url)
        seen["auth"] = req.headers["authorization"]
        seen["body"] = json.loads(req.content)
        return httpx.Response(200, json={"choices": [{"message": {"content": " Held because X. "},
                                                       "finish_reason": "stop"}]})

    resp = _client(handler).messages.create(model="m", max_tokens=400, temperature=0, system="SYS",
                                            messages=[{"role": "user", "content": "U"}])
    assert seen["url"].endswith("/openai/v1/chat/completions")
    assert seen["auth"] == "Bearer k"
    assert seen["body"]["messages"] == [{"role": "system", "content": "SYS"}, {"role": "user", "content": "U"}]
    assert seen["body"]["model"] == "m" and seen["body"]["temperature"] == 0
    assert [b.type for b in resp.content] == ["text"]
    assert resp.content[0].text.strip() == "Held because X."
    assert resp.stop_reason == "end_turn"


def test_http_error_raises_so_caller_falls_back_to_template():
    c = _client(lambda req: httpx.Response(401, json={"error": {"message": "bad key"}}))
    with pytest.raises(Exception):
        c.messages.create(model="m", max_tokens=10, temperature=0, system="s", messages=[])


def test_explain_uses_groq_client_end_to_end():
    from fraudshield.explain.llm import explain
    from tests.explain.test_explain import RECORD  # noqa: F401  (shared fixture record)

    def handler(req):
        return httpx.Response(200, json={"choices": [{"message": {"content": "irrelevant"}, "finish_reason": "stop"}]})

    exp, log = explain(RECORD, client=_client(handler), model_id="m")
    assert log["llm_output"] == "irrelevant"  # validator then decides llm vs template


def test_pick_llm_prefers_anthropic_then_groq(monkeypatch):
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    monkeypatch.delenv("GROQ_API_KEY", raising=False)
    assert pick_llm({}) == (None, None, None)
    monkeypatch.setenv("GROQ_API_KEY", "g")
    client, model, provider = pick_llm({"groq_model_id": "openai/gpt-oss-120b"})
    assert provider == "groq" and model == "openai/gpt-oss-120b" and isinstance(client, GroqClient)
    monkeypatch.setenv("ANTHROPIC_API_KEY", "a")
    assert pick_llm({"model_id": "claude-x"})[2] == "anthropic"


def test_load_dotenv_sets_missing_vars_only(tmp_path, monkeypatch):
    p = tmp_path / ".env"
    p.write_text("# comment\nFOO_TEST_VAR=from_file\nBAR_TEST_VAR = spaced \n")
    monkeypatch.setenv("FOO_TEST_VAR", "from_env")
    monkeypatch.delenv("BAR_TEST_VAR", raising=False)
    load_dotenv(p)
    import os
    assert os.environ["FOO_TEST_VAR"] == "from_env"
    assert os.environ["BAR_TEST_VAR"] == "spaced"


def test_reasoning_models_get_low_effort_and_room_for_reasoning():
    seen = {}

    def handler(req):
        seen["body"] = json.loads(req.content)
        return httpx.Response(200, json={"choices": [{"message": {"content": "ok"}, "finish_reason": "stop"}]})

    _client(handler).messages.create(model="openai/gpt-oss-120b", max_tokens=400, temperature=0, system="s",
                                     messages=[{"role": "user", "content": "u"}])
    assert seen["body"]["reasoning_effort"] == "low"
    assert seen["body"]["max_tokens"] >= 2000  # reasoning tokens count against the limit


def test_non_reasoning_models_unchanged():
    seen = {}

    def handler(req):
        seen["body"] = json.loads(req.content)
        return httpx.Response(200, json={"choices": [{"message": {"content": "ok"}, "finish_reason": "stop"}]})

    _client(handler).messages.create(model="qwen/qwen3.8-27b", max_tokens=400, temperature=0, system="s",
                                     messages=[])
    assert "reasoning_effort" not in seen["body"] and seen["body"]["max_tokens"] == 400
