import json
import os
import time

import httpx
import pytest

from fraudshield.contracts import QUESTIONS, SERVED_QUESTIONS
from fraudshield.models.laya_client import LayaClient, LayaUnavailable, cache_key

STATE = "BOOKING 2018-06-14 Thu 02:41 | channel api"


class FakeAgent:
    def __init__(self, delay=0.0, p=0.8):
        self.calls = []
        self.delay = delay
        self.p = p

    def predict(self, state, questions):
        self.calls.append((state, list(questions)))
        time.sleep(self.delay)
        return {"answers": {q: {"choice": "a", "probabilities": {"a": self.p, "b": 1 - self.p}} for q in questions},
                "usage": {"input_tokens": 300, "output_tokens": 0}}


def test_local_predicts_all_served_questions_in_one_call():
    agent = FakeAgent()
    c = LayaClient.local(agent=agent, revision="rev1")
    res = c.predict(STATE)
    assert len(agent.calls) == 1
    assert agent.calls[0][1] == list(SERVED_QUESTIONS)
    assert set(res.raw) == set(SERVED_QUESTIONS)
    assert res.raw["misuse"] == {"a": 0.8, "b": pytest.approx(0.2)}
    assert res.cached is False and res.mode == "local" and res.revision == "rev1"
    assert res.latency_ms >= 0


def test_local_timeout_raises_unavailable():
    c = LayaClient.local(agent=FakeAgent(delay=0.5), timeout_s=0.05)
    with pytest.raises(LayaUnavailable):
        c.predict(STATE)


def test_local_error_raises_unavailable():
    class Boom:
        def predict(self, *a, **k):
            raise RuntimeError("cuda oom")
    with pytest.raises(LayaUnavailable):
        LayaClient.local(agent=Boom()).predict(STATE)


def test_ask_custom_question_zero_shot():
    agent = FakeAgent(p=0.82)
    c = LayaClient.local(agent=agent)
    q = {"type": "choice", "instructions": "Is it a drop?", "criteria": {"a": "yes", "b": "no"}}
    res = c.ask(STATE, q, qid="custom_1")
    assert res.raw == {"custom_1": {"a": 0.82, "b": pytest.approx(0.18)}}


def test_http_mode_sends_jev_compatible_body():
    seen = {}

    def handler(request: httpx.Request):
        seen["url"] = str(request.url)
        seen["auth"] = request.headers.get("authorization")
        body = json.loads(request.content)
        seen["body"] = body
        ans = {q: {"choice": "b", "probabilities": {"a": 0.1, "b": 0.9}} for q in body["questions"]}
        return httpx.Response(200, json={"answers": ans, "usage": {}})

    client = httpx.Client(transport=httpx.MockTransport(handler))
    c = LayaClient.http("http://localhost:8001", api_key="k", http_client=client)
    res = c.predict(STATE)
    assert seen["url"] == "http://localhost:8001/v1/systemone"
    assert seen["auth"] == "Bearer k"
    assert seen["body"]["state"] == STATE
    assert seen["body"]["questions"]["misuse"] == QUESTIONS["misuse"]
    assert res.raw["misuse"]["b"] == 0.9 and res.mode == "http"


def test_http_5xx_and_timeout_raise_unavailable():
    c = LayaClient.http("http://x", http_client=httpx.Client(transport=httpx.MockTransport(lambda r: httpx.Response(503))))
    with pytest.raises(LayaUnavailable):
        c.predict(STATE)

    def slow(r):
        raise httpx.ReadTimeout("t")
    c2 = LayaClient.http("http://x", http_client=httpx.Client(transport=httpx.MockTransport(slow)))
    with pytest.raises(LayaUnavailable):
        c2.predict(STATE)


def test_cached_mode_hits_and_misses(tmp_path):
    p = tmp_path / "cache.json"
    key = cache_key(STATE, list(SERVED_QUESTIONS))
    p.write_text(json.dumps({key: {q: {"a": 0.7, "b": 0.3} for q in SERVED_QUESTIONS}}))
    c = LayaClient.cached(p)
    res = c.predict(STATE)
    assert res.cached is True and res.raw["misuse"]["a"] == 0.7
    with pytest.raises(LayaUnavailable):
        c.predict("some other state")


def test_record_to_cache_writes_entries(tmp_path):
    p = tmp_path / "cache.json"
    c = LayaClient.local(agent=FakeAgent(), record_cache=p)
    c.predict(STATE)
    c.flush_cache()
    assert cache_key(STATE, list(SERVED_QUESTIONS)) in json.loads(p.read_text())


def test_cache_key_depends_on_state_and_questions():
    assert cache_key("s", ["misuse"]) != cache_key("s2", ["misuse"])
    assert cache_key("s", ["misuse"]) != cache_key("s", ["payoff_max"])


@pytest.mark.gpu
@pytest.mark.skipif(os.environ.get("FS_GPU_TESTS") != "1", reason="set FS_GPU_TESTS=1 to load the real Laya model")
def test_real_laya_model_output_shape():
    c = LayaClient.local(model_path="convaiinnovations/laya", device="cuda", timeout_s=120)
    res = c.predict(STATE)
    assert set(res.raw) == set(SERVED_QUESTIONS)
    for q in SERVED_QUESTIONS:
        assert set(res.raw[q]) == {"a", "b"}
        assert sum(res.raw[q].values()) == pytest.approx(1.0, abs=1e-3)


def test_auto_mode_uses_local_when_cuda_and_laya_available(tmp_path):
    made = {}

    def loader(path, device):
        made["args"] = (path, device)
        return FakeAgent()

    c = LayaClient.auto(model_path="m", cache_path=tmp_path / "c.json", cuda_available=lambda: True, loader=loader)
    assert c.mode == "local" and made["args"] == ("m", "cuda")


def test_auto_mode_falls_back_to_cached_without_gpu(tmp_path):
    c = LayaClient.auto(model_path="m", cache_path=tmp_path / "c.json", cuda_available=lambda: False)
    assert c.mode == "cached"
    assert "no CUDA" in c.mode_reason


def test_auto_mode_falls_back_to_cached_when_model_load_fails(tmp_path):
    def loader(path, device):
        raise ImportError("laya not installed")
    c = LayaClient.auto(model_path="m", cache_path=tmp_path / "c.json", cuda_available=lambda: True, loader=loader)
    assert c.mode == "cached" and "laya not installed" in c.mode_reason


def test_module_imports_without_torch():
    import sys, subprocess
    code = ("import sys; sys.modules['torch']=None; sys.modules['laya']=None; "
            "import fraudshield.models.laya_client as m; print(m.cuda_available())")
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
    assert out.returncode == 0, out.stderr
    assert out.stdout.strip() == "False"


def test_cuda_available_is_checked_once(monkeypatch):
    import sys
    import types

    from fraudshield.models import laya_client

    calls = {"n": 0}

    def is_available():
        calls["n"] += 1
        return False

    fake = types.SimpleNamespace(cuda=types.SimpleNamespace(is_available=is_available))
    monkeypatch.setitem(sys.modules, "torch", fake)
    laya_client.cuda_available.cache_clear()
    laya_client.cuda_available()
    laya_client.cuda_available()
    assert calls["n"] == 1
    laya_client.cuda_available.cache_clear()


# ---------- LazyLaya: stock model for "ask a new question" while the fine-tuned model is not live ----------
def test_lazy_laya_loads_once_on_first_ask():
    from fraudshield.models.laya_client import LazyLaya
    made = []

    def factory():
        made.append(1)
        return LayaClient.local(agent=FakeAgent(), revision="stock")

    lz = LazyLaya(factory, label="stock Laya (not fine-tuned)")
    assert made == [] and lz.state() == "not loaded"
    q = {"type": "choice", "instructions": "Is it odd?", "criteria": {"a": "yes", "b": "no"}}
    r1 = lz.ask(STATE, q, qid="custom_1")
    r2 = lz.ask(STATE, q, qid="custom_2")
    assert made == [1] and set(r1.raw["custom_1"]) == {"a", "b"} and "custom_2" in r2.raw
    assert lz.state() == "ready" and lz.label == "stock Laya (not fine-tuned)"


def test_lazy_laya_warm_loads_in_the_background():
    from fraudshield.models.laya_client import LazyLaya
    lz = LazyLaya(lambda: LayaClient.local(agent=FakeAgent(), revision="stock"), label="stock")
    lz.warm().join(timeout=5)
    assert lz.state() == "ready"


def test_lazy_laya_load_failure_is_laya_unavailable_with_reason():
    from fraudshield.models.laya_client import LazyLaya

    def factory():
        raise RuntimeError("CUDA out of memory")

    lz = LazyLaya(factory, label="stock")
    with pytest.raises(LayaUnavailable, match="CUDA out of memory"):
        lz.ask(STATE, {"type": "choice", "instructions": "x", "criteria": {"a": "y", "b": "n"}})
    assert lz.state().startswith("failed")
