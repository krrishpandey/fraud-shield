"""Laya client. Modes:
- local: laya.load() in-process (CUDA), the 4 served questions in ONE predict call.
- http: laya-serve POST /v1/systemone (Jev-compatible body).
- cached: JSON file keyed by sha256(state + question ids); responses flagged cached.
- auto (factory): local if CUDA and laya load, otherwise cached, with the reason recorded.
torch and laya are imported lazily so the API runs without the `ml` extra.
Any timeout or error raises LayaUnavailable; the pipeline then degrades to the calibrated GBM.
"""
from __future__ import annotations

import functools

import hashlib
import json
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from concurrent.futures import TimeoutError as FutTimeout
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

from fraudshield.contracts import QUESTIONS, SERVED_QUESTIONS


class LayaUnavailable(RuntimeError):
    pass


@dataclass
class LayaResult:
    raw: dict[str, dict[str, float]]
    latency_ms: float
    mode: str
    model: str | None = None
    revision: str | None = None
    cached: bool = False
    usage: dict[str, Any] = field(default_factory=dict)


def cache_key(state: str, qids: list[str]) -> str:
    return hashlib.sha256((state + "\x1f" + "|".join(qids)).encode("utf-8")).hexdigest()


@functools.lru_cache(maxsize=1)
def cuda_available() -> bool:
    """Checked once per process: importing torch and probing CUDA on every /health call is slow."""
    try:
        import torch  # noqa: PLC0415
    except Exception:
        return False
    try:
        return bool(torch.cuda.is_available())
    except Exception:
        return False


def _default_loader(path: str, device: str):
    import laya  # noqa: PLC0415  (optional dependency)
    return laya.load(path, device=device)


def _parse_answers(resp: dict[str, Any], qids: list[str]) -> dict[str, dict[str, float]]:
    ans = resp.get("answers", resp)
    out = {}
    for q in qids:
        if q not in ans or "probabilities" not in ans[q]:
            raise LayaUnavailable(f"Laya answer missing for {q}")
        out[q] = {str(k): float(v) for k, v in ans[q]["probabilities"].items()}
    return out


class LayaClient:
    def __init__(self, mode: str, call: Callable[[str, dict], dict] | None, timeout_s: float = 0.18,
                 model: str | None = None, revision: str | None = None, cache: dict | None = None,
                 cache_path: Path | None = None, record_cache: Path | None = None, mode_reason: str = ""):
        self.mode = mode
        self._call = call
        self.timeout_s = timeout_s
        self.model = model
        self.revision = revision
        self._cache = cache or {}
        self.cache_path = cache_path
        self.record_cache = Path(record_cache) if record_cache else None
        self._recorded: dict[str, Any] = {}
        self._lock = threading.Lock()
        self._pool = ThreadPoolExecutor(max_workers=1) if mode == "local" else None
        self.mode_reason = mode_reason

    # ---------- constructors ----------
    @classmethod
    def local(cls, model_path: str = "convaiinnovations/laya", device: str = "cuda", agent=None,
              timeout_s: float = 0.18, revision: str | None = None, record_cache=None, loader=None, mode_reason=""):
        if agent is None:
            agent = (loader or _default_loader)(model_path, device)
        return cls("local", lambda s, qs: agent.predict(s, qs), timeout_s, model=model_path,
                   revision=revision or model_path, record_cache=record_cache, mode_reason=mode_reason)

    @classmethod
    def http(cls, base_url: str, api_key: str | None = None, timeout_s: float = 0.18, http_client=None,
             model: str | None = None):
        import httpx  # noqa: PLC0415
        client = http_client or httpx.Client()
        url = base_url.rstrip("/") + "/v1/systemone"
        headers = {"authorization": f"Bearer {api_key}"} if api_key else {}

        def call(state, questions):
            body = {"state": state, "questions": questions}
            if model:
                body["model"] = model
            try:
                r = client.post(url, json=body, headers=headers, timeout=timeout_s)
            except httpx.HTTPError as e:
                raise LayaUnavailable(f"laya-serve error: {e!r}") from e
            if r.status_code >= 400:
                raise LayaUnavailable(f"laya-serve HTTP {r.status_code}")
            return r.json()

        return cls("http", call, timeout_s, model=model or base_url, revision=model)

    @classmethod
    def cached(cls, cache_path, mode_reason: str = ""):
        p = Path(cache_path)
        data = json.loads(p.read_text(encoding="utf-8")) if p.exists() else {}
        return cls("cached", None, model="cache:" + p.name, revision=data.get("_revision") if isinstance(data, dict) else None,
                   cache=data, cache_path=p, mode_reason=mode_reason)

    @classmethod
    def auto(cls, model_path: str, cache_path, device: str = "cuda", timeout_s: float = 0.18,
             cuda_available: Callable[[], bool] = cuda_available, loader=None, record_cache=None):
        if not cuda_available():
            return cls.cached(cache_path, mode_reason="no CUDA GPU or torch not installed; using cached answers")
        try:
            return cls.local(model_path, device, timeout_s=timeout_s, loader=loader, record_cache=record_cache,
                             mode_reason="CUDA available; model loaded in-process")
        except Exception as e:  # missing laya, download failure, OOM
            return cls.cached(cache_path, mode_reason=f"local model load failed ({e}); using cached answers")

    # ---------- calls ----------
    def _run(self, state: str, questions: dict[str, dict]) -> LayaResult:
        qids = list(questions)
        t0 = time.perf_counter()
        if self.mode == "cached":
            hit = self._cache.get(cache_key(state, qids))
            if hit is None:
                raise LayaUnavailable("no cached Laya answer for this state")
            return LayaResult({q: dict(hit[q]) for q in qids}, (time.perf_counter() - t0) * 1000, "cached",
                              self.model, self.revision, cached=True)
        try:
            if self._pool is not None:
                resp = self._pool.submit(self._call, state, questions).result(timeout=self.timeout_s)
            else:
                resp = self._call(state, questions)
        except FutTimeout as e:
            raise LayaUnavailable(f"Laya timeout after {self.timeout_s}s") from e
        except LayaUnavailable:
            raise
        except Exception as e:
            raise LayaUnavailable(f"Laya error: {e!r}") from e
        raw = _parse_answers(resp, qids)
        if self.record_cache is not None:
            with self._lock:
                self._recorded[cache_key(state, qids)] = raw
        return LayaResult(raw, (time.perf_counter() - t0) * 1000, self.mode, self.model, self.revision,
                          usage=resp.get("usage", {}) if isinstance(resp, dict) else {})

    def predict(self, state: str, qids: tuple[str, ...] | list[str] = SERVED_QUESTIONS) -> LayaResult:
        return self._run(state, {q: QUESTIONS[q] for q in qids})

    def ask(self, state: str, question: dict[str, Any], qid: str = "custom") -> LayaResult:
        return self._run(state, {qid: question})

    def flush_cache(self) -> None:
        if self.record_cache is None:
            return
        with self._lock:
            old = json.loads(self.record_cache.read_text(encoding="utf-8")) if self.record_cache.exists() else {}
            old.update(self._recorded)
            self.record_cache.parent.mkdir(parents=True, exist_ok=True)
            self.record_cache.write_text(json.dumps(old), encoding="utf-8")

    def health(self) -> dict[str, Any]:
        return {"mode": self.mode, "reason": self.mode_reason, "model": self.model, "revision": self.revision,
                "cache_entries": len(self._cache) if self.mode == "cached" else None}
