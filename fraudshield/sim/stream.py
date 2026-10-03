"""Live booking stream: replay dataset bookings into the decision service at a steady rate and measure it.

- StreamItem: one booking from the dataset plus its ground truth (used ONLY for the counters here, never
  as a model input) and the offline score of the same model on the same row (for the consistency check).
- StreamRunner: sends items in the given (booked_at) order at `rate` per second with `concurrency` senders.
  It never queues more than `concurrency` bookings at once (back-pressure instead of an unbounded queue),
  retries a failed send with exponential backoff, and counts a booking as an error after the last retry:
  nothing is dropped silently.
- StreamMetrics: load, action mix and accuracy, computed with the same functions as the offline evaluation
  (fraudshield.data.metrics), plus the offline reference on exactly the rows streamed so far.

The stream is a simulation: it replays recorded bookings, time-compressed, not live carrier traffic.
"""
from __future__ import annotations

import threading
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from typing import Any, Callable

import numpy as np

from fraudshield.contracts import Booking
from fraudshield.data.metrics import flag_top_per_day, pr_auc, roc_auc

STOPPED = ("owner_confirm", "review", "hold", "block")  # as in docs/LEARNING_GATE.md
ACTIONS = ("allow", "allow_scan_gated", "owner_confirm", "review", "hold", "block")


@dataclass(frozen=True)
class StreamItem:
    booking: Booking
    is_fraud: bool
    typology: str = "none"
    offline_score: float | None = None


RECORD_DECIMALS = 4  # the decision record stores scores with 4 decimals (pipeline.public_view)


def _rec(x: float | None) -> float | None:
    """An offline score at the precision of the decision record, so live and offline compare like for like."""
    return None if x is None else round(float(x), RECORD_DECIMALS)


def _accuracy(rows: list[dict], key: str) -> dict[str, Any]:
    y = np.array([r["is_fraud"] for r in rows], dtype=bool)
    s = np.array([r[key] for r in rows], dtype=float)
    out: dict[str, Any] = {"n": int(len(rows)), "n_fraud": int(y.sum()), "pr_auc": pr_auc(y, s), "roc_auc": roc_auc(y, s),
                           "precision_top1pct_day": float("nan"), "recall_top1pct_day": float("nan")}
    if len(rows):
        f = flag_top_per_day([r["booked_at"] for r in rows], s, 0.01)
        out["precision_top1pct_day"] = float(y[f].mean()) if f.any() else float("nan")
        out["recall_top1pct_day"] = float(f[y].mean()) if y.any() else float("nan")
    return out


def offline_metrics(items: list[StreamItem]) -> dict[str, Any]:
    """The offline model's metrics on a list of items (e.g. the whole window), for the reference panel."""
    rows = [{"is_fraud": it.is_fraud, "offline": _rec(it.offline_score), "booked_at": it.booking.booked_at}
            for it in items if it.offline_score is not None]
    return _accuracy(rows, "offline")


class StreamMetrics:
    def __init__(self):
        self._lock = threading.Lock()
        self._rows: list[dict] = []
        self._errors: list[dict] = []
        self._retries = 0
        self._in_flight = 0
        self._t0 = time.monotonic()
        self.offline_full: dict | None = None
        self.explanations = {"llm": 0, "template": 0, "llm_cap_per_min": None}

    # ---------- recording ----------
    def record(self, item: StreamItem, d: dict, latency_ms: float | None = None) -> None:
        probs = d.get("probabilities") or {}
        b = item.booking
        row = {"booking_id": b.booking_id, "decision_id": d.get("decision_id"), "booked_at": b.booked_at,
               "action": d.get("action"), "score": float(probs.get("misuse", d.get("gbm_score") or 0.0)),
               "gbm": d.get("gbm_score"), "offline": _rec(item.offline_score), "is_fraud": bool(item.is_fraud),
               "typology": item.typology, "degraded": bool(d.get("degraded")),
               "latency_ms": float(latency_ms if latency_ms is not None else (d.get("latency_ms") or {}).get("total", 0.0)),
               "t": time.monotonic(), "route": f"{b.origin_uf} {b.origin_zip3} to {b.dest_uf} {b.dest_zip3}",
               "carrier_cost": float(b.carrier_cost)}
        with self._lock:
            self._rows.append(row)

    def record_error(self, item: StreamItem, err: str) -> None:
        with self._lock:
            self._errors.append({"booking_id": item.booking.booking_id, "error": err, "t": time.monotonic()})

    def add_retry(self) -> None:
        with self._lock:
            self._retries += 1

    def set_in_flight(self, n: int) -> None:
        with self._lock:
            self._in_flight = n

    def rows(self) -> list[dict]:
        with self._lock:
            return list(self._rows)

    def feed(self, limit: int = 30) -> list[dict]:
        with self._lock:
            rows = self._rows[-limit:]
        return [{k: r[k] for k in ("decision_id", "booking_id", "booked_at", "route", "carrier_cost", "action", "score")}
                for r in reversed(rows)]

    # ---------- snapshot ----------
    def snapshot(self, window_s: float = 60.0) -> dict[str, Any]:
        with self._lock:
            rows, errors, retries, in_flight = list(self._rows), list(self._errors), self._retries, self._in_flight
        now = time.monotonic()
        lat = np.array([r["latency_ms"] for r in rows], dtype=float)
        recent = [r for r in rows if now - r["t"] <= window_s]
        lat_recent = np.array([r["latency_ms"] for r in recent], dtype=float)
        pct = (lambda a, q: float(np.percentile(a, q)) if len(a) else float("nan"))
        elapsed = max(now - self._t0, 1e-9)
        timeline = []
        for k in range(int(min(elapsed, 120)), 0, -1):  # last 120 seconds, one bucket per second
            b = [r for r in rows if k - 1 < now - r["t"] <= k]
            timeline.append({"s_ago": k, "scored": len(b),
                             "latency_p95_ms": pct(np.array([r["latency_ms"] for r in b]), 95) if b else None})
        actions = {a: sum(1 for r in rows if r["action"] == a) for a in ACTIONS}

        acc = _accuracy(rows, "score")
        fraud = [r for r in rows if r["is_fraud"]]
        honest = [r for r in rows if not r["is_fraud"]]
        acc.update({
            "fraud_caught": sum(r["action"] in STOPPED for r in fraud),
            "fraud_scan_checked": sum(r["action"] == "allow_scan_gated" for r in fraud),
            "fraud_missed": sum(r["action"] == "allow" for r in fraud),
            "honest_stopped": sum(r["action"] in STOPPED for r in honest),
            "honest_stopped_rate": (sum(r["action"] in STOPPED for r in honest) / len(honest)) if honest else float("nan"),
            "stopped_by_type": {t: {"n": sum(r["typology"] == t for r in fraud),
                                    "stopped": sum(r["typology"] == t and r["action"] in STOPPED for r in fraud)}
                                for t in sorted({r["typology"] for r in fraud})},
        })
        compared = [r for r in rows if r["offline"] is not None and r["gbm"] is not None]
        return {
            "load": {"scored": len(rows), "errors": len(errors), "retries": retries, "in_flight": in_flight,
                     "elapsed_s": round(elapsed, 2), "throughput_total": len(rows) / elapsed,
                     "throughput_60s": len(recent) / min(window_s, elapsed),
                     "latency_p50_ms": pct(lat, 50), "latency_p95_ms": pct(lat, 95), "latency_p99_ms": pct(lat, 99),
                     "latency_p95_60s_ms": pct(lat_recent, 95)},
            "timeline": timeline,
            "actions": actions,
            "degraded_share": (sum(r["degraded"] for r in rows) / len(rows)) if rows else float("nan"),
            "accuracy": acc,
            "offline_same_rows": _accuracy([r for r in rows if r["offline"] is not None], "offline"),
            "offline_full": self.offline_full,
            "consistency": {"n_compared": len(compared),
                            "max_abs_score_diff": max((abs(r["gbm"] - r["offline"]) for r in compared), default=0.0)},
            "explanations": dict(self.explanations),
            "errors": errors[-20:],
        }


class StreamRunner:
    def __init__(self, items: list[StreamItem], send: Callable[[Booking], dict], metrics: StreamMetrics,
                 rate: float = 20.0, concurrency: int = 4, retries: int = 3, backoff_s: float = 0.2):
        if rate <= 0 or concurrency < 1:
            raise ValueError("rate must be > 0 and concurrency >= 1")
        self.items, self.send, self.metrics = items, send, metrics
        self.rate, self.concurrency, self.retries, self.backoff_s = float(rate), int(concurrency), int(retries), backoff_s
        self._state = "ready"
        self._sent = 0
        self._resume = threading.Event()
        self._resume.set()
        self._stop = threading.Event()
        self._slots = threading.Semaphore(self.concurrency)
        self._in_flight = 0
        self._lock = threading.Lock()
        self._thread = threading.Thread(target=self._run, name="booking-stream", daemon=True)

    # ---------- control ----------
    def start(self) -> None:
        self._state = "running"
        self._thread.start()

    def pause(self) -> None:
        if self._state == "running":
            self._resume.clear()
            self._state = "paused"

    def resume(self) -> None:
        if self._state == "paused":
            self._state = "running"
            self._resume.set()

    def stop(self) -> None:
        self._stop.set()
        self._resume.set()

    def join(self, timeout: float | None = None) -> None:
        self._thread.join(timeout)

    def is_alive(self) -> bool:
        return self._thread.is_alive()

    def status(self) -> dict[str, Any]:
        return {"state": self._state, "sent": self._sent, "total": len(self.items), "rate": self.rate,
                "concurrency": self.concurrency, "in_flight": self._in_flight}

    # ---------- work ----------
    def _one(self, item: StreamItem) -> None:
        try:
            for attempt in range(self.retries):
                t0 = time.perf_counter()
                try:
                    d = self.send(item.booking)
                except Exception as e:  # service busy or down: back off and retry
                    if attempt == self.retries - 1:
                        self.metrics.record_error(item, f"{type(e).__name__}: {e}")
                        return
                    self.metrics.add_retry()
                    time.sleep(self.backoff_s * (2 ** attempt))
                    continue
                lat = (d.get("latency_ms") or {}).get("total")
                self.metrics.record(item, d, latency_ms=lat if lat is not None else (time.perf_counter() - t0) * 1000)
                return
        finally:
            with self._lock:
                self._in_flight -= 1
                self.metrics.set_in_flight(self._in_flight)
            self._slots.release()

    def _run(self) -> None:
        interval = 1.0 / self.rate
        next_t = time.monotonic()
        with ThreadPoolExecutor(max_workers=self.concurrency, thread_name_prefix="stream-send") as pool:
            for item in self.items:
                if not self._resume.is_set():
                    self._resume.wait()
                    next_t = time.monotonic()  # after a pause, restart the schedule from now
                if self._stop.is_set():
                    break
                now = time.monotonic()
                if next_t > now:
                    if self._stop.wait(next_t - now):
                        break
                elif now - next_t > 1.0:
                    next_t = now  # far behind (the service is slower than the rate): no burst to catch up
                next_t += interval  # absolute schedule: small timer oversleeps are made up, not accumulated
                self._slots.acquire()  # back-pressure: at most `concurrency` bookings in flight
                if self._stop.is_set():
                    self._slots.release()
                    break
                with self._lock:
                    self._in_flight += 1
                    self.metrics.set_in_flight(self._in_flight)
                self._sent += 1
                pool.submit(self._one, item)
        self._state = "stopped" if self._stop.is_set() else "done"
