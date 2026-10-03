"""Live booking stream: runner (order, pacing, pause/stop, retries) and metrics (same functions as offline)."""
import threading
import time

import numpy as np
import pytest

from fraudshield.data.metrics import pr_auc, precision_recall_at_top_per_day, roc_auc
from fraudshield.sim.stream import StreamItem, StreamMetrics, StreamRunner
from tests.policy.fixtures import make_booking


def items(n, fraud_every=5):
    out = []
    for i in range(n):
        day = 1 + i // 10
        b = make_booking(booking_id=f"s{i:04d}", booked_at=f"2018-06-{day:02d}T{10 + i % 10:02d}:00:00")
        fraud = i % fraud_every == 0
        out.append(StreamItem(b, is_fraud=fraud, typology="T1" if fraud else "none", offline_score=0.9 if fraud else 0.1))
    return out


def ok_send(b):
    fraud = b.booking_id.endswith(("0", "5"))
    return {"decision_id": "dec_" + b.booking_id, "booking_id": b.booking_id, "action": "hold" if fraud else "allow",
            "probabilities": {"misuse": 0.9 if fraud else 0.1}, "gbm_score": 0.9 if fraud else 0.1,
            "degraded": True, "latency_ms": {"total": 12.0}}


def run(runner, timeout=10):
    runner.start()
    runner.join(timeout)
    assert not runner.is_alive()


def test_sends_every_booking_once_in_booked_at_order():
    seen = []
    lock = threading.Lock()

    def send(b):
        with lock:
            seen.append(b.booking_id)
        return ok_send(b)

    m = StreamMetrics()
    run(StreamRunner(items(30), send, m, rate=1000, concurrency=1))
    assert seen == [f"s{i:04d}" for i in range(30)]
    assert m.snapshot()["load"]["scored"] == 30


def test_rate_paces_the_stream():
    t0 = time.monotonic()
    run(StreamRunner(items(10), ok_send, StreamMetrics(), rate=20, concurrency=2))
    assert time.monotonic() - t0 >= 0.4  # 10 bookings at 20 per second


def test_pause_resume_and_stop():
    r = StreamRunner(items(200), ok_send, StreamMetrics(), rate=50, concurrency=1)
    r.start()
    time.sleep(0.2)
    r.pause()
    time.sleep(0.1)
    sent = r.status()["sent"]
    time.sleep(0.2)
    assert r.status()["sent"] == sent and r.status()["state"] == "paused"
    r.resume()
    time.sleep(0.1)
    r.stop()
    r.join(5)
    s = r.status()
    assert s["state"] == "stopped" and sent < s["sent"] < 200


def test_failed_sends_are_retried_then_counted_never_dropped_silently():
    calls = {}

    def flaky(b):
        calls[b.booking_id] = calls.get(b.booking_id, 0) + 1
        if b.booking_id == "s0001" and calls[b.booking_id] < 3:
            raise RuntimeError("busy")
        if b.booking_id == "s0002":
            raise RuntimeError("always down")
        return ok_send(b)

    m = StreamMetrics()
    run(StreamRunner(items(4), flaky, m, rate=1000, concurrency=1, retries=3, backoff_s=0.01))
    load = m.snapshot()["load"]
    assert load["scored"] == 3 and load["errors"] == 1 and load["retries"] >= 2
    assert m.snapshot()["errors"][0]["booking_id"] == "s0002"


def test_accuracy_uses_the_offline_metric_functions_on_the_same_rows():
    its = items(60)
    m = StreamMetrics()
    rng = np.random.default_rng(0)
    for it in its:
        d = ok_send(it.booking)
        d["probabilities"]["misuse"] = d["gbm_score"] = float(np.clip(d["gbm_score"] + rng.normal(0, 0.3), 0, 1))
        m.record(it, d, latency_ms=10.0)
    snap = m.snapshot()
    y = [it.is_fraud for it in its]
    s = [r["score"] for r in m.rows()]
    ts = [it.booking.booked_at for it in its]
    acc = snap["accuracy"]
    assert acc["pr_auc"] == pytest.approx(pr_auc(y, s))
    assert acc["roc_auc"] == pytest.approx(roc_auc(y, s))
    p, rc = precision_recall_at_top_per_day(ts, s, y, 0.01)
    assert acc["precision_top1pct_day"] == pytest.approx(p) and acc["recall_top1pct_day"] == pytest.approx(rc)
    # offline reference on exactly the rows streamed so far
    assert snap["offline_same_rows"]["pr_auc"] == pytest.approx(pr_auc(y, [it.offline_score for it in its]))
    assert snap["consistency"]["max_abs_score_diff"] >= 0


def test_metrics_load_and_action_mix():
    m = StreamMetrics()
    for i, it in enumerate(items(20)):
        m.record(it, ok_send(it.booking), latency_ms=float(10 + i))
    snap = m.snapshot()
    assert snap["actions"]["hold"] == 4 and snap["actions"]["allow"] == 16
    assert snap["load"]["latency_p50_ms"] == pytest.approx(np.percentile(np.arange(10, 30), 50))
    assert snap["accuracy"]["fraud_caught"] == 4 and snap["accuracy"]["fraud_missed"] == 0
    assert snap["accuracy"]["honest_stopped"] == 0
    assert snap["degraded_share"] == 1.0


def test_pacing_keeps_the_target_rate_despite_timer_oversleep():
    # Windows timers oversleep by up to ~15 ms; the pacer must keep to the absolute schedule, not drift slower.
    n, rate = 60, 50
    t0 = time.monotonic()
    run(StreamRunner(items(n), ok_send, StreamMetrics(), rate=rate, concurrency=2))
    elapsed = time.monotonic() - t0
    assert elapsed < (n / rate) * 1.15, f"{n / elapsed:.1f}/s achieved for a {rate}/s target"
