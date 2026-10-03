"""Live booking stream from the command line.

  run          send the seed-0 test window to a running server's POST /score, like a booking system would
               uv run python scripts/stream_bookings.py run --url http://127.0.0.1:8080 --rate 20
  load         measure throughput and latency at several rates  -> artifacts/stream_load.md
               uv run python scripts/stream_bookings.py load --rates 1 5 20 50 --seconds 60
  consistency  stream the whole window and compare with the offline evaluation -> artifacts/stream_consistency.md
               uv run python scripts/stream_bookings.py consistency

load and consistency run the app in-process with Laya in cached mode (no GPU), learning off, and audit logs in
a temporary folder, so they never touch artifacts/audit/. Every number in the reports is measured.
"""
from __future__ import annotations

import argparse
import math
import os
import platform
import sys
import tempfile
import time
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")  # never load a model on the GPU from this script

ART = ROOT / "artifacts"


def fmt(x, d=3):
    return "n/a" if x is None or (isinstance(x, float) and math.isnan(x)) else f"{x:.{d}f}"


def machine() -> str:
    return f"{platform.system()} {platform.release()}, {platform.processor() or platform.machine()}, {os.cpu_count()} CPU threads"


def safe_app(tmp: Path):
    from fastapi.testclient import TestClient

    from fraudshield.api.app import build_app, load_config
    cfg = load_config(os.environ.get("FS_CONFIG") or ROOT / "config" / "app.yaml")
    cfg["laya"] = {**cfg.get("laya", {}), "mode": "cached"}
    cfg["laya_ask"] = {**cfg.get("laya_ask", {}), "enabled": False}
    cfg["learning"] = {**cfg.get("learning", {}), "enabled": False}
    cfg["audit_path"] = str(tmp / "audit.jsonl")
    cfg["stream"] = {**cfg.get("stream", {}), "audit_path": str(tmp / "stream_audit.jsonl")}
    cfg["replay_path"] = str(tmp / "none.jsonl")
    from fraudshield.api import real_components
    real_components.warm()
    return TestClient(build_app(cfg))


def wait(c, poll=2.0, label=""):
    while True:
        s = c.get("/stream/status").json()
        if s["state"] in ("done", "stopped"):
            return s
        m = c.get("/stream/metrics").json()["load"]
        print(f"  {label} sent {s['sent']}/{s['total']}  scored {m['scored']}  "
              f"{fmt(m['throughput_60s'], 1)}/s  p95 {fmt(m['latency_p95_ms'], 0)} ms", flush=True)
        time.sleep(poll)


# ---------------------------------------------------------------- run (HTTP, external client)
def cmd_run(a):
    import httpx

    from fraudshield.api import real_components
    from fraudshield.sim.stream import StreamMetrics, StreamRunner
    items = real_components.stream_source(0)[: a.limit or None]
    client = httpx.Client(base_url=a.url, timeout=30)

    def send(b):
        r = client.post("/score", json={**b.__dict__})
        r.raise_for_status()
        return r.json()

    m = StreamMetrics()
    runner = StreamRunner(items, send, m, rate=a.rate, concurrency=a.concurrency)
    runner.start()
    while runner.is_alive():
        runner.join(5)
        s = m.snapshot()
        print(f"scored {s['load']['scored']}/{len(items)}  {fmt(s['load']['throughput_60s'], 1)}/s  "
              f"p95 {fmt(s['load']['latency_p95_ms'], 0)} ms  errors {s['load']['errors']}  "
              f"PR-AUC {fmt(s['accuracy']['pr_auc'])}", flush=True)
    print("done:", runner.status())


# ---------------------------------------------------------------- load report
def cmd_load(a):
    with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as td:
        c = safe_app(Path(td))
        rows = []
        for rate in a.rates:
            n = max(int(rate * a.seconds), 30)
            print(f"rate {rate}/s: {n} bookings", flush=True)
            r = c.post("/stream/start", json={"rate": rate, "concurrency": a.concurrency, "limit": n})
            r.raise_for_status()
            wait(c, label=f"[{rate}/s]")
            m = c.get("/stream/metrics").json()
            ld = m["load"]
            rows.append({"rate": rate, "n": n, "scored": ld["scored"], "errors": ld["errors"],
                         "achieved": ld["scored"] / ld["elapsed_s"] if ld["elapsed_s"] else float("nan"),
                         "p50": ld["latency_p50_ms"], "p95": ld["latency_p95_ms"], "p99": ld["latency_p99_ms"],
                         "degraded": m["degraded_share"]})
    lines = [
        "# Live stream load test",
        "",
        f"Measured {datetime.now():%Y-%m-%d %H:%M} on {machine()}. In-process FastAPI app, Laya in cached mode",
        f"(decisions from the LightGBM backup score), concurrency {a.concurrency}, about {a.seconds} s per rate.",
        "Bookings are the seed-0 test window replayed in booked_at order. Latency is the service's own",
        "end-to-end scoring time per booking (latency_ms.total), before the label would be issued.",
        "",
        "| Target rate (/s) | Bookings | Achieved (/s) | Errors | p50 ms | p95 ms | p99 ms |",
        "|---|---|---|---|---|---|---|",
    ]
    for r in rows:
        lines.append(f"| {r['rate']} | {r['scored']}/{r['n']} | {fmt(r['achieved'], 1)} | {r['errors']} | "
                     f"{fmt(r['p50'], 0)} | {fmt(r['p95'], 0)} | {fmt(r['p99'], 0)} |")
    sat = [r for r in rows if r["achieved"] < 0.9 * r["rate"]]
    lines += ["", (f"The service saturates at about {fmt(max(r['achieved'] for r in rows), 1)} bookings per second on this "
                   f"machine: from a target of {sat[0]['rate']}/s, the achieved rate stays below the target."
                   if sat else "The service kept up with every tested rate.")]
    (ART / "stream_load.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))


# ---------------------------------------------------------------- consistency report
def cmd_consistency(a):
    with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as td:
        c = safe_app(Path(td))
        body = {"rate": a.rate, "concurrency": a.concurrency}
        if a.limit:
            body["limit"] = a.limit
        c.post("/stream/start", json=body).raise_for_status()
        wait(c, poll=10, label="[consistency]")
        m = c.get("/stream/metrics").json()
    live, same, full, cons = m["accuracy"], m["offline_same_rows"], m["offline_full"], m["consistency"]
    keys = [("PR-AUC", "pr_auc"), ("ROC-AUC", "roc_auc"), ("Precision, top 1% per day", "precision_top1pct_day"),
            ("Recall, top 1% per day", "recall_top1pct_day")]
    diffs = [abs(live[k] - same[k]) for _, k in keys if live[k] is not None and same[k] is not None]
    # scores are stored with 4 decimals, so a per-booking difference up to half of the last digit is rounding
    ok = cons["max_abs_score_diff"] <= 5.0001e-5 and (not diffs or max(diffs) < 1e-9) and m["load"]["errors"] == 0
    lines = [
        "# Live stream consistency check",
        "",
        f"Measured {datetime.now():%Y-%m-%d %H:%M} on {machine()}. The seed-0 test window was streamed in booked_at",
        "order through the decision service (Laya cached, so each decision uses the LightGBM backup score), and",
        "compared with the offline evaluation of the same model (artifacts/models B2_F, scores in",
        "data/processed/gbm_scores_seed0.parquet) on the same bookings. The decision record stores scores with",
        "4 decimals, so the offline scores are rounded the same way and a per-booking difference up to 0.00005 is",
        "rounding, not a different score.",
        "",
        f"- Bookings streamed and scored: {m['load']['scored']} of {m['status']['total']}, errors {m['load']['errors']}",
        f"- Fraud bookings among them: {live['n_fraud']}",
        f"- Largest per-booking difference between the live and the offline score: {cons['max_abs_score_diff']:.2e}"
        f" (over {cons['n_compared']} bookings)",
        "",
        "| Metric | Live stream | Offline, same bookings | Offline, whole window |",
        "|---|---|---|---|",
    ]
    for name, k in keys:
        lines.append(f"| {name} | {fmt(live[k])} | {fmt(same[k])} | {fmt((full or {}).get(k))} |")
    lines += ["", f"Fraud stopped (owner confirm, review, hold or block): {live['fraud_caught']}; sent to a first-scan check: "
              f"{live['fraud_scan_checked']}; allowed: {live['fraud_missed']}. Honest bookings stopped: "
              f"{live['honest_stopped']} ({fmt(100 * live['honest_stopped_rate'], 2)}%).",
              "", "Result: " + ("PASS. The live stream reproduces the offline scores and metrics; accuracy does not drop "
                                "under continuous scoring." if ok else "FAIL. See the differences above.")]
    (ART / "stream_consistency.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))
    sys.exit(0 if ok else 1)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run")
    r.add_argument("--url", default="http://127.0.0.1:8080")
    r.add_argument("--rate", type=float, default=20)
    r.add_argument("--concurrency", type=int, default=4)
    r.add_argument("--limit", type=int, default=0)
    ld = sub.add_parser("load")
    ld.add_argument("--rates", type=float, nargs="+", default=[1, 5, 20, 50])
    ld.add_argument("--seconds", type=float, default=60)
    ld.add_argument("--concurrency", type=int, default=4)
    co = sub.add_parser("consistency")
    co.add_argument("--rate", type=float, default=200)
    co.add_argument("--concurrency", type=int, default=4)
    co.add_argument("--limit", type=int, default=0)
    a = ap.parse_args()
    {"run": cmd_run, "load": cmd_load, "consistency": cmd_consistency}[a.cmd](a)


if __name__ == "__main__":
    main()
