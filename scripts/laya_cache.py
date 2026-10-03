"""Demo cache + serving latency for the fine-tuned Laya checkpoint.

1. Loads the checkpoint the way LayaClient.local does (laya.load(dir, device="cuda")) and predicts.
2. Builds the state of every demo booking (data/processed/demo_bookings.json) in every way the app
   can produce it: real components on a fresh demo store (each booking alone), real components in
   scenario order with appends (as the desktop app does), and the API's fallback serializer. Runs the
   4 served questions and writes artifacts/laya_cache.json (LayaClient.cached format).
3. Measures single-request latency (4 questions, one predict call) over 200 test states: p50/p99,
   peak VRAM. Writes artifacts/laya/runtime.json.
"""
from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from fraudshield.contracts import QUESTIONS, SERVED_QUESTIONS, Booking  # noqa: E402
from fraudshield.models.laya_train.cache import build_cache  # noqa: E402
from fraudshield.models.laya_train.modeling import sha12  # noqa: E402

ART = ROOT / "artifacts"
LA = Path(os.environ.get("FS_LAYA_ART", ART / "laya"))
CKPT = LA / "fraudshield-laya"


def demo_states() -> list[tuple[str, str]]:
    from fraudshield.api import real_components as rc  # noqa: PLC0415
    from fraudshield.api.app import fallback_featurize, fallback_serialize  # noqa: PLC0415
    from fraudshield.data.demo import load_demo_bookings  # noqa: PLC0415

    demos = load_demo_bookings()
    bks = [Booking(**d["booking"]) for d in demos]
    out = []
    for b in bks:                                   # each alone on a fresh store
        rc.reset()
        out.append((f"{b.booking_id}:fresh", rc.serializer(b, rc.featurizer(b))))
    rc.reset()
    for b in bks:                                   # scenario order with appends (desktop app)
        out.append((f"{b.booking_id}:sequential", rc.serializer(b, rc.featurizer(b))))
    for b in bks:
        out.append((f"{b.booking_id}:fallback", fallback_serialize(b, fallback_featurize(b))))
    rc.reset()
    return out


def main() -> None:
    import argparse  # noqa: PLC0415
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--ckpt", default=str(CKPT))
    ap.add_argument("--states-json", default=None,
                    help="precomputed [[tag, state], ...] (Kaggle bundle: demo_states.json); default builds them "
                         "from the demo store with the app's real components")
    ap.add_argument("--dump-states", default=None, help="write the demo states to this json and exit (CPU only)")
    ap.add_argument("--n-latency", type=int, default=200)
    ap.add_argument("--out", default=str(ART / "laya_cache.json"))
    a = ap.parse_args()
    if a.dump_states:
        Path(a.dump_states).write_text(json.dumps(demo_states(), indent=1))
        print(f"wrote {a.dump_states}")
        return
    import laya  # noqa: PLC0415
    import torch  # noqa: PLC0415

    ckpt = Path(a.ckpt)
    is_cuda = a.device.startswith("cuda")
    t0 = time.time()
    agent = laya.load(str(ckpt), device=a.device)
    load_s = time.time() - t0
    qs = {q: QUESTIONS[q] for q in SERVED_QUESTIONS}
    rev = sha12(ckpt / "model.safetensors")

    states = ([tuple(x) for x in json.loads(Path(a.states_json).read_text())] if a.states_json
              else demo_states())
    uniq = list(dict.fromkeys(s for _, s in states))
    entries, shown = [], []
    for st in uniq:
        r = agent.predict(st, qs)
        entries.append((st, r))
    by_state = dict(entries)
    for tag, st in states:
        r = by_state[st]
        shown.append({"variant": tag, **{q: r["answers"][q]["probabilities"]["a"] for q in SERVED_QUESTIONS}})
    doc = build_cache(entries, revision=rev, meta={"model": "fraudshield-laya (fine-tuned)", "checkpoint": "artifacts/laya/fraudshield-laya",
                                                    "probabilities": "raw, temperature 1.0; apply artifacts/calibration.json",
                                                    "source": "data/processed/demo_bookings.json (fresh, sequential and fallback states)",
                                                    "built": time.strftime("%Y-%m-%dT%H:%M:%S")})
    Path(a.out).write_text(json.dumps(doc, indent=1))
    print(pd.DataFrame(shown).to_string())

    # latency: 200 test states, one predict call each (4 questions), after warmup
    ev = pd.read_parquet(LA / "evalset_test.parquet").sample(a.n_latency + 20, random_state=0)
    sts = ev.state.tolist()
    for st in sts[:20]:
        agent.predict(st, qs)
    sync = (lambda: torch.cuda.synchronize(a.device)) if is_cuda else (lambda: None)
    sync()
    if is_cuda:
        torch.cuda.reset_peak_memory_stats(a.device)
    lat = []
    for st in sts[20:]:
        t_req = time.perf_counter()
        agent.predict(st, qs)
        sync()
        lat.append((time.perf_counter() - t_req) * 1000)
    lat = np.array(lat)
    rt = {"load_seconds": round(load_s, 1), "n": len(lat), "p50_ms": round(float(np.percentile(lat, 50)), 1),
          "p90_ms": round(float(np.percentile(lat, 90)), 1), "p99_ms": round(float(np.percentile(lat, 99)), 1),
          "max_ms": round(float(lat.max()), 1),
          "peak_vram_alloc_gb": round(torch.cuda.max_memory_allocated(a.device) / 1e9, 2) if is_cuda else 0.0,
          "peak_vram_reserved_gb": round(torch.cuda.max_memory_reserved(a.device) / 1e9, 2) if is_cuda else 0.0,
          "device": a.device, "gpu": torch.cuda.get_device_name(a.device) if is_cuda else "cpu",
          "cache_entries": doc["_meta"]["entries"], "model_revision": rev}
    rt["summary"] = (f"Fine-tuned checkpoint loads with laya.load(<checkpoint dir>) in {rt['load_seconds']} s; "
                     f"one request with the 4 served questions: p50 {rt['p50_ms']} ms, p90 {rt['p90_ms']} ms, "
                     f"p99 {rt['p99_ms']} ms over {rt['n']} test states on {rt['gpu']} (measured where this script ran; "
                     f"a Kaggle T4 is not the demo laptop); peak VRAM "
                     f"{rt['peak_vram_alloc_gb']} GB allocated / {rt['peak_vram_reserved_gb']} GB reserved. "
                     f"Note the client timeout in config/app.yaml is 180 ms.")
    (LA / "runtime.json").write_text(json.dumps(rt, indent=2))
    print(json.dumps(rt, indent=2))


if __name__ == "__main__":
    main()
