"""Score an evaluation set with a Laya checkpoint (stock zero-shot B3 or fine-tuned M1).

Usage:
  uv run python scripts/laya_score.py --model ft --set test
  uv run python scripts/laya_score.py --model stock --set test
  uv run python scripts/laya_score.py --model ft --set cal
Eval sets (built once, deterministic): artifacts/laya/evalset_{test,cal}.parquet
  test: all injected rows of seeds 0-9 + all real hard negatives + 5,000 random other real legit (weighted)
  cal:  seed-0 calibration window, same recipe with 4,000 random legit (weighted)
Output: artifacts/laya/scores_{model}_{set}.parquet with P(option a = yes) per served question.
Probabilities are as served: the fine-tuned checkpoint exports temperature 1.0; the stock checkpoint
applies its vendor temperatures (monotone, so ranking metrics are unaffected).
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from fraudshield.contracts import QUESTIONS, SERVED_QUESTIONS  # noqa: E402
from fraudshield.models.laya_train.data import eval_sample  # noqa: E402

ART = Path(os.environ.get("FS_LAYA_ART", ROOT / "artifacts" / "laya"))
N_LEGIT = {"test": 5000, "cal": 4000}
KEEP = ["id", "seed", "split", "booking_ts", "account_id", "typology", "is_injected", "campaign_id",
        "camouflage_level", "hard_negative", "gbm_b2f", "weight", "state", "labels"]


def evalset(name: str) -> pd.DataFrame:
    p = ART / f"evalset_{name}.parquet"
    if p.exists():
        return pd.read_parquet(p)
    df = pd.read_json(ROOT / "data" / "processed" / "states.jsonl", lines=True)
    s = eval_sample(df, name, N_LEGIT[name], seed=0)[KEEP].copy()
    s["labels"] = s.labels.map(json.dumps)
    s["hard_negative"] = s.hard_negative.map(json.dumps)
    s["campaign_id"] = s.campaign_id.astype(object).where(s.campaign_id.notna(), None).astype(str)
    assert s.id.is_unique
    s.to_parquet(p)
    return s


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", choices=["stock", "ft"], required=True)
    ap.add_argument("--set", choices=["test", "cal"], required=True)
    ap.add_argument("--path", default=str(ART / "fraudshield-laya"))
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--limit", type=int, default=0, help="score only the first N bookings (smoke tests)")
    ap.add_argument("--chunk", type=int, default=512)
    ap.add_argument("--device", default="cuda", help="cuda, cuda:1 (run stock scoring next to training), cpu")
    ap.add_argument("--state-col", default="state", help="v2: state (with the LightGBM line) or state_nogbm")
    ap.add_argument("--with-action", action="store_true", help="v2: also score the action question")
    ap.add_argument("--tag", default="", help="suffix for the output files, e.g. _nogbm")
    a = ap.parse_args()
    s = evalset(a.set)
    out = ART / f"scores_{a.model}{a.tag}_{a.set}.parquet"
    part = ART / f"scores_{a.model}{a.tag}_{a.set}.partial.parquet"
    done = pd.read_parquet(part) if part.exists() else pd.DataFrame()
    if a.limit:
        s = s.iloc[: a.limit]
    todo = s[~s.id.isin(done.id)] if len(done) else s
    print(f"{a.model}/{a.set}: {len(s)} bookings, {len(todo)} to score", flush=True)

    import laya  # noqa: PLC0415
    import torch  # noqa: PLC0415
    agent = laya.Agent("convaiinnovations/laya" if a.model == "stock" else a.path, device=a.device)
    is_cuda = a.device.startswith("cuda")
    qids = SERVED_QUESTIONS + (("action",) if a.with_action else ())
    qs = {q: QUESTIONS[q] for q in qids}
    act_keys = tuple(QUESTIONS["action"]["criteria"])
    rows = [done] if len(done) else []
    t0 = time.time()
    n = 0
    for i in range(0, len(todo), a.chunk):
        ch = todo.iloc[i: i + a.chunk]
        res = agent.predict_batch(ch[a.state_col].tolist(), qs, batch_size=a.batch, sort_by_length=True)
        rec = pd.DataFrame({"id": ch.id.to_numpy()})
        for q in SERVED_QUESTIONS:
            rec[f"p_{q}"] = [float(r["answers"][q]["probabilities"]["a"]) for r in res]
        if a.with_action:
            for k in act_keys:
                rec[f"p_act_{k}"] = [float(r["answers"]["action"]["probabilities"][k]) for r in res]
        rows.append(rec)
        n += len(ch)
        pd.concat(rows, ignore_index=True).to_parquet(part)
        el = time.time() - t0
        print(f"  {n}/{len(todo)} bookings, {n * len(qs) / el:.1f} items/s, "
              f"peak {(torch.cuda.max_memory_allocated(a.device) / 1e9) if is_cuda else 0:.2f} GB", flush=True)
    allr = pd.concat(rows, ignore_index=True)
    allr.to_parquet(out)
    part.unlink(missing_ok=True)
    meta = {"model": a.model, "set": a.set, "state_col": a.state_col, "tag": a.tag, "n": len(allr), "seconds": round(time.time() - t0, 1), "device": a.device,
            "path": a.path if a.model == "ft" else "convaiinnovations/laya"}
    (ART / f"scores_{a.model}{a.tag}_{a.set}.json").write_text(json.dumps(meta, indent=2))
    print(json.dumps(meta))


if __name__ == "__main__":
    main()
