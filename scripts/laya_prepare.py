"""Build the Laya fine-tuning set: notebook-schema JSONL + tokenized items + token-budget report.

Usage: uv run python scripts/laya_prepare.py [--fraud-repeat 3 --n-hard-neg 1800 --n-legit 3000]
Writes artifacts/laya/train.jsonl, artifacts/laya/train_items.pt, artifacts/laya/prepare.json.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from fraudshield.contracts import QUESTIONS, SERVED_QUESTIONS, TRAINED_QUESTIONS  # noqa: E402
from fraudshield.models.laya_train import data as D  # noqa: E402
from fraudshield.models.laya_train.modeling import load_tokenizer, stock_model_dir  # noqa: E402

OUT = Path(os.environ.get("FS_LAYA_ART", ROOT / "artifacts" / "laya"))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--fraud-repeat", type=int, default=3)
    ap.add_argument("--n-hard-neg", type=int, default=1800)
    ap.add_argument("--n-legit", type=int, default=3000)
    ap.add_argument("--drop-risk-level", action="store_true")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--max-len", type=int, default=512)
    ap.add_argument("--head-max-len", type=int, default=192)
    ap.add_argument("--from-jsonl", default=None,
                    help="rebuild items from an existing notebook-schema train.jsonl (Kaggle); keeps the "
                         "sampling fields of the shipped prepare.json and re-verifies the token budget")
    ap.add_argument("--limit", type=int, default=0, help="first N records only (smoke tests)")
    a = ap.parse_args()
    t0 = time.time()
    OUT.mkdir(parents=True, exist_ok=True)
    if a.from_jsonl:
        rebuild_from_jsonl(Path(a.from_jsonl), a, t0)
        return
    df = pd.read_json(ROOT / "data" / "processed" / "states.jsonl", lines=True)
    qids = tuple(q for q in TRAINED_QUESTIONS if not (a.drop_risk_level and q == "risk_level"))
    s = D.sample_training_bookings(df, a.n_hard_neg, a.n_legit, a.fraud_repeat, seed=a.seed)
    recs = [D.to_notebook_record(r, qids) for _, r in s.iterrows()]
    with open(OUT / "train.jsonl", "w", encoding="utf-8") as f:
        for r in recs:
            f.write(json.dumps(r) + "\n")
    tok = load_tokenizer(stock_model_dir())
    items, stats = D.build_items(tok, recs, a.max_len, a.head_max_len)
    assert stats["dropped"] == 0, stats
    assert stats["truncated"] == 0, stats

    # token budget over EVERY state in states.jsonl for every trained + served question
    from laya.common import state_room  # noqa: PLC0415
    rooms = {q: state_room(tok, {"t": QUESTIONS[q]["type"], "ins": QUESTIONS[q]["instructions"],
                                 "crit": QUESTIONS[q]["criteria"]}, a.max_len, a.head_max_len)
             for q in dict.fromkeys(TRAINED_QUESTIONS + SERVED_QUESTIONS)}
    lens = np.array([len(x) for x in tok(df.state.tolist(), add_special_tokens=False)["input_ids"]])
    over = {q: int((lens > r).sum()) for q, r in rooms.items()}
    assert all(v == 0 for v in over.values()), over

    is_fraud = s.labels.map(lambda l: l["misuse"] == 1)
    is_act = [it for it in items if "cost" in it]
    cost_scale = float(np.mean([max(it["cost"]) for it in is_act])) if is_act else 1.0
    import torch  # noqa: PLC0415
    torch.save(items, OUT / "train_items.pt")
    manifest = {
        "states_sha256": hashlib.sha256((ROOT / "data/processed/states.jsonl").read_bytes()).hexdigest()[:16],
        "seed": a.seed, "fraud_repeat": a.fraud_repeat, "questions": list(qids),
        "bookings": len(s), "unique_bookings": int(s.id.nunique()),
        "fraud_rows": int(is_fraud.sum()), "fraud_unique": int(s[is_fraud].id.nunique()),
        "fraud_share": float(is_fraud.mean()),
        "typologies": s.typology.value_counts().to_dict(),
        "items": stats["items"], "dropped": stats["dropped"], "truncated": stats["truncated"],
        "max_seq_len_train": stats["max_len_seen"], "max_state_tokens_train": stats["max_state_tokens"],
        "state_room_by_question": rooms, "all_states_max_tokens": int(lens.max()),
        "all_states_p99_tokens": float(np.percentile(lens, 99)), "states_over_budget": over,
        "cost_scale_C": cost_scale, "max_len": a.max_len, "head_max_len": a.head_max_len,
        "seconds": round(time.time() - t0, 1),
    }
    (OUT / "prepare.json").write_text(json.dumps(manifest, indent=2))
    print(json.dumps(manifest, indent=2))


def rebuild_from_jsonl(path: Path, a, t0: float) -> None:
    recs = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if a.limit:
        recs = recs[: a.limit]
    tok = load_tokenizer(stock_model_dir())
    items, stats = D.build_items(tok, recs, a.max_len, a.head_max_len)
    assert stats["dropped"] == 0, stats
    assert stats["truncated"] == 0, stats
    base = json.loads((OUT / "prepare.json").read_text()) if (OUT / "prepare.json").exists() else {}
    if not a.limit and base.get("items") is not None:
        assert base["items"] == stats["items"], (base["items"], stats["items"])
    # token budget re-checked on this machine over the train records and the evaluation sets
    from laya.common import state_room  # noqa: PLC0415
    states = [json.loads(r["state"]) for r in recs]
    for name in ("test", "cal"):
        p = OUT / f"evalset_{name}.parquet"
        if p.exists():
            ev = pd.read_parquet(p)
            for col in ("state", "state_nogbm"):
                if col in ev.columns:
                    states += ev[col].tolist()
    # token room for the questions these records train plus every served question (v2 drops risk_level)
    rec_q = tuple(json.loads(recs[0]["questions"]))
    rooms = {q: state_room(tok, {"t": QUESTIONS[q]["type"], "ins": QUESTIONS[q]["instructions"],
                                 "crit": QUESTIONS[q]["criteria"]}, a.max_len, a.head_max_len)
             for q in dict.fromkeys(rec_q + SERVED_QUESTIONS)}
    lens = np.array([len(x) for x in tok(states, add_special_tokens=False)["input_ids"]])
    over = {q: int((lens > r).sum()) for q, r in rooms.items()}
    assert all(v == 0 for v in over.values()), over
    is_act = [it for it in items if "cost" in it]
    import torch  # noqa: PLC0415
    torch.save(items, OUT / "train_items.pt")
    manifest = {**base, "items": stats["items"], "dropped": stats["dropped"], "truncated": stats["truncated"],
                "max_seq_len_train": stats["max_len_seen"], "rebuilt_from": path.name,
                "rebuild_states_checked": int(len(lens)), "rebuild_states_max_tokens": int(lens.max()),
                "rebuild_states_over_budget": over, "seconds": round(time.time() - t0, 1)}
    manifest.setdefault("cost_scale_C", float(np.mean([max(it["cost"]) for it in is_act])) if is_act else 1.0)
    manifest.setdefault("questions", list(json.loads(recs[0]["questions"])))
    manifest.setdefault("states_sha256", "unknown")
    (OUT / "prepare.json").write_text(json.dumps(manifest, indent=2))
    print(json.dumps({k: manifest[k] for k in ("items", "dropped", "truncated", "rebuild_states_checked",
                                                "rebuild_states_max_tokens", "cost_scale_C")}, indent=2))


if __name__ == "__main__":
    main()
