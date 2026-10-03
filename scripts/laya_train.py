"""Partial fine-tune of Laya on the FraudShield items (single 6 GB GPU port of the notebook loop).

Frozen: embeddings + lower encoder layers (kept in the autocast dtype). Trained: top K encoder layers,
final norm, decision head. Precision is chosen automatically: bf16 autocast where supported (RTX 40xx,
A100), fp16 autocast + GradScaler on T4; fp32 on CPU (smoke tests only). Gradient checkpointing, grad
accumulation. Objective: fraudshield.models.laya_train.objective.
Resumable: after every epoch the full training state is saved to artifacts/laya/resume_state.pt;
--resume continues from the last finished epoch.

Pilot:  uv run python scripts/laya_train.py --pilot-steps 200 --k-top 8 --micro 8
Full:   uv run python scripts/laya_train.py --epochs 2 --k-top 8 --micro 8 --accum 4
Final checkpoint: artifacts/laya/fraudshield-laya/ (loadable with laya.Agent(<dir>)).
"""
from __future__ import annotations

import argparse
import json
import math
import os
import random
import shutil
import sys
import time
from pathlib import Path

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import torch  # noqa: E402
from safetensors.torch import load_file, save_file  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from fraudshield.models.laya_train.modeling import (  # noqa: E402
    collate, export_config, freeze_lower, load_tokenizer, stock_model_dir)
from fraudshield.models.laya_train.kaggle import select_precision  # noqa: E402
from fraudshield.models.laya_train.objective import compute_loss, sigma_at  # noqa: E402

ART = Path(os.environ.get("FS_LAYA_ART", ROOT / "artifacts" / "laya"))


def log(msg: str, fh) -> None:
    line = time.strftime("%H:%M:%S ") + msg
    print(line, flush=True)
    fh.write(line + "\n")
    fh.flush()


def save_checkpoint(model, tok, cfg, model_dir, out_dir: Path, extra: dict) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    sd = {k: v.detach().to(torch.float16).contiguous().cpu() for k, v in model.state_dict().items()}
    save_file(sd, str(out_dir / "model.safetensors"))
    model.encoder.config.save_pretrained(str(out_dir / "encoder"))
    tok.save_pretrained(str(out_dir / "tokenizer"))
    for f in os.listdir(os.path.join(model_dir, "tokenizer")):
        if not (out_dir / "tokenizer" / f).exists():
            shutil.copy(os.path.join(model_dir, "tokenizer", f), out_dir / "tokenizer" / f)
    (out_dir / "rl_agent_config.json").write_text(json.dumps(export_config(cfg, 512, 192, extra), indent=2))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--k-top", type=int, default=8)
    ap.add_argument("--micro", type=int, default=8)
    ap.add_argument("--accum", type=int, default=4)
    ap.add_argument("--epochs", type=int, default=2)
    ap.add_argument("--pilot-steps", type=int, default=0, help="run this many micro-steps, report, exit")
    ap.add_argument("--max-items", type=int, default=0, help="use only the first N items per epoch (fallback)")
    ap.add_argument("--group", type=int, default=4)
    ap.add_argument("--lr-enc", type=float, default=2.5e-5)
    ap.add_argument("--lr-head", type=float, default=1e-4)
    ap.add_argument("--misuse-weight", type=float, default=2.0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default=str(ART / "fraudshield-laya"))
    ap.add_argument("--log", default=str(ART / "train.log"))
    ap.add_argument("--device", default="cuda", help="cuda, cuda:0, ... or cpu (smoke test only)")
    ap.add_argument("--resume", action="store_true", help="continue from artifacts/laya/resume_state.pt")
    ap.add_argument("--stop-after-epochs", type=int, default=0,
                    help="end this session after N epochs (simulates a timeout; tests --resume)")
    a = ap.parse_args()

    random.seed(a.seed)
    torch.manual_seed(a.seed)
    dev = torch.device(a.device)
    is_cuda = dev.type == "cuda"
    prec = select_precision(dev.type, is_cuda and torch.cuda.is_bf16_supported())
    amp_dtype = {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}[prec["dtype"]]
    if is_cuda:
        torch.cuda.set_device(dev)
    ART.mkdir(parents=True, exist_ok=True)
    fh = open(a.log, "a", encoding="utf-8")
    log(f"=== run start {'PILOT' if a.pilot_steps else 'TRAIN'} args={vars(a)} precision={prec}", fh)
    prep = json.loads((ART / "prepare.json").read_text())
    keys = ("bookings", "unique_bookings", "fraud_rows", "items", "questions", "cost_scale_C", "states_sha256")
    log("prepare manifest: " + json.dumps({k: prep[k] for k in keys}), fh)

    from laya.common import build_model  # noqa: PLC0415
    model_dir = stock_model_dir()
    cfg = json.loads(open(os.path.join(model_dir, "rl_agent_config.json")).read())
    cfg["max_len"], cfg["head_max_len"] = 512, 192
    tok = load_tokenizer(model_dir)
    model = build_model(cfg, encoder_dir=os.path.join(model_dir, "encoder"), pretrained=False)
    model.load_state_dict(load_file(os.path.join(model_dir, "model.safetensors")), strict=True)
    finfo = freeze_lower(model, a.k_top)
    model.float()
    # frozen part in the autocast dtype to save memory; trainable part stays fp32 master weights
    model.encoder.embeddings.to(amp_dtype)
    for layer in model.encoder.layers[: finfo["n_layers"] - finfo["k_top"]]:
        layer.to(amp_dtype)
    model.encoder.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    model.head_checkpointing = True
    model.to(dev).train()
    log(f"freeze: {finfo}", fh)

    items = torch.load(ART / "train_items.pt", weights_only=False)
    if a.max_items:
        items = items[: a.max_items]
    cost_scale = float(prep["cost_scale_C"])
    qid_w = {"misuse": a.misuse_weight}

    enc_params = [p for n, p in model.named_parameters() if p.requires_grad and n.startswith("encoder.")]
    head_params = [p for n, p in model.named_parameters() if p.requires_grad and not n.startswith("encoder.")]
    opt = torch.optim.AdamW([{"params": enc_params, "lr": a.lr_enc}, {"params": head_params, "lr": a.lr_head}],
                            weight_decay=0.01)
    steps_per_epoch = math.ceil(len(items) / a.micro)
    total_micro = a.pilot_steps if a.pilot_steps else steps_per_epoch * a.epochs
    total_updates = max(1, math.ceil(total_micro / a.accum))
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=total_updates, eta_min=1e-6)
    log(f"items/epoch={len(items)} micro={a.micro} accum={a.accum} eff_batch={a.micro * a.accum} "
        f"micro_steps/epoch={steps_per_epoch} total_updates={total_updates} C={cost_scale:.2f}", fh)

    scaler = torch.amp.GradScaler("cuda", enabled=prec["grad_scaler"])
    mem = (lambda: torch.cuda.max_memory_allocated(dev) / 1e9) if is_cuda else (lambda: 0.0)
    memr = (lambda: torch.cuda.max_memory_reserved(dev) / 1e9) if is_cuda else (lambda: 0.0)
    if is_cuda:
        torch.cuda.reset_peak_memory_stats(dev)
    t0 = time.time()
    gstep = 0
    n_items_done = 0
    epochs = 1 if a.pilot_steps else a.epochs
    extra: dict = {}
    start_ep = 0
    prev_min = 0.0
    resume_path = ART / "resume_state.pt"
    if a.resume and resume_path.exists() and not a.pilot_steps:
        st = torch.load(resume_path, map_location="cpu", weights_only=False)
        if (st["k_top"], st["micro"], st["accum"], st["epochs"]) != (a.k_top, a.micro, a.accum, a.epochs):
            raise SystemExit(f"resume state was saved with a different config: {st['k_top'], st['micro'], st['accum'], st['epochs']}")
        model.load_state_dict(st["model"])
        opt.load_state_dict(st["opt"])
        sched.load_state_dict(st["sched"])
        scaler.load_state_dict(st["scaler"])
        start_ep, gstep, extra = st["epochs_done"], st["gstep"], st["extra"]
        prev_min = float(extra.get("train_minutes", 0.0))
        log(f"RESUMED from {resume_path}: {start_ep} epoch(s) done, step {gstep}", fh)
    for ep in range(start_ep, epochs):
        rng = random.Random(42 + ep)
        order = list(range(len(items)))
        rng.shuffle(order)
        opt.zero_grad(set_to_none=True)
        acc = {"loss": 0.0, "ce": 0.0, "rl": 0.0, "rp": 0.0, "rc": 0.0, "n": 0}
        for bi in range(0, len(order), a.micro):
            chunk = [items[i] for i in order[bi: bi + a.micro]]
            b = collate(chunk, tok.pad_token_id, qid_w)
            sigma = sigma_at(gstep / max(1, total_micro - 1))
            with torch.autocast(dev.type, dtype=amp_dtype, enabled=prec["dtype"] != "fp32"):
                logits, act = model(b["input_ids"].to(dev), b["attention_mask"].to(dev), b["marker_pos"].to(dev),
                                    b["marker_mask"].to(dev), b["qtype"].to(dev))
            loss, info = compute_loss(logits, b, sigma, a.group, cost_scale)
            loss = loss / a.accum + 0.0 * act.sum()
            scaler.scale(loss).backward()
            gstep += 1
            n_items_done += len(chunk)
            if gstep % a.accum == 0 or bi + a.micro >= len(order):
                scaler.unscale_(opt)
                torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], 1.0)
                scaler.step(opt)
                scaler.update()
                sched.step()
                opt.zero_grad(set_to_none=True)
            acc["loss"] += loss.item() * a.accum
            acc["ce"] += info["ce"]
            acc["rl"] += info["rl"]
            acc["rp"] += 0.0 if math.isnan(info["reward_prob"]) else info["reward_prob"]
            acc["rc"] += 0.0 if math.isnan(info["reward_cost"]) else info["reward_cost"]
            acc["n"] += 1
            if gstep % 50 == 0:
                el = time.time() - t0
                n = acc["n"]
                log(f"ep {ep + 1}/{epochs} step {gstep}/{total_micro} loss {acc['loss'] / n:.4f} "
                    f"ce {acc['ce'] / n:.4f} rl {acc['rl'] / n:.4f} r_prob {acc['rp'] / n:.3f} "
                    f"r_cost {acc['rc'] / n:.3f} sigma {sigma:.3f} lr {sched.get_last_lr()[0]:.2e} "
                    f"items/s {n_items_done / el:.2f} "
                    f"peak_vram_gb {mem():.2f} reserved_gb {memr():.2f} elapsed_min {el / 60:.1f}", fh)
                acc = {"loss": 0.0, "ce": 0.0, "rl": 0.0, "rp": 0.0, "rc": 0.0, "n": 0}
            if a.pilot_steps and gstep >= a.pilot_steps:
                break
        if a.pilot_steps:
            el = time.time() - t0
            res = {"pilot_micro_steps": gstep, "items": n_items_done, "seconds": round(el, 1),
                   "items_per_s": round(n_items_done / el, 2), "k_top": a.k_top, "micro": a.micro,
                   "peak_vram_alloc_gb": round(mem(), 2), "peak_vram_reserved_gb": round(memr(), 2),
                   "device": a.device, "precision": prec["dtype"],
                   "gpu": torch.cuda.get_device_name(dev) if is_cuda else "cpu",
                   "trainable_params": finfo["trainable"],
                   "est_hours_per_epoch_full": round(prep["items"] / (n_items_done / el) / 3600, 2)}
            log("PILOT RESULT " + json.dumps(res), fh)
            pilots = ART / "pilot.json"
            old = json.loads(pilots.read_text()) if pilots.exists() else []
            pilots.write_text(json.dumps(old + [res], indent=2))
            return
        el = time.time() - t0
        log(f"=== epoch {ep + 1} done in {el / 60:.1f} min", fh)
        extra = {"epoch": ep + 1, "epochs": a.epochs, "k_top": a.k_top, "micro": a.micro, "accum": a.accum,
                 "items_per_epoch": len(items), "states_sha256": prep["states_sha256"], "seed": a.seed,
                 "train_minutes": round(prev_min + el / 60, 1), "questions": prep["questions"],
                 "precision": prec["dtype"], "gpu": torch.cuda.get_device_name(dev) if is_cuda else "cpu",
                 "base": "convaiinnovations/laya@" + os.path.basename(model_dir)}
        ck = ART / f"ckpt_epoch{ep + 1}"
        save_checkpoint(model, tok, cfg, model_dir, ck, extra)
        torch.save({"model": model.state_dict(), "opt": opt.state_dict(), "sched": sched.state_dict(),
                    "scaler": scaler.state_dict(), "epochs_done": ep + 1, "gstep": gstep, "extra": extra,
                    "k_top": a.k_top, "micro": a.micro, "accum": a.accum, "epochs": a.epochs},
                   str(resume_path) + ".tmp")
        os.replace(str(resume_path) + ".tmp", resume_path)
        log(f"saved {ck} and resume state", fh)
        if a.stop_after_epochs and ep + 1 - start_ep >= a.stop_after_epochs and ep + 1 < epochs:
            log(f"stopping after {a.stop_after_epochs} epoch(s) this session; rerun with --resume", fh)
            return
    save_checkpoint(model, tok, cfg, model_dir, Path(a.out), extra)
    log(f"=== TRAIN DONE total {(time.time() - t0) / 60:.1f} min this session, peak_vram_gb "
        f"{mem():.2f}; final checkpoint {a.out}", fh)


if __name__ == "__main__":
    main()
