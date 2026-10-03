"""Helpers for running the Laya fine-tune on Kaggle (2x T4): precision choice, run config from the
pilot, and locating the uploaded bundle under /kaggle/input. Pure Python (no torch import)."""
from __future__ import annotations

import math
from pathlib import Path

MEM_FRACTION = 0.85     # safe rule: pilot peak reserved VRAM must stay under 85% of the card
EFFECTIVE_BATCH = 32


def native_bf16(capability: tuple[int, int]) -> bool:
    """bf16 runs natively from compute capability 8.0 (Ampere, Ada). torch.cuda.is_bf16_supported() also
    says True on a T4 (7.5) through emulation, which made Kaggle training about 5x slower than fp16."""
    return tuple(capability) >= (8, 0)


def select_precision(device_type: str, bf16_supported: bool) -> dict:
    """bf16 autocast where the GPU supports it (Ampere+), else fp16 autocast + GradScaler (T4).
    CPU runs (smoke tests only) stay in fp32."""
    if device_type != "cuda":
        return {"dtype": "fp32", "grad_scaler": False}
    if bf16_supported:
        return {"dtype": "bf16", "grad_scaler": False}
    return {"dtype": "fp16", "grad_scaler": True}


def choose_run_config(pilots: list[dict], gpu_total_gb: float, items_per_epoch: int,
                      budget_hours: float, epochs: int = 2) -> dict:
    """Pick the first pilot candidate (in preference order) that fits the memory rule and finishes
    `epochs` within budget_hours; otherwise the fastest memory-safe candidate with 1 epoch."""
    ok = [p for p in pilots if not p.get("oom") and p.get("items_per_s")
          and p["peak_vram_reserved_gb"] <= MEM_FRACTION * gpu_total_gb]
    if not ok:
        raise RuntimeError(f"no pilot configuration fits {MEM_FRACTION:.0%} of {gpu_total_gb} GB: {pilots}")

    def cfg(p, ep, note):
        return {"k_top": p["k_top"], "micro": p["micro"], "accum": max(1, EFFECTIVE_BATCH // p["micro"]),
                "epochs": ep, "est_hours": items_per_epoch * ep / p["items_per_s"] / 3600, "note": note}

    for p in ok:
        if items_per_epoch * epochs / p["items_per_s"] / 3600 <= budget_hours:
            return cfg(p, epochs, "planned")
    fastest = max(ok, key=lambda p: p["items_per_s"])
    return cfg(fastest, 1, f"fallback: {epochs} epochs would exceed {budget_hours} h, running 1 epoch")


def find_bundle(input_root: str | Path = "/kaggle/input", zip_name: str = "fraudshield_laya_bundle.zip"):
    """Kaggle usually auto-extracts an uploaded zip; accept either the extracted folder (the directory
    holding scripts/laya_train.py) or the zip itself. Returns ("dir", path) or ("zip", path)."""
    root = Path(input_root)
    hits = sorted(root.rglob("scripts/laya_train.py"))
    if hits:
        return "dir", hits[0].parent.parent
    zips = sorted(root.rglob(zip_name)) or sorted(root.rglob("*.zip"))
    if zips:
        return "zip", zips[0]
    raise FileNotFoundError(f"no FraudShield bundle (folder with scripts/laya_train.py or {zip_name}) under {root}")
