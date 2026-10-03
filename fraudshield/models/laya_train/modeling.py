"""Model loading, partial freezing and checkpoint export for the Laya fine-tune."""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Any

STOCK_ID = "convaiinnovations/laya"


def stock_model_dir(model_id: str = STOCK_ID) -> str:
    from huggingface_hub import snapshot_download  # noqa: PLC0415
    from laya.agent import _fix_tokenizer_config  # noqa: PLC0415

    allow = ["rl_agent_config.json", "model.safetensors", "tokenizer/*", "encoder/*"]
    try:
        d = snapshot_download(model_id, allow_patterns=allow, local_files_only=True)
    except Exception:
        d = snapshot_download(model_id, allow_patterns=allow)
    _fix_tokenizer_config(d)
    return d


def load_tokenizer(model_dir: str):
    from transformers import AutoTokenizer  # noqa: PLC0415

    return AutoTokenizer.from_pretrained(os.path.join(model_dir, "tokenizer"))


def sha12(path: str | Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()[:12]


def freeze_lower(model, k_top: int) -> dict[str, Any]:
    """Freeze the encoder embeddings and all but the top k_top encoder layers. The encoder's final
    norm, the decision head, type embedding, scorer and act head stay trainable."""
    enc = model.encoder
    layers = enc.layers
    n = len(layers)
    k_top = max(0, min(int(k_top), n))
    for p in model.parameters():
        p.requires_grad = True
    for p in enc.embeddings.parameters():
        p.requires_grad = False
    for layer in layers[: n - k_top]:
        for p in layer.parameters():
            p.requires_grad = False
    tr = sum(p.numel() for p in model.parameters() if p.requires_grad)
    fr = sum(p.numel() for p in model.parameters() if not p.requires_grad)
    return {"k_top": k_top, "n_layers": n, "trainable": tr, "frozen": fr}


def export_config(cfg: dict, max_len: int = 512, head_max_len: int = 192, extra: dict | None = None) -> dict:
    """Checkpoint config that serves raw softmax(z): temperature 1.0 everywhere, no option buckets.
    Our own per-question calibration (artifacts/calibration.json) is applied outside Laya."""
    out = json.loads(json.dumps(cfg))
    out["temperature"] = [1.0, 1.0, 1.0]
    out.pop("temperature_by_options", None)
    out["max_len"] = int(max_len)
    out["head_max_len"] = int(head_max_len)
    out["fine_tuned"] = True
    out["model_name"] = "fraudshield-laya"
    out["fraudshield"] = extra or {}
    return out


def collate(items: list[dict], pad_id: int, qid_weights: dict[str, float] | None = None) -> dict:
    import torch  # noqa: PLC0415

    qid_weights = qid_weights or {}
    n, L = len(items), max(len(it["ids"]) for it in items)
    kmax = max(len(it["markers"]) for it in items)
    ids = torch.full((n, L), pad_id, dtype=torch.long)
    att = torch.zeros((n, L), dtype=torch.long)
    mpos = torch.zeros((n, kmax), dtype=torch.long)
    mmask = torch.zeros((n, kmax), dtype=torch.bool)
    target = torch.zeros((n, kmax), dtype=torch.float32)
    cost = torch.zeros((n, kmax), dtype=torch.float32)
    for i, it in enumerate(items):
        ids[i, : len(it["ids"])] = torch.tensor(it["ids"])
        att[i, : len(it["ids"])] = 1
        k = len(it["markers"])
        mpos[i, :k] = torch.tensor(it["markers"])
        mmask[i, :k] = True
        target[i, : len(it["target"])] = torch.tensor(it["target"], dtype=torch.float32)
        if "cost" in it:
            cost[i, : len(it["cost"])] = torch.tensor(it["cost"], dtype=torch.float32)
    return {"input_ids": ids, "attention_mask": att, "marker_pos": mpos, "marker_mask": mmask,
            "target": target, "cost": cost,
            "qtype": torch.tensor([it["qtype"] for it in items]),
            "label": torch.tensor([it.get("label", -1) for it in items]),
            "is_action": torch.tensor(["cost" in it for it in items]),
            "weight": torch.tensor([float(qid_weights.get(it.get("qid"), 1.0)) for it in items])}
