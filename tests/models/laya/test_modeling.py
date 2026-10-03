import hashlib

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("transformers")

from fraudshield.models.laya_train import modeling as M  # noqa: E402


def _tiny_model():
    from transformers import ModernBertConfig, ModernBertModel
    from laya.common import DecisionModel
    cfg = ModernBertConfig(vocab_size=100, hidden_size=64, intermediate_size=96, num_hidden_layers=6,
                           num_attention_heads=4, max_position_embeddings=128, pad_token_id=0,
                           global_attn_every_n_layers=3)
    return DecisionModel(ModernBertModel(cfg), head_layers=1, n_act=2)


def test_freeze_lower_keeps_top_k_layers_and_head_trainable():
    m = _tiny_model()
    info = M.freeze_lower(m, k_top=2)
    layers = m.encoder.layers
    assert all(not p.requires_grad for l in layers[:4] for p in l.parameters())
    assert all(p.requires_grad for l in layers[4:] for p in l.parameters())
    assert all(not p.requires_grad for p in m.encoder.embeddings.parameters())
    assert all(p.requires_grad for p in m.scorer.parameters())
    assert all(p.requires_grad for p in m.head.parameters())
    assert info["trainable"] + info["frozen"] == sum(p.numel() for p in m.parameters())
    assert info["k_top"] == 2 and info["n_layers"] == 6


def test_freeze_lower_k_all_layers_trains_everything_but_embeddings():
    m = _tiny_model()
    M.freeze_lower(m, k_top=6)
    assert all(p.requires_grad for p in m.encoder.layers.parameters())


def test_export_config_serves_raw_probabilities():
    cfg = {"temperature": [1.6, 1.2, 1.9], "temperature_by_options": {"choice:2": 1.9}, "max_len": 1024,
           "head_max_len": 256, "model_name": "rl-agent"}
    out = M.export_config(cfg, max_len=512, head_max_len=192, extra={"run": "x"})
    assert out["temperature"] == [1.0, 1.0, 1.0]
    assert "temperature_by_options" not in out
    assert out["max_len"] == 512 and out["head_max_len"] == 192
    assert out["fine_tuned"] is True and out["fraudshield"] == {"run": "x"}
    assert cfg["temperature"] == [1.6, 1.2, 1.9]          # input not mutated


def test_collate_marks_action_items_costs_and_weights():
    items = [
        {"ids": [1, 2, 3], "markers": [1, 2], "qtype": 0, "target": [1.0, 0.0], "label": 0, "qid": "misuse"},
        {"ids": [1, 2, 3, 4, 5, 6, 7], "markers": [1, 2, 3, 4, 5, 6], "qtype": 0,
         "target": [0, 0, 0, 0, 0, 1.0], "label": 5, "qid": "action", "cost": [6, 5, 4, 3, 2, 0]},
    ]
    b = M.collate(items, pad_id=0, qid_weights={"misuse": 2.0})
    assert b["input_ids"].shape == (2, 7)
    assert b["attention_mask"][0].tolist() == [1, 1, 1, 0, 0, 0, 0]
    assert b["is_action"].tolist() == [False, True]
    assert b["cost"][1].tolist() == [6, 5, 4, 3, 2, 0]
    assert b["cost"][0].tolist() == [0] * 6
    assert b["weight"].tolist() == [2.0, 1.0]
    assert b["marker_mask"][0].tolist() == [True, True, False, False, False, False]


def test_sha12(tmp_path):
    p = tmp_path / "w.bin"
    p.write_bytes(b"abc")
    assert M.sha12(p) == hashlib.sha256(b"abc").hexdigest()[:12]
