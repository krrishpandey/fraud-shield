import zipfile

import pytest

from fraudshield.models.laya_train import kaggle as K


def test_select_precision_bf16_when_supported():
    assert K.select_precision("cuda", bf16_supported=True) == {"dtype": "bf16", "grad_scaler": False}


def test_select_precision_fp16_with_scaler_on_t4():
    assert K.select_precision("cuda", bf16_supported=False) == {"dtype": "fp16", "grad_scaler": True}


def test_select_precision_cpu_is_fp32():
    assert K.select_precision("cpu", bf16_supported=True) == {"dtype": "fp32", "grad_scaler": False}


PILOTS = [
    {"k_top": 12, "micro": 16, "items_per_s": 20.0, "peak_vram_reserved_gb": 11.0},
    {"k_top": 8, "micro": 8, "items_per_s": 25.0, "peak_vram_reserved_gb": 6.0},
]


def test_choose_prefers_first_candidate_that_fits_memory_and_time():
    c = K.choose_run_config(PILOTS, gpu_total_gb=15.0, items_per_epoch=30000, budget_hours=6.0)
    assert (c["k_top"], c["micro"], c["accum"], c["epochs"]) == (12, 16, 2, 2)
    assert c["est_hours"] == pytest.approx(60000 / 20 / 3600)


def test_choose_skips_candidate_over_memory_rule_or_oom():
    p = [{"k_top": 12, "micro": 16, "items_per_s": 20.0, "peak_vram_reserved_gb": 13.5}] + PILOTS[1:]
    c = K.choose_run_config(p, gpu_total_gb=15.0, items_per_epoch=30000, budget_hours=6.0)
    assert c["k_top"] == 8 and c["accum"] == 4
    p = [{"k_top": 12, "micro": 16, "oom": True}] + PILOTS[1:]
    assert K.choose_run_config(p, 15.0, 30000, 6.0)["k_top"] == 8


def test_choose_falls_back_to_one_epoch_when_too_slow():
    slow = [{"k_top": 8, "micro": 8, "items_per_s": 2.0, "peak_vram_reserved_gb": 6.0}]
    c = K.choose_run_config(slow, 15.0, 30000, budget_hours=6.0)
    assert c["epochs"] == 1 and "fallback" in c["note"]


def test_choose_raises_when_nothing_fits():
    with pytest.raises(RuntimeError):
        K.choose_run_config([{"k_top": 8, "micro": 8, "oom": True}], 15.0, 30000, 6.0)


def test_find_bundle_extracted_dir(tmp_path):
    d = tmp_path / "input" / "fraudshield-laya-bundle" / "fraudshield_laya_bundle"
    (d / "scripts").mkdir(parents=True)
    (d / "scripts" / "laya_train.py").write_text("x")
    assert K.find_bundle(tmp_path / "input") == ("dir", d)


def test_find_bundle_zip(tmp_path):
    z = tmp_path / "input" / "ds" / "fraudshield_laya_bundle.zip"
    z.parent.mkdir(parents=True)
    with zipfile.ZipFile(z, "w") as f:
        f.writestr("scripts/laya_train.py", "x")
    assert K.find_bundle(tmp_path / "input") == ("zip", z)


def test_find_bundle_missing(tmp_path):
    (tmp_path / "input").mkdir()
    with pytest.raises(FileNotFoundError):
        K.find_bundle(tmp_path / "input")
