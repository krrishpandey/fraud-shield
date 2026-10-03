"""Laya fine-tuning, calibration inputs and evaluation helpers (DESIGN section 7 and 10).

torch / transformers / laya are imported lazily inside the modules that need them, so the
data and metric helpers stay importable without the `ml` extra.
"""
