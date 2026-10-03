"""Real components for the decision service, as config import strings:

    components:
      featurizer: fraudshield.api.real_components:featurizer
      serializer: fraudshield.api.real_components:serializer
      gbm: fraudshield.api.real_components:gbm

The demo feature store and the GBM load lazily once per process. Each scored booking is appended to
the store, so later bookings on the same account see it in their history (as in production).
"""
from __future__ import annotations

import threading
from pathlib import Path

from fraudshield.contracts import Booking, FeatureVector

ROOT = Path(__file__).resolve().parents[2]
STORE_PATH = ROOT / "data/processed/feature_store_demo.parquet"
MODELS_DIR = ROOT / "artifacts/models"
SYSTEM, TIER = "B2", "F"

_lock = threading.RLock()
_store = None
_seen: set[str] = set()
_gbm_model = None


def reset() -> None:
    global _store, _gbm_model
    with _lock:
        _store, _gbm_model = None, None
        _seen.clear()


def _get_store():
    global _store
    with _lock:
        if _store is None:
            from fraudshield.data.demo import load_demo_store

            _store = load_demo_store(STORE_PATH)
            _seen.update(_store.frame["booking_id"].astype(str))
        return _store


def _get_gbm():
    global _gbm_model
    with _lock:
        if _gbm_model is None:
            from fraudshield.models.gbm import GBMModel

            # Base model; after a learning deploy, learning/wiring.py swaps in the active version.
            _gbm_model = GBMModel.load(MODELS_DIR, SYSTEM, TIER)
            _gbm_model.metadata["active_version"] = f"gbm-{SYSTEM}-{TIER}-v1"
        return _gbm_model


def featurizer(booking: Booking) -> FeatureVector:
    from fraudshield.features.featurize import featurize

    store = _get_store()
    with _lock:
        fv = featurize(booking, store)
        if booking.booking_id not in _seen:
            store.append(booking)
            _seen.add(booking.booking_id)
    return fv


def serializer(booking: Booking, fv: FeatureVector) -> str:
    from fraudshield.features.serialize import serialize

    return serialize(fv)  # no GBM line: Laya was fine-tuned on states without it


def gbm(fv: FeatureVector) -> float:
    return _get_gbm().score_values(fv.values)


def account_story(booking: Booking) -> dict:
    """The account's own bookings as of this booking (read only; the booking is not appended)."""
    from fraudshield.features.story import account_story as build

    store = _get_store()
    with _lock:
        return build(store.history(booking.account_id, booking.booked_at), booking)


def gbm_version() -> str:
    return _get_gbm().metadata.get("active_version", "gbm")


def store_size() -> int:
    return len(_get_store())


def warm() -> None:
    """Load the store and model before the first booking, so the first score is not slow."""
    _get_store()
    _get_gbm()
