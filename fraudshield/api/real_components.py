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


def serializer_v2(booking: Booking, fv: FeatureVector, gbm_risk: float | None = None) -> str:
    """Laya v2 state (docs/LAYA_V2.md): v1 lines + MORE + EVIDENCE (LightGBM risk as a witness, the two rules)."""
    from fraudshield.features.serialize import evidence_for, serialize_v2

    return serialize_v2(fv, evidence_for(fv.values, gbm_risk))


serializer_v2.wants_gbm = True  # the pipeline passes the LightGBM score in


def gbm(fv: FeatureVector) -> float:
    return _get_gbm().score_values(fv.values)


def account_story(booking: Booking) -> dict:
    """The account's own bookings as of this booking (read only; the booking is not appended)."""
    from fraudshield.features.story import account_story as build

    store = _get_store()
    with _lock:
        return build(store.history(booking.account_id, booking.booked_at), booking)


PROC = ROOT / "data/processed"
STREAM_WINDOW = ("test",)  # 2018-05-15 .. 2018-08-31
_stream_cache: dict[int, list] = {}


def stream_source(seed: int = 0) -> list:
    """The live stream's bookings: the seed-0 test window (real Olist bookings plus the injected rows), in
    booked_at order, with original booking ids so every booking's history is exactly as of its own time.

    Ground truth and the offline score of the same model ride beside each booking (StreamItem), never in it.
    Only seed 0 can be streamed: the demo feature store holds seed 0's injected rows.
    """
    if seed != 0:
        raise ValueError("only injection seed 0 can be streamed: the demo feature store holds seed 0's fraud rows")
    with _lock:
        if 0 in _stream_cache:
            return _stream_cache[0]
    import json as _json

    import pandas as pd

    from fraudshield.features.store import FeatureStore
    from fraudshield.sim.stream import StreamItem

    from fraudshield.api.pipeline import DROP_PRIOR
    from fraudshield.features.mix import rule_flags_frame

    truth = pd.read_parquet(PROC / "features" / "seed_0.parquet",
                            columns=["booking_id", "split", "booked_at", "is_fraud", "typology",
                                     "consignee_first_seen_days", "consignee_bookings_30d", "consignee_other_accts_30d",
                                     "dims_z", "weight_z", "n_prior", "hn_new_state", "hn_new_seller",
                                     "hn_billing_change", "hn_injected", "hn_multi_account_consignee"])
    hn_cols = ["hn_new_state", "hn_new_seller", "hn_billing_change", "hn_injected", "hn_multi_account_consignee"]
    hn_of = dict(zip(truth.booking_id, truth[hn_cols].fillna(False).astype(bool).any(axis=1)))
    truth = truth.assign(_drop=rule_flags_frame(truth).drop_pattern)
    truth = truth[truth.split.isin(STREAM_WINDOW)].sort_values(["booked_at", "booking_id"], kind="mergesort")
    offline = pd.read_parquet(PROC / "gbm_scores_seed0.parquet", columns=["booking_id", "gbm_b2f"])
    off = dict(zip(offline.booking_id, offline.gbm_b2f))
    frame = _get_store().frame.set_index("booking_id", drop=False)
    demo = {x["booking"]["booking_id"]: x["booking"]
            for x in _json.loads((PROC / "demo_bookings.json").read_text(encoding="utf-8"))}
    items = []
    drop_of = dict(zip(truth.booking_id, truth._drop))
    inj = pd.read_parquet(PROC / "features" / "injected_seed_0.parquet", columns=["booking_id", "true_weight_kg"])
    true_w = dict(zip(inj.booking_id, inj.true_weight_kg))
    for bid, fraud, typ in zip(truth.booking_id, truth.is_fraud, truth.typology):
        if bid in frame.index:
            b = FeatureStore.booking_from_row(frame.loc[bid])
        else:  # the five demo bookings are kept out of the store so the demo can score them live
            b = Booking(**{**demo[bid], "meta": None})
        b = Booking(**{**b.__dict__, "meta": {"scenario": "stream", "source": "replay of seed 0 test window"}})
        items.append(StreamItem(b, is_fraud=bool(fraud), typology=str(typ) if fraud else "none",
                                offline_gbm=float(off[bid]) if bid in off else None,
                                hard_negative=bool(hn_of.get(bid, False)) and not bool(fraud),
                                true_weight_kg=float(true_w[bid]) if bid in true_w and true_w[bid] == true_w[bid] else None,
                                offline_score=(1 - (1 - float(off[bid])) * (1 - DROP_PRIOR * drop_of[bid]))
                                if bid in off else None))
    with _lock:
        _stream_cache[0] = items
    return items


def gbm_version() -> str:
    return _get_gbm().metadata.get("active_version", "gbm")


def store_size() -> int:
    return len(_get_store())


def warm() -> None:
    """Load the store and model before the first booking, so the first score is not slow."""
    _get_store()
    _get_gbm()
