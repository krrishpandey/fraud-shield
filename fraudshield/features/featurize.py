"""Feature computation, strictly from history before booked_at.

Two implementations of the same definitions:
  featurize(booking, store)  - reference, pandas over store.history(); used online by the API.
  featurize_frame(df, ...)   - single time-ordered pass with incremental state; used offline.
tests/features verifies they agree on a sample. Neither reads Booking.meta or label columns.
"""
from __future__ import annotations

import math
from bisect import bisect_left
from collections import Counter, defaultdict

import numpy as np
import pandas as pd

from fraudshield.contracts import Booking, FeatureVector
from fraudshield.data.olist import CATEGORY_VOCAB, HIGH_VALUE_CATEGORIES
from fraudshield.features.mix import FAR_KM, account_mix, rule_flags, under_score
from fraudshield.features.geo import dist_km
from fraudshield.features.spec import CATEGORY_INDEX, CHANNELS, FEATURES, PAYMENTS
from fraudshield.features.store import FeatureStore, booking_to_row, to_ts

PROFILE_CAP = 200   # robust profile uses the account's last 200 bookings
MIN_PROFILE = 5     # fewer prior bookings -> profile stats are NaN ("n/a")
DAY_NS = 86400 * 10**9
NAN = float("nan")


# shared math ---------------------------------------------------------------------------------
def robust_z(x: float, arr: np.ndarray, log: bool = False) -> float:
    if log:
        x, arr = math.log1p(max(x, 0.0)), np.log1p(np.maximum(arr, 0.0))
    med = float(np.median(arr))
    mad = float(np.median(np.abs(arr - med)))
    scale = max(1.4826 * mad, 0.1 if log else max(0.1 * abs(med), 0.05))
    return float(np.clip((x - med) / scale, -20, 20))


def entropy(counts) -> float:
    c = np.asarray([v for v in counts if v > 0], dtype=float)
    if c.size <= 1:
        return 0.0
    p = c / c.sum()
    return float(-(p * np.log2(p)).sum())


def hour_pct(h: int, hours: np.ndarray) -> float:
    d = np.abs(hours - h)
    d = np.minimum(d, 24 - d)
    return float(100.0 * np.mean(d <= 1))


def _cat(c: str) -> str:
    return c if c in CATEGORY_INDEX else "other"


def raw_features(row: dict) -> dict:
    t: pd.Timestamp = row["booked_at"]
    w = float(row["weight_kg"])
    L, W, H = float(row["length_cm"]), float(row["width_cm"]), float(row["height_cm"])
    cost = float(row["carrier_cost"])
    cat = _cat(row["category"])
    oc = row["owner_contact_age_days"]
    oc = NAN if oc is None or (isinstance(oc, float) and math.isnan(oc)) else float(oc)
    return {
        "origin_uf": row["origin_uf"], "origin_zip3": row["origin_zip3"], "dest_uf": row["dest_uf"],
        "dest_zip3": row["dest_zip3"], "category": cat, "service": row["service"],
        "channel": row["channel"], "payment_method": row["payment_method"],
        "hour": t.hour, "dow": t.dayofweek, "night": int(t.hour < 6),
        "dist_km": dist_km(row["origin_zip3"], row["origin_uf"], row["dest_zip3"], row["dest_uf"]),
        "weight_kg": w, "volume_l": L * W * H / 1000.0, "max_dim_cm": max(L, W, H),
        "length_cm": L, "width_cm": W, "height_cm": H,
        "declared_value": float(row["declared_value"]), "carrier_cost": cost,
        "cost_per_kg": cost / max(w, 0.1), "express": int(row["service"] == "express"),
        "category_code": CATEGORY_INDEX[cat], "high_value": int(cat in HIGH_VALUE_CATEGORIES),
        "channel_code": CHANNELS.index(row["channel"]) if row["channel"] in CHANNELS else -1,
        "payment_code": PAYMENTS.index(row["payment_method"]) if row["payment_method"] in PAYMENTS else -1,
        "login_device_age_days": float(row["login_device_age_days"]),
        "owner_contact_age_days": oc, "owner_contact_ok": int(not math.isnan(oc) and oc >= 30),
    }


def _profile(v: dict, row: dict, w, vol, val, cost, hours, exp) -> None:
    if len(w) >= MIN_PROFILE:
        v["weight_z"] = robust_z(v["weight_kg"], w)
        v["dims_z"] = robust_z(v["volume_l"], vol)
        v["value_z"] = robust_z(v["declared_value"], val, log=True)
        v["payoff_z"] = robust_z(v["carrier_cost"], cost, log=True)
        v["cost_vs_median"] = float(v["carrier_cost"] / max(float(np.median(cost)), 0.01))
        v["hour_pct"] = hour_pct(v["hour"], hours)
        v["svc_exp_pct"] = float(100.0 * np.mean(exp))
    else:
        for k in ("weight_z", "dims_z", "value_z", "payoff_z", "cost_vs_median", "hour_pct", "svc_exp_pct"):
            v[k] = NAN


def _events(v: dict, acc: str, t: pd.Timestamp, entities: list[str], changes, confirmed) -> None:
    v["verified_change_30d"] = int(any(c["account_id"] == acc and c["announced_at"] < t < c["at"] + pd.Timedelta(days=30)
                                       for c in changes))
    v["prior_confirmed_fraud"] = sum(1 for c in confirmed if c["account_id"] == acc and c["confirmed_at"] < t)
    seen = set()
    for c in confirmed:
        if c["confirmed_at"] < t:
            seen.update(c["entities"])
    v["links_confirmed_fraud"] = sum(1 for e in entities if e in seen)


def _entities(row: dict) -> list[str]:
    ent = [row["consignee_id"]]
    if row["sender_id"] != row["account_id"]:
        ent.append(row["sender_id"])
    return ent


# reference implementation ------------------------------------------------------------------
def featurize(booking: Booking, store: FeatureStore) -> FeatureVector:
    row = booking_to_row(booking)
    t = row["booked_at"]
    H = store.history(row["account_id"], t)
    C = store.consignee_history(row["consignee_id"], t)
    v = raw_features(row)
    _reference_history(v, row, H, C)
    _events(v, row["account_id"], t, _entities(row), store.changes, store.confirmed)
    if len(H):
        hv = H.category.map(lambda c: int(_cat(c) in HIGH_VALUE_CATEGORIES)).to_numpy(float)
        dist = np.array([dist_km(r.origin_zip3, r.origin_uf, r.dest_zip3, r.dest_uf) for r in H.itertuples()])
        v.update(account_mix(len(H), hv.sum(), float((dist >= FAR_KM).sum()), dist.sum()))
    else:
        v.update(account_mix(0, 0.0, 0.0, 0.0))
    v.update(rule_flags(v))
    v["under_score"] = under_score(v)
    return FeatureVector(booking_id=booking.booking_id, as_of=t.isoformat(), values=v)


def _days(td) -> float:
    return td.total_seconds() / 86400.0


def _reference_history(v: dict, row: dict, H: pd.DataFrame, C: pd.DataFrame) -> None:
    t, acc = row["booked_at"], row["account_id"]
    o, duf, snd = row["origin_zip3"], row["dest_uf"], row["sender_id"]
    n = len(H)
    ts = H.booked_at
    v["n_prior"] = n
    v["tenure_days"] = _days(t - ts.iloc[0]) if n else 0.0
    win = {d: H[ts >= t - pd.Timedelta(days=d)] for d in (90, 30, 7, 3, 1)}
    v["bookings_90d"], v["bookings_72h"], v["bookings_24h"] = len(win[90]), len(win[3]), len(win[1])
    v["rate_90d"] = len(win[90]) / min(90.0, max(v["tenure_days"], 1.0))
    v["burst_ratio"] = len(win[1]) / max(v["rate_90d"], 0.1)
    v["express_72h"] = int((win[3].service == "express").sum())
    v["mean_kg_72h"] = float(win[3].weight_kg.mean()) if len(win[3]) else NAN
    if n:
        pc = H.groupby(["origin_zip3", "origin_uf"]).size()
        best = sorted(k for k, c in pc.items() if c == pc.max())[0]
        v["home_origin_zip3"], v["home_origin_uf"] = best
        v["is_home_origin"] = int(o == best[0])
    else:
        v["home_origin_zip3"], v["home_origin_uf"], v["is_home_origin"] = row["origin_zip3"], row["origin_uf"], 1
    same_o = H[H.origin_zip3 == o]
    v["origin_seen"] = len(same_o)
    v["payer_origin_pair_age_days"] = _days(t - same_o.booked_at.min()) if len(same_o) else 0.0
    v["lane_seen"] = int(((H.origin_zip3 == o) & (H.dest_uf == duf)).sum())
    v["dest_region_seen"] = int((H.dest_uf == duf).sum())
    v["cat_seen"] = int((H.category.map(_cat) == v["category"]).sum())
    P = H.tail(PROFILE_CAP)
    _profile(v, row, P.weight_kg.to_numpy(float), (P.length_cm * P.width_cm * P.height_cm / 1000.0).to_numpy(float),
             P.declared_value.to_numpy(float), P.carrier_cost.to_numpy(float),
             P.booked_at.dt.hour.to_numpy(), (P.service == "express").to_numpy())
    v["sender_differs"] = int(snd != acc)
    same_s = H[H.sender_id == snd]
    v["sender_seen"] = len(same_s)
    v["payer_sender_pair_age_days"] = _days(t - same_s.booked_at.min()) if len(same_s) else 0.0
    # change block
    pos90 = n - len(win[90])
    for key, col in (("consignees", "consignee_id"), ("origins", "origin_zip3"), ("senders", "sender_id")):
        flag = (ts == H.groupby(col).booked_at.transform("min")).to_numpy() if n else np.zeros(0, bool)
        v[f"new_{key}_l10"] = int(flag[max(0, n - 10):].sum())
        base = flag[pos90:max(pos90, n - 10)]
        v[f"base_{key}_l10"] = float(base.mean() * 10) if len(base) else NAN
    old = H[(ts >= t - pd.Timedelta(days=90)) & (ts < t - pd.Timedelta(days=7))]
    for key, col, cur in (("origins", "origin_zip3", o), ("senders", "sender_id", snd)):
        v[f"distinct_{key}_7d"] = len(set(win[7][col]) | {cur})
        v[f"base_{key}_7d"] = int(old[col].nunique())
        e7 = Counter(win[7][col]); e7[cur] += 1
        e90 = Counter(win[90][col]); e90[cur] += 1
        v[("origin" if key == "origins" else "sender") + "_entropy_jump"] = entropy(e7.values()) - entropy(e90.values())
    v["payer_origins_30d"] = len(set(win[30].origin_zip3) | {o})
    v["payer_senders_30d"] = len(set(win[30].sender_id) | {snd})
    # consignee graph
    v["consignee_prior"] = len(C)
    v["consignee_first_seen_days"] = _days(t - C.booked_at.min()) if len(C) else 0.0
    C30 = C[C.booked_at >= t - pd.Timedelta(days=30)]
    v["consignee_bookings_30d"] = len(C30)
    v["consignee_other_accts_30d"] = len(set(C30.account_id) - {acc})
    v["consignee_accts_30d"] = len(set(C30.account_id) | {acc})


# single-pass implementation ----------------------------------------------------------------
class _Acc:
    __slots__ = ("ts", "origin", "sender", "cons", "dest", "w", "vol", "val", "cost", "hour", "exp",
                 "f_cons", "f_origin", "f_sender", "pair_cnt", "o_cnt", "o_first", "s_cnt", "s_first",
                 "lane", "dest_cnt", "cat_cnt", "cons_seen", "hv_sum", "far_sum", "dist_sum")

    def __init__(self):
        for s in self.__slots__[:14]:
            setattr(self, s, [])
        self.pair_cnt, self.o_cnt, self.s_cnt = Counter(), Counter(), Counter()
        self.o_first, self.s_first = {}, {}
        self.lane, self.dest_cnt, self.cat_cnt = Counter(), Counter(), Counter()
        self.cons_seen = set()
        self.hv_sum = self.far_sum = self.dist_sum = 0.0


def _stream_one(v: dict, row: dict, tn: int, a: _Acc | None, cons: tuple[list, list] | None) -> dict:
    """Features from incremental state; returns the novelty flags this row will carry."""
    acc, o, duf, snd = row["account_id"], row["origin_zip3"], row["dest_uf"], row["sender_id"]
    n = len(a.ts) if a else 0
    ts = a.ts if a else []
    lo = {d: bisect_left(ts, tn - d * DAY_NS) for d in (90, 30, 7, 3, 1)}
    v["n_prior"] = n
    v["tenure_days"] = (tn - ts[0]) / DAY_NS if n else 0.0
    v["bookings_90d"], v["bookings_72h"], v["bookings_24h"] = n - lo[90], n - lo[3], n - lo[1]
    v["rate_90d"] = (n - lo[90]) / min(90.0, max(v["tenure_days"], 1.0))
    v["burst_ratio"] = (n - lo[1]) / max(v["rate_90d"], 0.1)
    v["express_72h"] = int(sum(a.exp[lo[3]:])) if n else 0
    v["mean_kg_72h"] = float(np.mean(a.w[lo[3]:])) if n - lo[3] > 0 else NAN
    if n:
        mx = max(a.pair_cnt.values())
        best = min(k for k, c in a.pair_cnt.items() if c == mx)
        v["home_origin_zip3"], v["home_origin_uf"] = best
        v["is_home_origin"] = int(o == best[0])
        v["origin_seen"] = a.o_cnt[o]
        v["payer_origin_pair_age_days"] = (tn - a.o_first[o]) / DAY_NS if o in a.o_first else 0.0
        v["lane_seen"] = a.lane[(o, duf)]
        v["dest_region_seen"] = a.dest_cnt[duf]
        v["cat_seen"] = a.cat_cnt[v["category"]]
        k = max(0, n - PROFILE_CAP)
        _profile(v, row, np.asarray(a.w[k:]), np.asarray(a.vol[k:]), np.asarray(a.val[k:]), np.asarray(a.cost[k:]),
                 np.asarray(a.hour[k:]), np.asarray(a.exp[k:]))
        v["sender_seen"] = a.s_cnt[snd]
        v["payer_sender_pair_age_days"] = (tn - a.s_first[snd]) / DAY_NS if snd in a.s_first else 0.0
    else:
        v["home_origin_zip3"], v["home_origin_uf"], v["is_home_origin"] = row["origin_zip3"], row["origin_uf"], 1
        for k in ("origin_seen", "lane_seen", "dest_region_seen", "cat_seen", "sender_seen"):
            v[k] = 0
        v["payer_origin_pair_age_days"] = v["payer_sender_pair_age_days"] = 0.0
        _profile(v, row, [], [], [], [], [], [])
    v["sender_differs"] = int(snd != acc)
    l10 = max(0, n - 10)
    for key, fl in (("consignees", a.f_cons if n else []), ("origins", a.f_origin if n else []),
                    ("senders", a.f_sender if n else [])):
        v[f"new_{key}_l10"] = int(sum(fl[l10:]))
        base = fl[lo[90]:max(lo[90], l10)]
        v[f"base_{key}_l10"] = float(np.mean(base) * 10) if len(base) else NAN
    lo7 = lo[7]
    for key, lst, cur, ename in (("origins", a.origin if n else [], o, "origin_entropy_jump"),
                                 ("senders", a.sender if n else [], snd, "sender_entropy_jump")):
        w7 = lst[lo7:]
        v[f"distinct_{key}_7d"] = len(set(w7) | {cur})
        v[f"base_{key}_7d"] = len(set(lst[lo[90]:lo7]))
        e7 = Counter(w7); e7[cur] += 1
        e90 = Counter(lst[lo[90]:]); e90[cur] += 1
        v[ename] = entropy(e7.values()) - entropy(e90.values())
    v["payer_origins_30d"] = len(set(a.origin[lo[30]:] if n else []) | {o})
    v["payer_senders_30d"] = len(set(a.sender[lo[30]:] if n else []) | {snd})
    if cons:
        cts, caccs = cons
        v["consignee_prior"] = len(cts)
        v["consignee_first_seen_days"] = (tn - cts[0]) / DAY_NS
        c30 = bisect_left(cts, tn - 30 * DAY_NS)
        v["consignee_bookings_30d"] = len(cts) - c30
        s30 = set(caccs[c30:])
        v["consignee_other_accts_30d"] = len(s30 - {acc})
        v["consignee_accts_30d"] = len(s30 | {acc})
    else:
        v["consignee_prior"] = 0
        v["consignee_first_seen_days"] = 0.0
        v["consignee_bookings_30d"] = v["consignee_other_accts_30d"] = 0
        v["consignee_accts_30d"] = 1
    return {
        "f_cons": int(row["consignee_id"] not in a.cons_seen) if n else 1,
        "f_origin": int(o not in a.o_first) if n else 1,
        "f_sender": int(snd not in a.s_first) if n else 1,
    }


def _push(a: _Acc, row: dict, v: dict, tn: int, flags: dict) -> None:
    o, snd = row["origin_zip3"], row["sender_id"]
    a.ts.append(tn); a.origin.append(o); a.sender.append(snd); a.cons.append(row["consignee_id"])
    a.dest.append(row["dest_uf"]); a.w.append(v["weight_kg"]); a.vol.append(v["volume_l"])
    a.val.append(v["declared_value"]); a.cost.append(v["carrier_cost"]); a.hour.append(v["hour"])
    a.exp.append(v["express"])
    a.f_cons.append(flags["f_cons"]); a.f_origin.append(flags["f_origin"]); a.f_sender.append(flags["f_sender"])
    a.pair_cnt[(o, row["origin_uf"])] += 1
    a.o_cnt[o] += 1; a.o_first.setdefault(o, tn)
    a.s_cnt[snd] += 1; a.s_first.setdefault(snd, tn)
    a.lane[(o, row["dest_uf"])] += 1; a.dest_cnt[row["dest_uf"]] += 1; a.cat_cnt[v["category"]] += 1
    a.cons_seen.add(row["consignee_id"])
    a.hv_sum += v["high_value"]; a.far_sum += float(v["dist_km"] >= FAR_KM); a.dist_sum += v["dist_km"]


def featurize_frame(df: pd.DataFrame, changes: pd.DataFrame | None = None,
                    confirmed: list[dict] | None = None, progress: bool = False) -> pd.DataFrame:
    """Features for every row of df (one time-ordered pass). Returns booking_id + FEATURES columns."""
    store_ev = FeatureStore.from_frame(df.iloc[0:0], changes=changes, confirmed=confirmed)
    ch_by_acc = defaultdict(list)
    for c in store_ev.changes:
        ch_by_acc[c["account_id"]].append(c)
    conf = sorted(store_ev.confirmed, key=lambda c: c["confirmed_at"])
    conf_cnt: Counter = Counter()
    conf_ent: set = set()
    ci = 0

    cols = ["booking_id", "booked_at"] + [c for c in df.columns if c in FeatureStore().frame.columns and c not in ("booking_id", "booked_at")]
    d = df[cols].copy()
    d["booked_at"] = pd.to_datetime(d.booked_at)
    d = d.sort_values(["booked_at", "booking_id"], kind="mergesort").reset_index(drop=True)
    records = d.to_dict("records")
    tns = d.booked_at.astype("int64").to_numpy()
    accs: dict[str, _Acc] = {}
    cons: dict[str, tuple[list, list]] = {}
    out = []
    i, N = 0, len(records)
    while i < N:
        j = i
        while j < N and tns[j] == tns[i]:
            j += 1
        t = records[i]["booked_at"]
        while ci < len(conf) and conf[ci]["confirmed_at"] < t:
            conf_cnt[conf[ci]["account_id"]] += 1
            conf_ent.update(conf[ci]["entities"])
            ci += 1
        group = []
        for k in range(i, j):
            row = records[k]
            oc = row["owner_contact_age_days"]
            row["owner_contact_age_days"] = None if oc is None or (isinstance(oc, float) and math.isnan(oc)) else oc
            v = raw_features(row)
            flags = _stream_one(v, row, tns[k], accs.get(row["account_id"]), cons.get(row["consignee_id"]))
            acc = row["account_id"]
            v["verified_change_30d"] = int(any(c["announced_at"] < t < c["at"] + pd.Timedelta(days=30)
                                               for c in ch_by_acc.get(acc, ())))
            v["prior_confirmed_fraud"] = conf_cnt[acc]
            v["links_confirmed_fraud"] = sum(1 for e in _entities(row) if e in conf_ent)
            prev = accs.get(acc)
            v.update(account_mix(len(prev.ts) if prev else 0, *((prev.hv_sum, prev.far_sum, prev.dist_sum) if prev else (0, 0, 0))))
            v["booking_id"] = row["booking_id"]
            out.append(v)
            group.append((row, v, tns[k], flags))
        for row, v, tn, flags in group:
            _push(accs.setdefault(row["account_id"], _Acc()), row, v, tn, flags)
            cts, caccs = cons.setdefault(row["consignee_id"], ([], []))
            cts.append(tn); caccs.append(row["account_id"])
        if progress and (j // 20000) != (i // 20000):
            print(f"  featurized {j:,}/{N:,}", flush=True)
        i = j
    res = pd.DataFrame(out)
    return res[["booking_id"] + list(FEATURES)]
