"""Rule-based injection of fraud typologies and injected-legit hard negatives into real Olist histories.

Typologies (DESIGN section 9; sources in DATA_CARD.md):
  T1 label-resale takeover (Harrod DOJ; Hao et al.)    T2 bust-out new account (Butt, Grizzle DOJ)
  T3 reshipping drops, HELD OUT (Hao et al. CCS 2015)  T4 payoff-maximizing takeover (LabelsBank/NullShip pricing)
  T5 test-then-burst, UNSOURCED hypothesis, HELD OUT   T6 declared weight/dims manipulation (Kavanaugh; billing audits)
  T7 bill-to-third-party misuse of a leaked account number (carrier guidance, I1)
  HN_new_channel, HN_3pl: injected LEGIT hard negatives.
Campaign sizes are truncated relative to the sources (real campaigns run to thousands of labels) to
keep booking prevalence near 1%; longer campaigns would be easier to catch, not harder.
Values are snapped to real grids: parcels are real Olist parcels (weight, dims, category, value),
zips are real zip prefixes, freight = linear lane model fitted on real rows + a real residual from the
same weight/distance bin, scan lags are real lags. Camouflage 0/1/2 and a mimic knob control difficulty.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from fraudshield.data.labels import risk_level
from fraudshield.data.olist import HIGH_VALUE_CATEGORIES, haversine_km
from fraudshield.data.split import SPLITS, assign_split

MASTER_SEED = 20261002
FRAUD_TYPOLOGIES = ("T1", "T2", "T3", "T4", "T5", "T6", "T7")
HN_TYPOLOGIES = ("HN_new_channel", "HN_3pl")
HELD_OUT = ("T3", "T5")
# campaigns per 1,000 real bookings in a split (about 30 fraud campaigns in the test window)
CAMPAIGN_RATES = {"T1": 0.27, "T4": 0.22, "T2": 0.18, "T3": 0.09, "T5": 0.18, "T6": 0.18, "T7": 0.22,
                  "HN_new_channel": 0.18, "HN_3pl": 0.13}
CAMO_P = (0.5, 0.3, 0.2)
MIMIC_P = 0.3
MIN_PRIOR = 20
POOL_SHARE = {"train": 0.4, "cal": 0.2, "test": 0.4}
CONFIRM_DELAY_DAYS = 30  # simulated delay before a campaign is confirmed as fraud
GLOBAL_EXPRESS = 0.2
LABEL_COLUMNS = ("is_injected", "scenario_id", "campaign_id", "typology", "camouflage_level", "mimic",
                 "params_json", "seed", "is_fraud", "label_misuse", "label_foreign_senders", "label_payoff_max",
                 "label_drop_consignee", "risk_level", "phase", "cost_ratio", "true_weight_kg", "split")
SPLIT_NAMES = ("train", "cal", "test")


def _hex(rng, n):
    return "".join(rng.choice(list("0123456789abcdef"), n))


@dataclass
class InjectionResult:
    rows: pd.DataFrame
    confirmed: list[dict]
    changes: pd.DataFrame
    summary: dict = field(default_factory=dict)


@dataclass
class Victim:
    account_id: str
    home_zip5: str
    home_uf: str
    rows: pd.DataFrame  # history before onset
    median_cost: float
    p90_weight: float
    p90_km: float
    rate_active: float     # bookings per active day
    p90_daily: float
    express_share: float


class InjectionContext:
    """Real value pools and samplers built once from the real bookings (with billing layer)."""

    def __init__(self, real: pd.DataFrame, pool: pd.DataFrame):
        self.real = real.sort_values(["booked_at", "booking_id"]).reset_index(drop=True)
        r = self.real
        self.parcels = r[["weight_kg", "length_cm", "width_cm", "height_cm", "category", "declared_value",
                          "n_items"]].reset_index(drop=True)
        self.parcel_w_order = np.argsort(self.parcels.weight_kg.to_numpy())
        self.parcel_w_sorted = self.parcels.weight_kg.to_numpy()[self.parcel_w_order]
        self.by_cat = {c: np.flatnonzero(self.parcels.category.to_numpy() == c) for c in self.parcels.category.unique()}
        self.high_value = np.flatnonzero(self.parcels.category.isin(HIGH_VALUE_CATEGORIES).to_numpy())
        self.pool = pool.reset_index(drop=True)
        self.zip_idx = {z: i for i, z in enumerate(self.pool.zip5)}
        self.lat, self.lng = self.pool.lat.to_numpy(), self.pool.lng.to_numpy()
        dest_w = self.pool.n_customers.to_numpy(float)
        orig_w = self.pool.n_sellers.to_numpy(float) + 0.05 * dest_w + 1e-3
        self.dest_w, self.orig_w = dest_w / dest_w.sum(), orig_w / orig_w.sum()
        self.uf_idx = {uf: np.flatnonzero(self.pool.uf.to_numpy() == uf) for uf in self.pool.uf.unique()}
        reg = self.pool.zip5.str[:2].to_numpy()
        self.region_idx = {g: np.flatnonzero(reg == g) for g in np.unique(reg)}
        self.region_idx = {g: ix for g, ix in self.region_idx.items() if self.dest_w[ix].sum() > 0 and len(ix) >= 5}
        # freight lane model: cost ~ a + b*kg + c*km/1000, residual sampled from the same bin
        ok = r[["weight_kg", "distance_km", "carrier_cost"]].dropna()
        X = np.column_stack([np.ones(len(ok)), ok.weight_kg, ok.distance_km / 1000.0])
        self.freight_coef = np.linalg.lstsq(X, ok.carrier_cost.to_numpy(), rcond=None)[0]
        resid = ok.carrier_cost.to_numpy() - X @ self.freight_coef
        self.w_edges = np.quantile(ok.weight_kg, [0.2, 0.4, 0.6, 0.8])
        self.d_edges = np.quantile(ok.distance_km, [0.2, 0.4, 0.6, 0.8])
        wb, db = np.digitize(ok.weight_kg, self.w_edges), np.digitize(ok.distance_km, self.d_edges)
        self.resid = {(i, j): resid[(wb == i) & (db == j)] for i in range(5) for j in range(5)}
        # fine bins of REAL freight values: injected freight is a real value from the same weight/distance bin
        self.fw_edges = np.quantile(ok.weight_kg, np.linspace(0, 1, 13)[1:-1])
        self.fd_edges = np.quantile(ok.distance_km, np.linspace(0, 1, 13)[1:-1])
        fwb, fdb = np.digitize(ok.weight_kg, self.fw_edges), np.digitize(ok.distance_km, self.fd_edges)
        costs = ok.carrier_cost.to_numpy()
        self.fine = {(i, j): costs[(fwb == i) & (fdb == j)] for i in range(12) for j in range(12)}
        self.dest_by_origin_uf = {u: g.to_numpy() for u, g in r.groupby("origin_uf").dest_zip5}
        lag = (r.first_scan_at - r.booked_at).dt.total_seconds().to_numpy() / 3600.0
        self.lag_hours = np.where(np.isnan(lag) | (lag >= 0), lag, np.nan)
        tod = r.booked_at - r.booked_at.dt.floor("D")
        self.tod_seconds = tod.dt.total_seconds().to_numpy()
        self.sec_of_min = np.floor(self.tod_seconds % 60)  # real seconds-of-minute (approval latency pattern)
        self.acc_rows = r.groupby("account_id", sort=False).indices
        self.split_counts = r.split.value_counts().to_dict() if "split" in r else {}

    @classmethod
    def build(cls, real: pd.DataFrame, pool: pd.DataFrame) -> "InjectionContext":
        return cls(real, pool)

    # samplers -------------------------------------------------------------------------------
    def freight(self, rng, weight: float, km: float) -> float:
        vals = self.fine[(int(np.digitize(weight, self.fw_edges)), int(np.digitize(km, self.fd_edges)))]
        if len(vals):
            return round(float(rng.choice(vals)), 2)
        pred = self.freight_coef @ np.array([1.0, weight, km / 1000.0])
        res = self.resid[(int(np.digitize(weight, self.w_edges)), int(np.digitize(km, self.d_edges)))]
        e = rng.choice(res) if len(res) else 0.0
        return round(float(max(pred + e, 0.0)), 2)

    def km(self, z_a: str, z_b: str) -> float:
        i, j = self.zip_idx[z_a], self.zip_idx[z_b]
        return float(haversine_km(self.lat[i], self.lng[i], self.lat[j], self.lng[j]))

    def zip_dest(self, rng, uf: str | None = None, region: str | None = None) -> str:
        ix = self.region_idx[region] if region else (self.uf_idx.get(uf) if uf else None)
        if ix is None or self.dest_w[ix].sum() == 0:
            return str(self.pool.zip5[rng.choice(len(self.pool), p=self.dest_w)])
        w = self.dest_w[ix] / self.dest_w[ix].sum()
        return str(self.pool.zip5[ix[rng.choice(len(ix), p=w)]])

    def zip_dest_like(self, rng, origin_uf: str) -> str:
        """Destination of a random real booking shipped from this state (a real lane)."""
        cands = self.dest_by_origin_uf.get(origin_uf)
        for _ in range(20):
            if cands is None:
                break
            z = str(rng.choice(cands))
            if z in self.zip_idx:
                return z
        return self.zip_dest(rng)

    def zip_origin(self, rng, uf: str | None = None, exclude_uf: str | None = None) -> str:
        if uf is not None:
            ix = self.uf_idx[uf]
            w = self.orig_w[ix] / self.orig_w[ix].sum()
            return str(self.pool.zip5[ix[rng.choice(len(ix), p=w)]])
        while True:
            z = str(self.pool.zip5[rng.choice(len(self.pool), p=self.orig_w)])
            if exclude_uf is None or self.uf_of(z) != exclude_uf:
                return z

    def uf_of(self, z: str) -> str:
        return str(self.pool.uf[self.zip_idx[z]])

    def zip_far(self, rng, origin: str, min_km: float, uf: str | None = None) -> str:
        cands = [self.zip_dest(rng, uf=uf) for _ in range(40)]
        d = [self.km(origin, c) for c in cands]
        far = [c for c, x in zip(cands, d) if x >= min_km]
        return str(rng.choice(far)) if far else cands[int(np.argmax(d))]

    def parcel_heavy(self, rng, min_kg: float) -> int:
        k = np.searchsorted(self.parcel_w_sorted, min_kg, side="right")
        if k >= len(self.parcel_w_sorted) - 5:
            k = int(len(self.parcel_w_sorted) * 0.95)
        return int(self.parcel_w_order[rng.integers(k, len(self.parcel_w_sorted))])

    def parcel_cat(self, rng, cats: list[str]) -> int:
        parts = [self.by_cat[c] for c in cats if c in self.by_cat]
        if not parts:
            return self.parcel_any(rng)
        return int(rng.choice(np.concatenate(parts)))

    def parcel_any(self, rng) -> int:
        return int(rng.integers(len(self.parcels)))

    def lag(self, rng) -> float:
        return float(rng.choice(self.lag_hours))

    def victim(self, account_id: str, onset: pd.Timestamp) -> Victim | None:
        rows = self.real.iloc[self.acc_rows[account_id]]
        rows = rows[rows.booked_at < onset]
        if len(rows) < MIN_PRIOR:
            return None
        home = rows.origin_zip5.mode().sort_values().iloc[0]
        days = rows.booked_at.dt.floor("D").value_counts()
        return Victim(
            account_id=account_id, home_zip5=str(home), home_uf=str(rows[rows.origin_zip5 == home].origin_uf.iloc[0]),
            rows=rows, median_cost=float(max(rows.carrier_cost.median(), 1.0)),
            p90_weight=float(rows.weight_kg.quantile(0.9)), p90_km=float(rows.distance_km.quantile(0.9)),
            rate_active=float(days.mean()), p90_daily=float(max(days.quantile(0.9), 1.0)),
            express_share=float((rows.service == "express").mean()),
        )


class _Campaign:
    """Accumulates rows for one campaign (one or more scenario accounts)."""

    def __init__(self, inj: "_Injector", typ: str, split: str, camo: int, mimic: bool, params: dict):
        self.inj, self.typ, self.split, self.camo, self.mimic = inj, typ, split, camo, mimic
        self.id = f"{typ}_s{inj.seed}_{split}_{inj.next_id()}"
        self.params = {"typology": typ, "camouflage_level": camo, "mimic": mimic, **params}
        self.rows: list[dict] = []
        self.end = next(s.end for s in SPLITS if s.name == split)

    def add(self, *, scenario: str, account_id: str, sender_id: str, t: pd.Timestamp, origin: str, dest: str,
            parcel: int, service: str, billing: dict, consignee: str | None = None, phase: str = "",
            foreign: bool = False, payoff: bool = False, drop: bool = False, burst: bool = False,
            ref_cost: float = 17.06, shrink: tuple[float, float] | None = None) -> None:
        if t >= self.end - pd.Timedelta(minutes=1):
            return
        ctx, rng = self.inj.ctx, self.inj.rng
        p = ctx.parcels.iloc[parcel]
        w, L, W, H = float(p.weight_kg), float(p.length_cm), float(p.width_cm), float(p.height_cm)
        true_w = w
        if shrink is not None:  # T6: declared weight and dims below the truth
            w = round(w * shrink[0], 3)
            L, W, H = (round(x * shrink[1], 1) for x in (L, W, H))
        km = ctx.km(origin, dest)
        cost = ctx.freight(rng, w, km)
        fraud = self.typ in FRAUD_TYPOLOGIES
        lag = ctx.lag(rng)
        self.rows.append({
            "booking_id": "bk_" + _hex(rng, 16), "order_id": None, "account_id": account_id, "sender_id": sender_id,
            "booked_at": t, "first_scan_at": (t + pd.Timedelta(hours=lag)).floor("s") if not np.isnan(lag) else pd.NaT,
            "origin_zip5": origin, "origin_zip3": origin[:3], "origin_uf": ctx.uf_of(origin),
            "consignee_id": consignee or _hex(rng, 32),
            "dest_zip5": dest, "dest_zip3": dest[:3], "dest_uf": ctx.uf_of(dest),
            "weight_kg": w, "length_cm": L, "width_cm": W, "height_cm": H, "n_items": int(p.n_items),
            "category": str(p.category), "declared_value": float(p.declared_value), "carrier_cost": cost,
            "distance_km": km, "service": service, **billing, "origin_synthetic": False,
            "hn_change": "" if fraud else self.typ.removeprefix("HN_"),
            "is_injected": True, "scenario_id": scenario, "campaign_id": self.id, "typology": self.typ,
            "camouflage_level": self.camo, "mimic": self.mimic, "params_json": json.dumps(self.params, sort_keys=True),
            "seed": self.inj.seed, "is_fraud": fraud, "label_misuse": fraud,
            "label_foreign_senders": bool(fraud and foreign), "label_payoff_max": bool(fraud and payoff),
            "label_drop_consignee": bool(fraud and drop),
            "risk_level": risk_level(fraud, cost / max(ref_cost, 1.0), burst=burst),
            "phase": phase, "cost_ratio": cost / max(ref_cost, 1.0), "true_weight_kg": true_w, "split": self.split,
        })


class _Injector:
    def __init__(self, ctx: InjectionContext, seed: int):
        self.ctx, self.seed = ctx, seed
        self.rng = np.random.default_rng(MASTER_SEED + seed)
        self._n = 0
        accs = np.array(sorted(ctx.acc_rows))
        lab = self.rng.choice(SPLIT_NAMES, size=len(accs), p=[POOL_SHARE[s] for s in SPLIT_NAMES])
        self.pools = {s: list(accs[lab == s]) for s in SPLIT_NAMES}
        self.used: set[str] = set()
        self.campaigns: list[_Campaign] = []
        self.changes: list[dict] = []

    def next_id(self) -> int:
        self._n += 1
        return self._n

    # helpers --------------------------------------------------------------------------------
    def pick_victim(self, split: str, onset: pd.Timestamp) -> Victim | None:
        pool = self.pools[split]
        for _ in range(200):
            acc = pool[self.rng.integers(len(pool))]
            if acc in self.used:
                continue
            v = self.ctx.victim(acc, onset)
            if v is None or v.rows.booked_at.iloc[-1] < onset - pd.Timedelta(days=60):
                continue
            self.used.add(acc)
            return v
        return None

    def onset(self, split: str, room_days: float = 20) -> pd.Timestamp:
        s = next(x for x in SPLITS if x.name == split)
        span = (s.end - s.start).total_seconds() / 86400 - room_days
        return (s.start + pd.Timedelta(days=float(self.rng.uniform(0, max(span, 1))))).floor("min")

    def tod(self, camo: int, victim: Victim | None, off_hours: bool = False) -> float:
        rng = self.rng
        if camo < 0:  # legit behaviour: the real global time-of-day distribution
            return float(rng.choice(self.ctx.tod_seconds))
        if off_hours:
            return float(rng.uniform(0, 6 * 3600))
        if camo >= 1 and victim is not None:
            secs = (victim.rows.booked_at - victim.rows.booked_at.dt.floor("D")).dt.total_seconds().to_numpy()
            return float((rng.choice(secs) + rng.normal(0, 1800)) % 86400)
        if rng.random() < 0.5:
            return float(rng.uniform(0, 86400))
        return float(rng.choice(self.ctx.tod_seconds))

    def times(self, start: pd.Timestamp, n: int, days: float, camo: int, victim: Victim | None,
              ramp: bool = False, off_hours: bool = False) -> list[pd.Timestamp]:
        u = self.rng.uniform(0, 1, n)
        off = np.sort(np.sqrt(u) if ramp else u) * days
        out = []
        for o in off:
            day = (start + pd.Timedelta(days=float(o))).floor("D")
            t = day + pd.Timedelta(seconds=self.tod(camo, victim, off_hours))
            t = t if t >= start else t + pd.Timedelta(days=1)
            t = t.floor("min") + pd.Timedelta(seconds=float(self.rng.choice(self.ctx.sec_of_min)))
            out.append(t)
        return sorted(out)

    def billing_victim(self, v: Victim, t: pd.Timestamp, new_dev: dict | None, contact_reset: pd.Timestamp | None) -> dict:
        last = v.rows.iloc[-1]
        el = (t - last.booked_at).total_seconds() / 86400
        if new_dev is not None:
            ch, dev, age = new_dev["channel"], new_dev["device_id"], (t - new_dev["linked_at"]).total_seconds() / 86400
        else:
            ch, dev, age = last.channel, last.device_id, last.login_device_age_days + el
        oc = last.owner_contact_age_days
        if contact_reset is not None:
            oc = (t - contact_reset).total_seconds() / 86400
        elif not np.isnan(oc):
            oc = oc + el
        return {"channel": ch, "device_id": dev, "login_device_age_days": max(float(age), 0.0),
                "payment_method": last.payment_method, "payer_instrument_id": last.payer_instrument_id,
                "owner_contact_age_days": float(oc) if not np.isnan(oc) else np.nan}

    def new_device(self, at: pd.Timestamp) -> dict:
        return {"device_id": "dev_" + _hex(self.rng, 12), "linked_at": at,
                "channel": "api" if self.rng.random() < 0.6 else "web"}

    def victim_parcel(self, v: Victim) -> int:
        # the victim's own real parcels are rows of ctx.real; parcels share its index
        return int(self.rng.choice(v.rows.index.to_numpy()))

    def victim_dest(self, v: Victim) -> str:
        return self.ctx.zip_dest(self.rng, uf=str(self.rng.choice(v.rows.dest_uf.to_numpy())))

    def service(self, payoff: bool, v: Victim | None = None) -> str:
        p = 0.7 if payoff else (v.express_share if v is not None else GLOBAL_EXPRESS)
        return "express" if self.rng.random() < p else "standard"

    def takeover_login(self, camo: int, onset: pd.Timestamp, p_new: float = 0.8):
        rng = self.rng
        new_dev = self.new_device(onset) if camo == 0 and rng.random() < p_new else None
        reset = onset if camo == 0 and rng.random() < 0.3 else None
        return new_dev, reset

    def payoff_parcel_and_dest(self, v: Victim, origin: str, uf: str | None) -> tuple[int, str]:
        rng, ctx = self.rng, self.ctx
        parcel = ctx.parcel_heavy(rng, max(v.p90_weight, float(np.quantile(ctx.parcel_w_sorted, 0.75))))
        dest = ctx.zip_far(rng, origin, max(v.p90_km, 800.0), uf=uf)
        return parcel, dest

    # typologies -----------------------------------------------------------------------------
    def t1(self, split, camo, mimic):
        rng, ctx = self.rng, self.ctx
        onset = self.onset(split, 25)
        v = self.pick_victim(split, onset)
        if v is None:
            return
        K = int(rng.integers(5, 41))
        c = _Campaign(self, "T1", split, camo, mimic, {"K_origin_pool": K})
        same_state = camo >= 1 or rng.random() < 0.3
        origins = [ctx.zip_origin(rng, uf=v.home_uf if same_state else None,
                                  exclude_uf=None if same_state else v.home_uf) for _ in range(K)]
        senders = [_hex(rng, 32) for _ in range(K)]
        self._t1_on(c, v, onset, origins, senders, f"{c.id}_a")
        if camo == 0 and rng.random() < 0.5 and c.rows:  # ring moves to a second victim (Harrod)
            last = max(r["booked_at"] for r in c.rows)
            onset2 = (last + pd.Timedelta(days=float(rng.uniform(1, 10)))).floor("min")
            v2 = self.pick_victim(split, onset2)
            if v2 is not None:
                k = max(1, K // 2)
                o2 = origins[:k] + [ctx.zip_origin(rng) for _ in range(K - k)]
                self._t1_on(c, v2, onset2, o2, senders[:k] + [_hex(rng, 32) for _ in range(K - k)], f"{c.id}_b")
        self.campaigns.append(c)

    def _t1_on(self, c, v, onset, origins, senders, scen):
        rng, ctx, camo = self.rng, self.ctx, c.camo
        new_dev, reset = self.takeover_login(camo, onset)
        dest_uf = (lambda: str(rng.choice(v.rows.dest_uf.to_numpy()))) if (camo >= 1 or c.mimic) else (lambda: None)
        # phase A: own use from one new origin, one sender, shoes/sports-like contents
        nA, dA = int(rng.integers(2, 5)), float(rng.uniform(2, 7))
        oA = ctx.zip_origin(rng, uf=v.home_uf) if camo >= 1 else ctx.zip_origin(rng, exclude_uf=v.home_uf if rng.random() < 0.7 else None)
        sA = _hex(rng, 32)
        for t in self.times(onset, nA, dA, camo, v):
            parcel = self.victim_parcel(v) if camo == 2 else ctx.parcel_cat(rng, ["fashion", "sports_leisure"])
            c.add(scenario=scen, account_id=v.account_id, sender_id=sA, t=t, origin=oA,
                  dest=ctx.zip_dest(rng, uf=dest_uf()), parcel=parcel, service=self.service(False, v),
                  billing=self.billing_victim(v, t, new_dev, reset), phase="A", foreign=True, ref_cost=v.median_cost)
        # phase B: resale, ramping volume, pool of third-party sender origins
        nB, dB = int(rng.integers(4, 11)), float(rng.uniform(3, 14))
        if camo == 2:
            dB = max(dB, nB / v.p90_daily)
        startB = onset + pd.Timedelta(days=dA)
        for t in self.times(startB, nB, dB, camo, v, ramp=True):
            k = int(rng.integers(len(origins)))
            payoff = rng.random() < (0.5 if camo < 2 else 0.35)
            if payoff:
                parcel, dest = self.payoff_parcel_and_dest(v, origins[k], dest_uf())
            else:
                parcel = self.victim_parcel(v) if (camo == 2 or c.mimic) else ctx.parcel_any(rng)
                dest = ctx.zip_dest(rng, uf=dest_uf())
            c.add(scenario=scen, account_id=v.account_id, sender_id=senders[k], t=t, origin=origins[k], dest=dest,
                  parcel=parcel, service=self.service(payoff, v), billing=self.billing_victim(v, t, new_dev, reset),
                  phase="B", foreign=True, payoff=payoff, ref_cost=v.median_cost)

    def t2(self, split, camo, mimic):
        rng, ctx = self.rng, self.ctx
        start = self.onset(split, 30)
        acc = _hex(rng, 32)
        created = start - pd.Timedelta(days=float(rng.uniform(0, 3)))
        origin = ctx.zip_origin(rng)
        prior = [x for x in self.campaigns if x.typ == "T2" and x.split == split and x.rows]
        link_p = (0.6, 0.2, 0.0)[camo]
        linked = prior[int(rng.integers(len(prior)))] if prior and rng.random() < link_p else None
        if linked is not None:  # same drop-off zip and reused consignees as a defaulted bust-out account
            origin = linked.rows[0]["origin_zip5"]
        n, D = int(rng.integers(5, 13)), float(rng.uniform(7, 30))
        c = _Campaign(self, "T2", split, camo, mimic, {"n": n, "days": round(D, 1),
                                                         "linked_to": linked.id if linked else None})
        dev = {"device_id": "dev_" + _hex(rng, 12), "channel": "web" if rng.random() < 0.7 else "api"}
        pay = str(rng.choice(["account_billing", "card", "ach"], p=[0.3, 0.5, 0.2]))
        ins = "ins_" + _hex(rng, 12)
        long_p = (0.7, 0.4, 0.2)[camo]
        for t in self.times(start, n, D, camo, None):
            hv = camo < 2 or rng.random() < 0.5
            parcel = int(rng.choice(ctx.high_value)) if hv else ctx.parcel_any(rng)
            far = rng.random() < long_p
            dest = ctx.zip_far(rng, origin, float(np.quantile(ctx.real.distance_km.dropna(), 0.75))) if far else ctx.zip_dest(rng)
            cons = None
            if linked is not None and rng.random() < 0.3:
                cons = linked.rows[int(rng.integers(len(linked.rows)))]["consignee_id"]
            age = (t - created).total_seconds() / 86400
            c.add(scenario=c.id, account_id=acc, sender_id=acc, t=t, origin=origin, dest=dest, parcel=parcel,
                  service=self.service(far), consignee=cons, payoff=bool(far and hv),
                  billing={"channel": dev["channel"], "device_id": dev["device_id"], "login_device_age_days": age,
                           "payment_method": pay, "payer_instrument_id": ins, "owner_contact_age_days": age})
        self.campaigns.append(c)

    def t3(self, split, camo, mimic):
        rng, ctx = self.rng, self.ctx
        start = self.onset(split, 45)
        region = str(rng.choice(sorted(ctx.region_idx)))
        M = int(rng.integers(3, 6))
        victims = [v for v in (self.pick_victim(split, start) for _ in range(M)) if v is not None]
        if len(victims) < 2:
            return
        D = int(rng.integers(2, 4))
        c = _Campaign(self, "T3", split, camo, mimic, {"drops": D, "senders": len(victims), "metro_zip2": region})
        logins = {v.account_id: self.takeover_login(camo, start, p_new=0.5) for v in victims}
        third = {v.account_id: (_hex(rng, 32), ctx.zip_origin(rng)) for v in victims}
        for d in range(D):
            cons = _hex(rng, 32)
            dz = ctx.zip_dest(rng, region=region) if rng.random() < 0.9 else ctx.zip_dest(rng)
            first = start + pd.Timedelta(days=float(rng.uniform(0, 10) + rng.uniform(2, 7)))
            life = float(rng.uniform(21, 35))
            k = int(min(25, 3 + rng.negative_binomial(2, 2 / 7)))
            for t in self.times(first, k, life, camo, None):
                v = victims[int(rng.integers(len(victims)))]
                foreign = camo == 0 and rng.random() < 0.5
                snd, org = third[v.account_id] if foreign else (v.account_id, v.home_zip5)
                r = rng.random()
                if camo == 2 and mimic and rng.random() < 0.5:
                    parcel = self.victim_parcel(v)
                elif r < 0.55:
                    parcel = ctx.parcel_cat(rng, ["phones"])
                elif r < 0.8:
                    parcel = ctx.parcel_cat(rng, ["electronics", "computers"])
                else:
                    parcel = ctx.parcel_cat(rng, ["watches_gifts"])
                nd, reset = logins[v.account_id]
                c.add(scenario=f"{c.id}_drop{d}", account_id=v.account_id, sender_id=snd, t=t, origin=org, dest=dz,
                      parcel=parcel, service=self.service(False, v), consignee=cons,
                      billing=self.billing_victim(v, t, nd, reset), foreign=foreign, drop=True, ref_cost=v.median_cost)
        self.campaigns.append(c)

    def t4(self, split, camo, mimic):
        rng, ctx = self.rng, self.ctx
        onset = self.onset(split, 25)
        v = self.pick_victim(split, onset)
        if v is None:
            return
        n = int(rng.integers(5, 13))
        D = max(float(rng.uniform(5, 21)), n / (v.rate_active * rng.uniform(0.8, 1.5)))
        c = _Campaign(self, "T4", split, camo, mimic, {"n": n, "days": round(D, 1)})
        new_dev, reset = self.takeover_login(camo, onset)
        for t in self.times(onset, n, D, camo, v):
            foreign = camo == 0 and rng.random() < 0.5
            org = ctx.zip_origin(rng) if foreign else v.home_zip5
            snd = _hex(rng, 32) if foreign else v.account_id
            uf = str(rng.choice(v.rows.dest_uf.to_numpy())) if mimic else None
            parcel, dest = self.payoff_parcel_and_dest(v, org, uf)
            c.add(scenario=c.id, account_id=v.account_id, sender_id=snd, t=t, origin=org, dest=dest, parcel=parcel,
                  service=self.service(True), billing=self.billing_victim(v, t, new_dev, reset),
                  foreign=foreign, payoff=True, ref_cost=v.median_cost)
        self.campaigns.append(c)

    def t5(self, split, camo, mimic):
        rng, ctx = self.rng, self.ctx
        onset = self.onset(split, 20)
        v = self.pick_victim(split, onset)
        if v is None:
            return
        n_test, gap, n_burst, D = int(rng.integers(1, 4)), float(rng.uniform(1, 7)), int(rng.integers(5, 13)), float(rng.uniform(1, 3))
        c = _Campaign(self, "T5", split, camo, mimic, {"tests": n_test, "gap_days": round(gap, 1), "burst": n_burst})
        new_dev, reset = self.takeover_login(camo, onset)
        for t in self.times(onset, n_test, 1.0, camo, v, off_hours=camo == 0):
            c.add(scenario=c.id, account_id=v.account_id, sender_id=v.account_id, t=t, origin=v.home_zip5,
                  dest=self.victim_dest(v), parcel=self.victim_parcel(v), service="standard",
                  billing=self.billing_victim(v, t, new_dev, reset), phase="test", ref_cost=v.median_cost)
        K = int(rng.integers(3, 9))
        origins = [ctx.zip_origin(rng, uf=v.home_uf if camo >= 1 else None) for _ in range(K)]
        senders = [_hex(rng, 32) for _ in range(K)]
        for t in self.times(onset + pd.Timedelta(days=1 + gap), n_burst, D, camo, v):
            k = int(rng.integers(K))
            payoff = rng.random() < 0.5
            if payoff:
                parcel, dest = self.payoff_parcel_and_dest(v, origins[k], None)
            else:
                parcel, dest = (self.victim_parcel(v) if mimic else ctx.parcel_any(rng)), ctx.zip_dest(rng)
            c.add(scenario=c.id, account_id=v.account_id, sender_id=senders[k], t=t, origin=origins[k], dest=dest,
                  parcel=parcel, service=self.service(payoff, v), billing=self.billing_victim(v, t, new_dev, reset),
                  phase="burst", foreign=True, payoff=payoff, burst=True, ref_cost=v.median_cost)
        self.campaigns.append(c)

    def t6(self, split, camo, mimic):
        rng = self.rng
        onset = self.onset(split, 30)
        v = self.pick_victim(split, onset)
        if v is None:
            return
        n, D = int(rng.integers(5, 11)), float(rng.uniform(7, 30))
        c = _Campaign(self, "T6", split, camo, mimic, {"n": n, "days": round(D, 1)})
        for t in self.times(onset, n, D, max(camo, 1), v):
            shrink = (float(rng.uniform(0.3, 0.7)), float(rng.uniform(0.6, 0.9)))
            c.add(scenario=c.id, account_id=v.account_id, sender_id=v.account_id, t=t, origin=v.home_zip5,
                  dest=self.victim_dest(v), parcel=self.victim_parcel(v), service=self.service(False, v),
                  billing=self.billing_victim(v, t, None, None), shrink=shrink, ref_cost=v.median_cost)
        self.campaigns.append(c)

    def t7(self, split, camo, mimic):
        rng, ctx = self.rng, self.ctx
        onset = self.onset(split, 30)
        v = self.pick_victim(split, onset)
        if v is None:
            return
        S, n, D = int(rng.integers(1, 4)), int(rng.integers(4, 11)), float(rng.uniform(3, 30))
        c = _Campaign(self, "T7", split, camo, mimic, {"senders": S, "n": n, "days": round(D, 1)})
        snd = []
        for _ in range(S):
            same = camo >= 1 or rng.random() < 0.3
            snd.append((_hex(rng, 32), ctx.zip_origin(rng, uf=v.home_uf) if same else ctx.zip_origin(rng, exclude_uf=v.home_uf),
                        {"device_id": "dev_" + _hex(rng, 12), "age0": float(rng.uniform(30, 900)),
                         "channel": "api" if rng.random() < 0.5 else "web"}))
        last = v.rows.iloc[-1]
        for t in self.times(onset, n, D, camo, v):
            sid, org, dev = snd[int(rng.integers(S))]
            payoff = rng.random() < 0.3
            if payoff:
                parcel, dest = self.payoff_parcel_and_dest(v, org, None)
            else:
                parcel = self.victim_parcel(v) if (mimic or camo == 2) else ctx.parcel_any(rng)
                dest = self.victim_dest(v) if camo >= 1 else ctx.zip_dest(rng)
            bill = self.billing_victim(v, t, None, None)
            bill.update(channel=dev["channel"], device_id=dev["device_id"],
                        login_device_age_days=dev["age0"] + (t - onset).total_seconds() / 86400,
                        payment_method="account_billing", payer_instrument_id=last.payer_instrument_id)
            c.add(scenario=c.id, account_id=v.account_id, sender_id=sid, t=t, origin=org, dest=dest, parcel=parcel,
                  service=self.service(payoff, v), billing=bill, foreign=True, payoff=payoff, ref_cost=v.median_cost)
        self.campaigns.append(c)

    def hn_new_channel(self, split, camo, mimic):
        rng, ctx = self.rng, self.ctx
        onset = self.onset(split, 20)
        v = self.pick_victim(split, onset)
        if v is None:
            return
        n, D = int(rng.integers(15, 41)), float(rng.uniform(14, 45))
        announced = bool(rng.random() < 0.5)
        c = _Campaign(self, "HN_new_channel", split, 0, True, {"n": n, "days": round(D, 1), "announced": announced})
        dev = {"device_id": "dev_" + _hex(rng, 12), "linked_at": onset, "channel": "api"}
        for t in self.times(onset, n, D, -1, v):
            c.add(scenario=c.id, account_id=v.account_id, sender_id=v.account_id, t=t, origin=v.home_zip5,
                  dest=ctx.zip_dest_like(rng, v.home_uf), parcel=self.victim_parcel(v), service=self.service(False, v),
                  billing=self.billing_victim(v, t, dev, None), ref_cost=v.median_cost)
        if announced:
            self.changes.append({"account_id": v.account_id, "kind": "new_channel", "at": onset,
                                 "announced_at": onset - pd.Timedelta(days=float(rng.uniform(1, 7))), "announced": True})
        self.campaigns.append(c)

    def hn_3pl(self, split, camo, mimic):
        rng, ctx = self.rng, self.ctx
        start = self.onset(split, 30)
        acc = _hex(rng, 32)
        created = start - pd.Timedelta(days=float(rng.uniform(5, 60)))
        uf = ctx.uf_of(ctx.zip_origin(rng))
        K, n, D = int(rng.integers(5, 21)), int(rng.integers(20, 51)), float(rng.uniform(30, 90))
        c = _Campaign(self, "HN_3pl", split, 0, False, {"senders": K, "n": n, "days": round(D, 1)})
        snd = [(_hex(rng, 32), ctx.zip_origin(rng, uf=uf)) for _ in range(K)]
        dev = "dev_" + _hex(rng, 12)
        ins = "ins_" + _hex(rng, 12)
        for t in self.times(start, n, D, -1, None):
            sid, org = snd[int(rng.integers(K))]
            age = (t - created).total_seconds() / 86400
            c.add(scenario=c.id, account_id=acc, sender_id=sid, t=t, origin=org, dest=ctx.zip_dest_like(rng, uf),
                  parcel=ctx.parcel_any(rng), service=self.service(False),
                  billing={"channel": "api", "device_id": dev, "login_device_age_days": age,
                           "payment_method": "account_billing", "payer_instrument_id": ins, "owner_contact_age_days": age})
        self.campaigns.append(c)

    def run(self) -> InjectionResult:
        gens = {"T1": self.t1, "T2": self.t2, "T3": self.t3, "T4": self.t4, "T5": self.t5, "T6": self.t6,
                "T7": self.t7, "HN_new_channel": self.hn_new_channel, "HN_3pl": self.hn_3pl}
        counts = {}
        for split in SPLIT_NAMES:
            n_real = self.ctx.split_counts.get(split, 0)
            for typ, rate in CAMPAIGN_RATES.items():
                if split != "test" and typ in HELD_OUT:
                    continue
                k = max(1, int(round(rate * n_real / 1000))) if n_real else 0
                counts[(split, typ)] = k
                for _ in range(k):
                    camo = int(self.rng.choice(3, p=CAMO_P))
                    mimic = bool(camo == 2 or self.rng.random() < MIMIC_P)
                    gens[typ](split, camo, mimic)
        rows = [r for c in self.campaigns for r in c.rows]
        df = pd.DataFrame(rows)
        df = df.sort_values(["booked_at", "booking_id"]).reset_index(drop=True)
        df["split"] = assign_split(df.booked_at).to_numpy()
        confirmed = []
        for c in self.campaigns:
            if c.typ not in FRAUD_TYPOLOGIES or not c.rows:
                continue
            last = max(r["booked_at"] for r in c.rows)
            ents = sorted({r["consignee_id"] for r in c.rows} |
                          {r["sender_id"] for r in c.rows if r["sender_id"] != r["account_id"]})
            for acc in sorted({r["account_id"] for r in c.rows}):
                confirmed.append({"account_id": acc, "campaign_id": c.id,
                                  "confirmed_at": last + pd.Timedelta(days=CONFIRM_DELAY_DAYS), "entities": ents})
        ch = pd.DataFrame(self.changes, columns=["account_id", "kind", "at", "announced_at", "announced"])
        summary = {"campaigns": {f"{s}/{t}": k for (s, t), k in counts.items()}}
        return InjectionResult(rows=df, confirmed=confirmed, changes=ch, summary=summary)


def inject(ctx: InjectionContext, seed: int) -> InjectionResult:
    return _Injector(ctx, seed).run()
