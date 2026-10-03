import numpy as np
import pandas as pd
import pytest

UFS = ["SP", "RJ", "MG", "PR", "RS", "BA"]


def make_bookings(n_accounts=300, per_account=(5, 60), seed=0, start="2017-01-01", days=500):
    """Synthetic booking table in the olist.build_bookings schema (for unit tests only)."""
    rng = np.random.default_rng(seed)
    rows = []
    t0 = pd.Timestamp(start)
    for a in range(n_accounts):
        acc = f"{a:032x}"
        uf = UFS[a % len(UFS)]
        z5 = f"{(a * 37) % 99999:05d}"
        n = int(rng.integers(*per_account))
        first = rng.uniform(0, days * 0.6)
        ts = np.sort(rng.uniform(first, days, n))
        for i, d in enumerate(ts):
            duf = UFS[int(rng.integers(len(UFS)))]
            dz = f"{int(rng.integers(1000, 99999)):05d}"
            w = float(rng.lognormal(0, 0.8))
            rows.append(dict(
                booking_id=f"bk_{a:06d}_{i:04d}", order_id=f"o{a}_{i}", account_id=acc, sender_id=acc,
                booked_at=t0 + pd.Timedelta(days=float(d)), first_scan_at=t0 + pd.Timedelta(days=float(d) + 2),
                origin_zip5=z5, origin_zip3=z5[:3], origin_uf=uf,
                consignee_id=f"c{int(rng.integers(0, 10**9)):031x}", dest_zip5=dz, dest_zip3=dz[:3], dest_uf=duf,
                weight_kg=round(w, 3), length_cm=30.0, width_cm=20.0, height_cm=10.0, n_items=1,
                category=["home", "fashion", "phones"][i % 3], declared_value=round(50 + 30 * w, 2),
                carrier_cost=round(9 + 3 * w + rng.normal(0, 2) ** 2, 2), distance_km=float(rng.uniform(5, 2500)),
                service="express" if rng.random() < 0.2 else "standard",
            ))
    return pd.DataFrame(rows).sort_values(["booked_at", "booking_id"]).reset_index(drop=True)


@pytest.fixture
def toy_bookings():
    return make_bookings()
