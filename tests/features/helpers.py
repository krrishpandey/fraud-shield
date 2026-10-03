import numpy as np
import pandas as pd

from fraudshield.contracts import Booking

BASE = dict(channel="web", login_device_age_days=100.0, payment_method="account_billing",
            origin_uf="SP", origin_zip3="013", dest_uf="RJ", dest_zip3="220", weight_kg=1.0,
            length_cm=20.0, width_cm=10.0, height_cm=10.0, service="standard", category="home",
            declared_value=50.0, carrier_cost=15.0, owner_contact_age_days=300.0)


def bk(booking_id, account_id, booked_at, **kw):
    d = dict(BASE)
    d.update(kw)
    d.setdefault("sender_id", account_id)
    d.setdefault("consignee_id", "cons_" + booking_id)
    return Booking(booking_id=booking_id, account_id=account_id, booked_at=booked_at, **d)


def history_frame(n=30, account="acc1", start="2018-01-01", seed=0):
    rng = np.random.default_rng(seed)
    rows = []
    t0 = pd.Timestamp(start)
    for i in range(n):
        t = t0 + pd.Timedelta(hours=float(i * 20 + rng.integers(0, 5)))
        rows.append(bk(f"h{i:03d}", account, t.isoformat(), weight_kg=float(1 + rng.random()),
                       carrier_cost=float(14 + 3 * rng.random()), dest_uf=["RJ", "MG"][i % 2]))
    return rows
