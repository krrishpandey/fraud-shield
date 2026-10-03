"""Shared test fixtures for the decision service (no data artifacts needed)."""
from dataclasses import asdict

from fraudshield.contracts import Booking

DEMO = dict(
    booking_id="demo-takeover-01", account_id="acc_7c3e", booked_at="2018-06-14T02:41:00", channel="api",
    login_device_age_days=0.0, payment_method="account_billing", sender_id="snd_91aa", origin_uf="PR",
    origin_zip3="806", dest_uf="AM", dest_zip3="690", consignee_id="cns_4410", weight_kg=9.8, length_cm=50,
    width_cm=40, height_cm=35, service="express", category="electronics", declared_value=1450.0,
    carrier_cost=212.6, owner_contact_age_days=412.0, meta={"scenario": "T1"},
)


def make_booking(**kw) -> Booking:
    d = dict(DEMO)
    d.update(kw)
    return Booking(**d)


def booking_json(**kw) -> dict:
    return asdict(make_booking(**kw))
