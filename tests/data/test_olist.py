import math
from pathlib import Path

import pandas as pd
import pytest

from fraudshield.data.olist import (
    CATEGORY_VOCAB,
    build_bookings,
    haversine_km,
    load_raw,
    map_category,
)


def tiny_raw():
    orders = pd.DataFrame({
        "order_id": ["o1", "o2"],
        "customer_id": ["c1", "c2"],
        "order_status": ["delivered", "delivered"],
        "order_purchase_timestamp": ["2017-07-01 10:00:00", "2017-07-02 09:00:00"],
        "order_approved_at": ["2017-07-01 10:30:00", None],
        "order_delivered_carrier_date": ["2017-07-03 12:00:00", "2017-07-04 08:00:00"],
        "order_delivered_customer_date": ["2017-07-10 12:00:00", None],
        "order_estimated_delivery_date": ["2017-07-21 00:00:00", "2017-07-05 00:00:00"],
    })
    items = pd.DataFrame({
        "order_id": ["o1", "o1", "o1", "o2"],
        "order_item_id": [1, 2, 3, 1],
        "product_id": ["p1", "p2", "p1", "p3"],
        "seller_id": ["sA", "sA", "sB", "sB"],
        "shipping_limit_date": ["2017-07-05"] * 4,
        "price": [100.0, 50.0, 100.0, 20.0],
        "freight_value": [10.0, 5.0, 12.0, 8.0],
    })
    products = pd.DataFrame({
        "product_id": ["p1", "p2", "p3"],
        "product_category_name": ["telefonia", "moveis_sala", None],
        "product_weight_g": [500.0, 1500.0, 200.0],
        "product_length_cm": [20.0, 40.0, 16.0],
        "product_height_cm": [5.0, 10.0, 2.0],
        "product_width_cm": [10.0, 30.0, 11.0],
    })
    sellers = pd.DataFrame({
        "seller_id": ["sA", "sB"],
        "seller_zip_code_prefix": [1234, 80010],
        "seller_city": ["sao paulo", "curitiba"],
        "seller_state": ["SP", "PR"],
    })
    customers = pd.DataFrame({
        "customer_id": ["c1", "c2"],
        "customer_unique_id": ["u1", "u2"],
        "customer_zip_code_prefix": [20000, 1234],
        "customer_city": ["rio", "sao paulo"],
        "customer_state": ["RJ", "SP"],
    })
    geolocation = pd.DataFrame({
        "geolocation_zip_code_prefix": [1234, 1234, 80010, 20000],
        "geolocation_lat": [-23.5, -23.7, -25.4, -22.9],
        "geolocation_lng": [-46.6, -46.6, -49.3, -43.2],
        "geolocation_city": ["sp", "sp", "cwb", "rio"],
        "geolocation_state": ["SP", "SP", "PR", "RJ"],
    })
    return dict(orders=orders, order_items=items, products=products, sellers=sellers,
                customers=customers, geolocation=geolocation)


def test_one_booking_per_order_seller_pair():
    b = build_bookings(tiny_raw())
    assert len(b) == 3
    assert set(zip(b.order_id, b.account_id)) == {("o1", "sA"), ("o1", "sB"), ("o2", "sB")}
    assert b.booking_id.is_unique


def test_aggregates_items_into_a_parcel():
    b = build_bookings(tiny_raw()).set_index(["order_id", "account_id"])
    r = b.loc[("o1", "sA")]
    assert r.weight_kg == pytest.approx(2.0)          # summed
    assert r.length_cm == 40 and r.width_cm == 30    # max footprint
    assert r.height_cm == 15                          # stacked heights
    assert r.carrier_cost == pytest.approx(15.0)      # sum freight
    assert r.declared_value == pytest.approx(150.0)   # sum price
    assert r.n_items == 2
    assert r.category == "phones"                     # category of the most expensive item


def test_mapping_of_accounts_places_and_times():
    b = build_bookings(tiny_raw()).set_index(["order_id", "account_id"])
    r = b.loc[("o1", "sA")]
    assert r.sender_id == "sA"
    assert r.origin_zip5 == "01234" and r.origin_zip3 == "012" and r.origin_uf == "SP"
    assert r.dest_zip3 == "200" and r.dest_uf == "RJ" and r.consignee_id == "u1"
    assert r.booked_at == pd.Timestamp("2017-07-01 10:30:00")
    assert r.first_scan_at == pd.Timestamp("2017-07-03 12:00:00")
    # approved_at missing -> purchase timestamp
    assert b.loc[("o2", "sB")].booked_at == pd.Timestamp("2017-07-02 09:00:00")
    # SP centroid is the mean of the two geolocation rows
    expect = haversine_km(-23.6, -46.6, -22.9, -43.2)
    assert r.distance_km == pytest.approx(expect, rel=1e-6)


def test_haversine_known_distance():
    # Sao Paulo to Rio de Janeiro is about 360 km
    assert 340 < haversine_km(-23.55, -46.63, -22.91, -43.17) < 380
    assert haversine_km(0, 0, 0, 0) == 0


def test_category_vocab_is_closed():
    assert map_category("telefonia") == "phones"
    assert map_category("moveis_sala") == "furniture"
    assert map_category(None) == "other"
    assert map_category("something_new") == "other"
    assert set(CATEGORY_VOCAB) >= {"phones", "electronics", "computers", "other"}


def test_service_is_standard_or_express():
    b = build_bookings(tiny_raw())
    assert set(b.service) <= {"standard", "express"}


RAW = Path(__file__).resolve().parents[2] / "data" / "raw"


@pytest.mark.slow
@pytest.mark.skipif(not (RAW / "olist_orders_dataset.csv.gz").exists(), reason="raw Olist data missing")
def test_real_olist_counts():
    b = build_bookings(load_raw(RAW))
    assert len(b) == 100_010
    assert b.account_id.nunique() == 3_095
    assert b.booked_at.notna().all()
    assert b.origin_uf.notna().all() and b.dest_uf.notna().all()
    assert (b.weight_kg > 0).mean() > 0.999
    assert 0.15 < (b.service == "express").mean() < 0.25
    assert b.distance_km.notna().mean() > 0.99
    assert set(b.category) <= set(CATEGORY_VOCAB)
