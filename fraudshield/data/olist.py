"""Olist Brazilian E-Commerce -> carrier booking table.

One booking = one (order_id, seller_id) pair (DESIGN section 9). Mapping:
  account_id = sender_id = seller_id; booked_at = order_approved_at (fallback purchase ts);
  origin = seller zip prefix (5 digits) -> zip3 + UF; consignee = customer_unique_id; dest = customer zip;
  weight = sum of item weights; footprint = max item length and width; height = sum of item heights
  (items stacked in one box, a documented simplification); carrier_cost = sum freight_value (buyer-paid
  in reality, used as a carrier-cost proxy); declared_value = sum price; category = category of the most
  expensive item mapped to a closed vocabulary; first_scan_at = order_delivered_carrier_date.
  service proxy [ASSUMPTION]: express if the promised delivery window (estimated - purchase) is in the
  fastest 20% for its distance band.
"""
from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np
import pandas as pd

RAW_TABLES = {
    "orders": "olist_orders_dataset.csv.gz",
    "order_items": "olist_order_items_dataset.csv.gz",
    "products": "olist_products_dataset.csv.gz",
    "sellers": "olist_sellers_dataset.csv.gz",
    "customers": "olist_customers_dataset.csv.gz",
    "geolocation": "olist_geolocation_dataset.csv.gz",
}

CATEGORY_VOCAB: tuple[str, ...] = (
    "phones", "electronics", "computers", "watches_gifts", "fashion", "home", "furniture",
    "beauty_health", "sports_leisure", "toys_baby", "tools_auto", "books_media", "food_drink", "other",
)
HIGH_VALUE_CATEGORIES = frozenset({"phones", "electronics", "computers", "watches_gifts"})

_EXACT = {
    "telefonia": "phones", "telefonia_fixa": "phones",
    "eletronicos": "electronics", "audio": "electronics", "cine_foto": "electronics",
    "consoles_games": "electronics", "tablets_impressao_imagem": "electronics",
    "informatica_acessorios": "computers", "pcs": "computers", "pc_gamer": "computers",
    "relogios_presentes": "watches_gifts",
    "malas_acessorios": "fashion",
    "cama_mesa_banho": "home", "utilidades_domesticas": "home", "la_cuisine": "home",
    "climatizacao": "home", "eletroportateis": "home", "flores": "home",
    "beleza_saude": "beauty_health", "perfumaria": "beauty_health", "fraldas_higiene": "beauty_health",
    "esporte_lazer": "sports_leisure",
    "brinquedos": "toys_baby", "bebes": "toys_baby", "cool_stuff": "toys_baby",
    "automotivo": "tools_auto", "ferramentas_jardim": "tools_auto", "sinalizacao_e_seguranca": "tools_auto",
    "agro_industria_e_comercio": "tools_auto", "industria_comercio_e_negocios": "tools_auto",
    "cds_dvds_musicais": "books_media", "dvds_blu_ray": "books_media", "musica": "books_media",
    "instrumentos_musicais": "books_media", "papelaria": "books_media",
    "alimentos": "food_drink", "alimentos_bebidas": "food_drink", "bebidas": "food_drink",
}
_PREFIX = (
    ("fashion_", "fashion"), ("moveis_", "furniture"), ("casa_", "home"), ("eletrodomesticos", "home"),
    ("portateis_", "home"), ("artigos_de_", "home"), ("construcao_", "tools_auto"), ("livros_", "books_media"),
    ("artes", "books_media"),
)

DISTANCE_BANDS_KM = (0.0, 100.0, 300.0, 700.0, 1500.0, np.inf)
EXPRESS_QUANTILE = 0.20
BR_LAT = (-34.0, 6.0)
BR_LNG = (-74.0, -34.0)


def map_category(name: str | None) -> str:
    if name is None or (isinstance(name, float) and np.isnan(name)):
        return "other"
    if name in _EXACT:
        return _EXACT[name]
    for prefix, cat in _PREFIX:
        if name.startswith(prefix):
            return cat
    return "other"


def haversine_km(lat1, lng1, lat2, lng2):
    lat1, lng1, lat2, lng2 = (np.radians(np.asarray(x, dtype=float)) for x in (lat1, lng1, lat2, lng2))
    a = np.sin((lat2 - lat1) / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin((lng2 - lng1) / 2) ** 2
    d = 2 * 6371.0 * np.arcsin(np.sqrt(np.clip(a, 0, 1)))
    return float(d) if np.ndim(d) == 0 else d


def zip5(prefix) -> pd.Series:
    return pd.Series(prefix).astype(int).astype(str).str.zfill(5)


def load_raw(raw_dir: str | Path) -> dict[str, pd.DataFrame]:
    raw_dir = Path(raw_dir)
    return {k: pd.read_csv(raw_dir / f) for k, f in RAW_TABLES.items()}


def zip_centroids(geo: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """Centroids by zip5 prefix, zip3 and UF (points outside Brazil's bounding box dropped)."""
    g = geo[geo.geolocation_lat.between(*BR_LAT) & geo.geolocation_lng.between(*BR_LNG)].copy()
    g["zip5"] = zip5(g.geolocation_zip_code_prefix).to_numpy()
    g["zip3"] = g.zip5.str[:3]
    out = {}
    for key, col in (("zip5", "zip5"), ("zip3", "zip3"), ("uf", "geolocation_state")):
        c = g.groupby(col)[["geolocation_lat", "geolocation_lng"]].mean()
        c.columns = ["lat", "lng"]
        out[key] = c
    return out


def lookup_latlng(cent: dict[str, pd.DataFrame], z5: pd.Series, uf: pd.Series) -> tuple[np.ndarray, np.ndarray]:
    lat = z5.map(cent["zip5"].lat)
    lng = z5.map(cent["zip5"].lng)
    z3 = z5.str[:3]
    lat = lat.fillna(z3.map(cent["zip3"].lat)).fillna(uf.map(cent["uf"].lat))
    lng = lng.fillna(z3.map(cent["zip3"].lng)).fillna(uf.map(cent["uf"].lng))
    return lat.to_numpy(dtype=float), lng.to_numpy(dtype=float)


def _clean_products(products: pd.DataFrame) -> pd.DataFrame:
    p = products.copy()
    p["category"] = p.product_category_name.map(map_category)
    p.loc[p.product_weight_g <= 0, "product_weight_g"] = np.nan
    for col in ("product_weight_g", "product_length_cm", "product_height_cm", "product_width_cm"):
        p[col] = p[col].fillna(p.groupby("category")[col].transform("median")).fillna(p[col].median())
    return p


def express_flag(window_days: pd.Series, distance_km: pd.Series) -> pd.Series:
    """Service proxy: fastest EXPRESS_QUANTILE of promised windows within each distance band."""
    band = pd.cut(distance_km.fillna(distance_km.median()), DISTANCE_BANDS_KM, right=False, labels=False)
    cutoff = window_days.groupby(band).transform(lambda s: s.quantile(EXPRESS_QUANTILE))
    return (window_days <= cutoff).fillna(False)


def booking_id_for(order_id: str, seller_id: str) -> str:
    return "bk_" + hashlib.md5(f"{order_id}|{seller_id}".encode()).hexdigest()[:16]


def build_bookings(raw: dict[str, pd.DataFrame]) -> pd.DataFrame:
    orders = raw["orders"].copy()
    for c in ("order_purchase_timestamp", "order_approved_at", "order_delivered_carrier_date",
              "order_delivered_customer_date", "order_estimated_delivery_date"):
        orders[c] = pd.to_datetime(orders[c])
    products = _clean_products(raw["products"])

    it = raw["order_items"].merge(products, on="product_id", how="left")
    it["category"] = it["category"].fillna("other")
    it = it.sort_values(["order_id", "seller_id", "price"], ascending=[True, True, False])
    grp = it.groupby(["order_id", "seller_id"], sort=False)
    b = grp.agg(
        n_items=("order_item_id", "size"),
        weight_g=("product_weight_g", "sum"),
        length_cm=("product_length_cm", "max"),
        width_cm=("product_width_cm", "max"),
        height_cm=("product_height_cm", "sum"),
        carrier_cost=("freight_value", "sum"),
        declared_value=("price", "sum"),
        category=("category", "first"),
        top_product_id=("product_id", "first"),
    ).reset_index()
    b["weight_kg"] = b.pop("weight_g") / 1000.0

    b = b.merge(orders, on="order_id", how="left")
    b = b.merge(raw["customers"][["customer_id", "customer_unique_id", "customer_zip_code_prefix", "customer_state"]],
                on="customer_id", how="left")
    b = b.merge(raw["sellers"][["seller_id", "seller_zip_code_prefix", "seller_state"]], on="seller_id", how="left")

    out = pd.DataFrame({
        "booking_id": [booking_id_for(o, s) for o, s in zip(b.order_id, b.seller_id)],
        "order_id": b.order_id,
        "account_id": b.seller_id,
        "sender_id": b.seller_id,
        "booked_at": b.order_approved_at.fillna(b.order_purchase_timestamp),
        "purchase_at": b.order_purchase_timestamp,
        "first_scan_at": b.order_delivered_carrier_date,
        "delivered_at": b.order_delivered_customer_date,
        "estimated_at": b.order_estimated_delivery_date,
        "order_status": b.order_status,
        "origin_zip5": zip5(b.seller_zip_code_prefix).to_numpy(),
        "origin_uf": b.seller_state,
        "consignee_id": b.customer_unique_id,
        "dest_zip5": zip5(b.customer_zip_code_prefix).to_numpy(),
        "dest_uf": b.customer_state,
        "weight_kg": b.weight_kg,
        "length_cm": b.length_cm,
        "width_cm": b.width_cm,
        "height_cm": b.height_cm,
        "n_items": b.n_items,
        "category": b.category,
        "top_product_id": b.top_product_id,
        "declared_value": b.declared_value.round(2),
        "carrier_cost": b.carrier_cost.round(2),
    })
    out["origin_zip3"] = out.origin_zip5.str[:3]
    out["dest_zip3"] = out.dest_zip5.str[:3]

    cent = zip_centroids(raw["geolocation"])
    olat, olng = lookup_latlng(cent, out.origin_zip5, out.origin_uf)
    dlat, dlng = lookup_latlng(cent, out.dest_zip5, out.dest_uf)
    out["distance_km"] = haversine_km(olat, olng, dlat, dlng)

    window = (out.estimated_at - out.purchase_at).dt.total_seconds() / 86400.0
    out["promised_window_days"] = window
    out["service"] = np.where(express_flag(window, out.distance_km), "express", "standard")
    return out.sort_values(["booked_at", "booking_id"]).reset_index(drop=True)
