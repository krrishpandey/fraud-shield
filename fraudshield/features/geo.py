"""Zip3 and UF centroids (built from the real Olist geolocation table by scripts/build_dataset.py and
shipped next to this module) and the distance used by features."""
from __future__ import annotations

from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd

CENTROIDS_PATH = Path(__file__).with_name("zip3_centroids.csv")


@lru_cache(maxsize=1)
def _table() -> tuple[dict, dict]:
    if not CENTROIDS_PATH.exists():
        return {}, {}
    df = pd.read_csv(CENTROIDS_PATH, dtype={"key": str})
    z = {r.key: (r.lat, r.lng) for r in df[df.kind == "zip3"].itertuples()}
    u = {r.key: (r.lat, r.lng) for r in df[df.kind == "uf"].itertuples()}
    return z, u


def latlng(zip3: str, uf: str) -> tuple[float, float] | None:
    z, u = _table()
    return z.get(zip3) or u.get(uf)


@lru_cache(maxsize=200_000)
def dist_km(o_zip3: str, o_uf: str, d_zip3: str, d_uf: str) -> float:
    a, b = latlng(o_zip3, o_uf), latlng(d_zip3, d_uf)
    if a is None or b is None:
        return float("nan")
    lat1, lng1, lat2, lng2 = map(np.radians, (a[0], a[1], b[0], b[1]))
    h = np.sin((lat2 - lat1) / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin((lng2 - lng1) / 2) ** 2
    return float(2 * 6371.0 * np.arcsin(np.sqrt(min(1.0, h))))


def write_centroids(zip3: pd.DataFrame, uf: pd.DataFrame, path: Path = CENTROIDS_PATH) -> None:
    """zip3 / uf: frames indexed by key with lat, lng columns."""
    out = pd.concat([zip3.assign(kind="zip3"), uf.assign(kind="uf")]).rename_axis("key").reset_index()
    out[["kind", "key", "lat", "lng"]].round(5).to_csv(path, index=False)
    _table.cache_clear()
    dist_km.cache_clear()
