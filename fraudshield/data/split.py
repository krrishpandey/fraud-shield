"""Time-based splits (DESIGN section 9). End dates are exclusive (half-open intervals)."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class Split:
    name: str
    start: pd.Timestamp
    end: pd.Timestamp  # exclusive


SPLITS: tuple[Split, ...] = (
    Split("warmup", pd.Timestamp("2016-01-01"), pd.Timestamp("2017-06-01")),
    Split("train", pd.Timestamp("2017-06-01"), pd.Timestamp("2018-02-01")),
    Split("cal", pd.Timestamp("2018-02-15"), pd.Timestamp("2018-05-01")),
    Split("test", pd.Timestamp("2018-05-15"), pd.Timestamp("2018-09-01")),
)
SPLIT_BY_NAME = {s.name: s for s in SPLITS}
EMBARGO_DAYS = 14


def assign_split(ts: pd.Series) -> pd.Series:
    """Map timestamps to warmup/train/cal/test; gaps between splits are 'embargo', later is 'after'."""
    ts = pd.to_datetime(ts)
    out = np.full(len(ts), "embargo", dtype=object)
    out[(ts >= SPLITS[-1].end).to_numpy()] = "after"
    out[(ts < SPLITS[0].start).to_numpy()] = "warmup"
    for s in SPLITS:
        out[((ts >= s.start) & (ts < s.end)).to_numpy()] = s.name
    return pd.Series(out, index=ts.index, name="split")
