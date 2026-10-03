import pandas as pd

from fraudshield.data.split import SPLITS, assign_split


def test_assign_split_boundaries():
    ts = pd.to_datetime(pd.Series([
        "2016-10-01", "2017-05-31 23:59", "2017-06-01 00:00", "2018-01-31 23:00",
        "2018-02-05", "2018-02-15", "2018-04-30 12:00", "2018-05-10",
        "2018-05-15", "2018-08-31 22:00", "2018-09-02",
    ]), format="mixed")
    assert list(assign_split(ts)) == [
        "warmup", "warmup", "train", "train", "embargo", "cal", "cal", "embargo",
        "test", "test", "after",
    ]


def test_splits_are_ordered_and_disjoint():
    names = [s.name for s in SPLITS]
    assert names == ["warmup", "train", "cal", "test"]
    for a, b in zip(SPLITS, SPLITS[1:]):
        assert a.end <= b.start
