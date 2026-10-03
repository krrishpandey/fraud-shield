import numpy as np
import pandas as pd

from fraudshield.data.dataset import assemble, hard_negative_slices
from fraudshield.data.inject import LABEL_COLUMNS
from fraudshield.features import featurize_frame
from tests.data.test_inject import world  # noqa: F401  (module fixture)


def test_assemble_labels_real_rows_and_keeps_injected(world):  # noqa: F811
    real, pool, ctx, res = world
    legit_changes = pd.DataFrame(columns=["account_id", "kind", "at", "announced_at", "announced"])
    df, changes, confirmed = assemble(real, res, legit_changes)
    assert len(df) == len(real) + len(res.rows)
    for c in LABEL_COLUMNS:
        assert c in df.columns
    r = df[~df.is_injected]
    assert (~r.is_fraud).all() and (r.typology == "none").all() and (r.risk_level == 0).all()
    assert df.booking_id.is_unique
    assert df.booked_at.is_monotonic_increasing
    assert confirmed == res.confirmed


def test_hard_negative_slices(world):  # noqa: F811
    real, pool, ctx, res = world
    legit_changes = pd.DataFrame(columns=["account_id", "kind", "at", "announced_at", "announced"])
    df, changes, confirmed = assemble(real, res, legit_changes)
    feats = featurize_frame(df, changes=changes, confirmed=confirmed)
    hn = hard_negative_slices(df, feats)
    assert len(hn) == len(df)
    assert not (hn.any(axis=1) & df.is_fraud.to_numpy()).any()
    assert hn.hn_injected.sum() == df.typology.str.startswith("HN").sum()
    assert hn.hn_new_state.sum() > 0
