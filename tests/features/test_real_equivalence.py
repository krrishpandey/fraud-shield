from pathlib import Path

import pandas as pd
import pytest

from fraudshield.features import FEATURES, featurize
from tests.features.test_store_featurize import _same

PROC = Path(__file__).resolve().parents[2] / "data" / "processed"


@pytest.mark.slow
@pytest.mark.skipif(not (PROC / "feature_store_demo.parquet").exists(), reason="processed data not built")
def test_offline_seed0_features_equal_online_reference_on_real_sample():
    from fraudshield.data.demo import load_demo_store
    store = load_demo_store()
    offline = pd.read_parquet(PROC / "features" / "seed_0.parquet").set_index("booking_id")
    frame = store.frame
    sample = frame[frame.booked_at >= "2017-06-01"].sample(150, random_state=1)
    inj = frame[frame.booking_id.str.len() > 0].merge(offline[offline.is_injected][[]], left_on="booking_id", right_index=True)
    sample = pd.concat([sample, inj.sample(50, random_state=2)])
    for _, r in sample.iterrows():
        ref = featurize(store.booking_from_row(r), store).values
        assert _same(ref, offline.loc[r.booking_id].to_dict(), keys=FEATURES.keys()), r.booking_id
