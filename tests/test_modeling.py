from datetime import datetime, timedelta

import pandas as pd

from src.modeling import time_aware_split


def test_time_aware_split_respects_cutoff():
    dates = [datetime(2024, 1, 1) + timedelta(days=i) for i in range(10)]
    df = pd.DataFrame({"date": dates, "count": range(10)})
    train, test = time_aware_split(df, cutoff_date="2024-01-06")
    assert train["date"].max() < pd.Timestamp("2024-01-06")
    assert test["date"].min() >= pd.Timestamp("2024-01-06")
    assert len(train) + len(test) == len(df)
