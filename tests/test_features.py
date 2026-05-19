import pandas as pd

from src.features import add_cyclical_encoding, add_spatial_features


def test_cyclical_encoding_adds_four_columns():
    df = pd.DataFrame({"day_of_week": [0, 3, 6], "month": [1, 6, 12]})
    out = add_cyclical_encoding(df)
    assert {"day_of_week_sin", "day_of_week_cos",
            "month_sin", "month_cos"}.issubset(out.columns)
    assert len(out) == len(df)


def test_spatial_features_join_attaches_coords():
    df = pd.DataFrame({"site_id": ["1", "2"], "count": [100, 200]})
    sites = pd.DataFrame({
        "site_id": ["1", "2"],
        "lat": [50.88, 51.04],
        "lon": [4.70, 3.72],
    })
    out = add_spatial_features(df, sites)
    assert "lat" in out.columns
    assert "lon" in out.columns
    assert len(out) == 2
