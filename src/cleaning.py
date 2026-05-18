import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest

RANDOM_STATE = 42


def clean_counts(counts):
    df = counts.copy()

    # Filter to cyclists only
    df = df[df["voertuig_type"] == "FIETSERS"]

    # Parse datetimes
    df["van"] = pd.to_datetime(df["van"], errors="coerce")
    df = df.dropna(subset=["van", "aantal"])

    # Aggregate 15-min to hourly per site + direction
    df["timestamp"] = df["van"].dt.floor("h")
    df = (
        df.groupby(["site_id", "richting", "timestamp"], as_index=False)["aantal"]
          .sum()
          .rename(columns={"aantal": "count"})
    )
    df["count"] = df["count"].astype("int32")
    df["site_id"] = df["site_id"].astype(str)
    return df


def merge_with_sites(counts, sites):
    merged = counts.merge(sites, on="site_id", how="inner", validate="many_to_one")
    n_dropped = len(counts) - len(merged)
    if n_dropped:
        print(f"Dropped {n_dropped:,} rows with unknown site_id")
    return merged


def flag_outliers(df, contamination=0.01):
    df = df.copy()
    iso = IsolationForest(contamination=contamination, random_state=RANDOM_STATE, n_jobs=-1)
    df["is_outlier"] = iso.fit_predict(np.log1p(df[["count"]])) == -1
    print(f"Flagged {df['is_outlier'].sum():,} outliers ({df['is_outlier'].mean():.2%})")
    return df