from pathlib import Path

import pandas as pd


# =====================================================
# PATHS
# =====================================================

RAW_DIR = Path("data/raw")

PROCESSED_DIR = Path("data/processed")

PRECOMPUTED_DIR = Path("data/precomputed")

PRECOMPUTED_DIR.mkdir(
    parents=True,
    exist_ok=True
)


# =====================================================
# LOAD MUNICIPALITY MAPPING
# =====================================================

richtingen = pd.read_csv(
    RAW_DIR / "richtingen.csv",
    sep=";"
)

richtingen = (
    richtingen[
        ["site_id", "municipality"]
    ]
    .drop_duplicates()
    .groupby("site_id", as_index=False)
    .first()
)


# =====================================================
# FIND MONTHLY RAW FILES
# =====================================================

monthly_files = list(
    RAW_DIR.glob("data-*.csv")
)

all_hourly = []


# =====================================================
# PROCESS FILES
# =====================================================

for file in monthly_files:

    print(f"Processing {file.name}")

    df = pd.read_csv(

        file,

        header=None,

        names=[

            "site_id",
            "direction",
            "type",
            "start_time",
            "end_time",
            "count"
        ]
    )

    # only cyclists
    df = df[
        df["type"] == "FIETSERS"
    ]

    # datetime
    df["start_time"] = pd.to_datetime(
        df["start_time"],
        errors="coerce"
    )

    # remove bad rows
    df = df.dropna(
        subset=["start_time"]
    )

    # extract hour + weekday
    df["hour"] = (
        df["start_time"].dt.hour
    )

    df["weekday"] = (
        df["start_time"].dt.day_name()
    )

    # merge municipality
    df = df.merge(

        richtingen,

        on="site_id",

        how="left"
    )

    # keep valid municipalities
    df = df.dropna(
        subset=["municipality"]
    )

    all_hourly.append(df)


# =====================================================
# COMBINE
# =====================================================

hourly = pd.concat(
    all_hourly,
    ignore_index=True
)


# =====================================================
# HOURLY CURVE SUMMARY
# =====================================================

hourly_curve_summary = (

    hourly.groupby(
        ["municipality", "hour"],
        as_index=False
    )["count"]
    .mean()
    .rename(columns={
        "count": "avg_cyclists"
    })

)

hourly_curve_summary.to_csv(

    PRECOMPUTED_DIR / "hourly_curve_summary.csv",

    index=False
)


# =====================================================
# HOURLY HEATMAP SUMMARY
# =====================================================

hourly_heatmap_summary = (

    hourly.groupby(
        ["municipality", "weekday", "hour"],
        as_index=False
    )["count"]
    .mean()
    .rename(columns={
        "count": "avg_cyclists"
    })

)

hourly_heatmap_summary.to_csv(

    PRECOMPUTED_DIR / "hourly_heatmap_summary.csv",

    index=False
)


print("✅ Hourly municipality precompute completed.")
