from pathlib import Path

import pandas as pd


# =====================================================
# PATHS
# =====================================================

RAW_DIR = Path("data/raw")

PROCESSED_DIR = Path("data/processed")

PROCESSED_DIR.mkdir(
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

all_daily = []


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

    # =================================================
    # KEEP ONLY CYCLISTS
    # =================================================

    df = df[
        df["type"] == "FIETSERS"
    ]

    # =================================================
    # DATETIME
    # =================================================

    df["start_time"] = pd.to_datetime(
        df["start_time"],
        errors="coerce"
    )

    df = df.dropna(
        subset=["start_time"]
    )

    # =================================================
    # EXTRACT DATE
    # =================================================

    df["date"] = (
        df["start_time"]
        .dt.date
    )

    # =================================================
    # MERGE MUNICIPALITY
    # =================================================

    df = df.merge(

        richtingen,

        on="site_id",

        how="left"
    )

    # =================================================
    # KEEP VALID MUNICIPALITIES
    # =================================================

    df = df.dropna(
        subset=["municipality"]
    )

    all_daily.append(df)


# =====================================================
# COMBINE ALL FILES
# =====================================================

daily = pd.concat(
    all_daily,
    ignore_index=True
)


# =====================================================
# DAILY MUNICIPALITY COUNTS
# =====================================================

municipality_daily_counts = (

    daily.groupby(
        ["municipality", "date"],
        as_index=False
    )["count"]
    .sum()
    .rename(columns={
        "count": "daily_count",
        "date": "datetime"
    })

)

municipality_daily_counts.to_csv(

    PROCESSED_DIR / "municipality_daily_counts.csv",

    index=False
)


print(
    "✅ municipality_daily_counts.csv rebuilt successfully."
)
