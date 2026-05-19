from pathlib import Path

import pandas as pd


# =========================================================
# PATHS
# =========================================================

PROCESSED_DIR = Path("data/processed")
PRECOMPUTED_DIR = Path("data/precomputed")

PRECOMPUTED_DIR.mkdir(
    parents=True,
    exist_ok=True
)


# =========================================================
# LOAD DATA
# =========================================================

df = pd.read_csv(
    PROCESSED_DIR / "municipality_daily_counts.csv"
)

coords = pd.read_csv(
    PROCESSED_DIR / "municipality_coordinates.csv"
)

df["datetime"] = pd.to_datetime(
    df["datetime"],
    errors="coerce"
)

df["datetime"] = df["datetime"].dt.date

df = df.dropna(
    subset=["datetime"]
)

# =========================================================
# TEMPORAL FEATURES
# =========================================================

df["weekday"] = pd.to_datetime(
    df["datetime"]
).dt.day_name()

df["month"] = pd.to_datetime(
    df["datetime"]
).dt.month_name()

df["year"] = pd.to_datetime(
    df["datetime"]
).dt.year


# =========================================================
# WEEKDAY SUMMARY
# =========================================================

weekday_summary = (

    df.groupby(
        ["municipality", "weekday"],
        as_index=False
    )["daily_count"]
    .mean()

)

weekday_summary.to_csv(
    PRECOMPUTED_DIR / "weekday_summary.csv",
    index=False
)


# =========================================================
# MONTHLY SUMMARY
# =========================================================

monthly_summary = (

    df.groupby(
        ["municipality", "month"],
        as_index=False
    )["daily_count"]
    .mean()

)

monthly_summary.to_csv(
    PRECOMPUTED_DIR / "monthly_summary.csv",
    index=False
)


# =========================================================
# YEARLY SUMMARY
# =========================================================

yearly_summary = (

    df.groupby(
        ["municipality", "year"],
        as_index=False
    )["daily_count"]
    .mean()

)

yearly_summary.to_csv(
    PRECOMPUTED_DIR / "yearly_summary.csv",
    index=False
)

# =====================================================
# LOAD WEATHER DATA
# =====================================================

weather = pd.read_csv(
    PROCESSED_DIR / "weather_all.csv"
)

weather["date"] = pd.to_datetime(
    weather["date"],
    errors="coerce"
).dt.date

df["datetime"] = pd.to_datetime(
    df["datetime"],
    errors="coerce"
).dt.date


# =====================================================
# MERGE WEATHER
# =====================================================

print(df.head())
print(df.columns)

print(weather.head())
print(weather.columns)

print(df["datetime"].dtype)
print(weather["date"].dtype)

df = df.merge(
    weather,
    left_on=["municipality", "datetime"],
    right_on=["municipality", "date"],
    how="left"
)

# =====================================================
# FIX WEATHER COLUMN NAMES
# =====================================================

df = df.rename(columns={

    "temperature_y": "temperature",
    "precipitation_y": "precipitation",
    "wind_speed_y": "wind_speed",
    "sunshine_duration_y": "sunshine_duration",
    "precipitation_hours_y": "precipitation_hours"

})

if "municipality_x" in df.columns:

    df["municipality"] = df["municipality_x"]

elif "municipality" not in df.columns:

    raise ValueError("municipality column missing after weather merge")

df = df.drop(
    columns=[
        "municipality_x",
        "municipality_y"
    ],
    errors="ignore"
)

weather_merged = df.copy()

weather_merged.to_csv(
    PRECOMPUTED_DIR / "weather_merged.csv",
    index=False
)

print(df.shape)
print(df.head())

# =========================================================
# WEATHER CORRELATIONS
# =========================================================

weather_variables = [

    "temperature",
    "precipitation",
    "wind_speed",
    "sunshine_duration",
    "precipitation_hours"

]

weather_rows = []

for municipality in df["municipality"].dropna().unique():

    sub = df[
        df["municipality"] == municipality
    ]

    for variable in weather_variables:

        if variable not in sub.columns:
            continue

        sub[variable] = pd.to_numeric(
            sub[variable],
            errors="coerce"
        )

        sub["daily_count"] = pd.to_numeric(
            sub["daily_count"],
            errors="coerce"
        )

        sub = sub.dropna(
            subset=[variable, "daily_count"]
        )

        if len(sub) < 2:
            continue

        corr = sub["daily_count"].corr(
            sub[variable]
        )

        weather_rows.append({

            "municipality": municipality,

            "weather_variable": variable,

            "correlation": corr

        })

weather_corr = pd.DataFrame(weather_rows)

weather_corr.to_csv(
    PRECOMPUTED_DIR / "weather_correlations.csv",
    index=False
)


# =========================================================
# HOURLY DATA
# =========================================================

hourly_path = PROCESSED_DIR / "hourly_counts.csv"

if hourly_path.exists():

    hourly = pd.read_csv(hourly_path)

    if "datetime" in hourly.columns:

        hourly["datetime"] = pd.to_datetime(
            hourly["datetime"],
            errors="coerce"
        )

        hourly["hour"] = (
            hourly["datetime"].dt.hour
        )

        hourly["weekday"] = (
            hourly["datetime"].dt.day_name()
        )

    elif "hour" not in hourly.columns:

        raise ValueError(
            "No hour or datetime column found in hourly_counts.csv"
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


# =========================================================
# MUNICIPALITY KPI TABLE
# =========================================================

kpi_rows = []

for municipality in df["municipality"].dropna().unique():

    sub = df[
        df["municipality"] == municipality
    ]

    peak = sub.loc[
        sub["daily_count"].idxmax()
    ]

    kpi_rows.append({

        "municipality": municipality,

        "avg_daily": sub["daily_count"].mean(),

        "total_cyclists": sub["daily_count"].sum(),

        "peak_day": peak["datetime"],

        "peak_count": peak["daily_count"],

        "start_date": sub["datetime"].min(),

        "end_date": sub["datetime"].max(),

        "active_days": sub["datetime"].nunique()

    })

municipality_kpis = pd.DataFrame(
    kpi_rows
)

municipality_kpis.to_csv(

    PRECOMPUTED_DIR / "municipality_kpis.csv",

    index=False
)


# =========================================================
# SPATIAL SUMMARY
# =========================================================

spatial_summary = (

    df.groupby(
        "municipality",
        as_index=False
    )["daily_count"]
    .sum()
    .rename(columns={
        "daily_count": "total_cyclists"
    })

)

spatial_summary = spatial_summary.merge(

    coords,

    on="municipality",

    how="left"
)

spatial_summary.to_csv(

    PRECOMPUTED_DIR / "spatial_summary.csv",

    index=False
)


print("✅ Precomputed tables successfully created.")
