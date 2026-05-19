from pathlib import Path
import time

from src.preprocessing import Preprocessor
from src.weather_service import WeatherService
from src.geocoding import MunicipalityGeocoder

import pandas as pd

from sklearn.ensemble import RandomForestRegressor


# =========================================================
# PIPELINE FLAGS
# =========================================================

RUN_PREPROCESSING = False
RUN_WEATHER = True
RUN_MODEL = False


# =========================================================
# OUTPUT FOLDERS
# =========================================================

Path("data/processed").mkdir(
    parents=True,
    exist_ok=True
)

Path("data/weather").mkdir(
    parents=True,
    exist_ok=True
)


# =========================================================
# PREPROCESSING
# =========================================================

pre = Preprocessor()

raw = pre.merge_datasets()

daily, municipality_daily = (
    pre.create_daily_aggregations(raw)
)

spatial = pre.create_spatial_analysis(raw)

hourly = pre.create_hourly_analysis(raw)


# =========================================================
# GEOCODING
# =========================================================

municipalities = (

    municipality_daily["municipality"]
    .dropna()
    .unique()

)

geocoder = MunicipalityGeocoder()

coordinates = (

    geocoder.geocode_municipalities(
        municipalities
    )

)

coordinates.to_csv(
    "data/processed/municipality_coordinates.csv",
    index=False
)


# =========================================================
# MERGE COORDINATES INTO SPATIAL
# =========================================================

spatial = spatial.merge(

    coordinates,

    on="municipality",

    how="left"

)


# =========================================================
# WEATHER
# =========================================================

if RUN_WEATHER:

    weather_service = WeatherService()

    weather_frames = []

    for row in coordinates.itertuples():

        if pd.isna(row.latitude) or pd.isna(row.longitude):
            continue

        municipality = row.municipality

        print(f"\nDownloading weather for {municipality}")

        weather = weather_service.get_weather_data(

            municipality=municipality,

            latitude=row.latitude,
            longitude=row.longitude,

            start_date="2019-08-01",

            end_date="2026-05-16"

        )

        # =====================================================
        # STANDARDIZE WEATHER SCHEMA
        # =====================================================

        if "temperature_2m_mean" in weather.columns:

            weather["temperature"] = (
                weather["temperature_2m_mean"]
            )

        if "precipitation_sum" in weather.columns:

            weather["precipitation"] = (
                weather["precipitation_sum"]
            )

        if "wind_speed_10m_max" in weather.columns:

            weather["wind_speed"] = (
                weather["wind_speed_10m_max"]
            )

        weather = weather[[
            "date",
            "temperature",
            "precipitation",
            "wind_speed",
            "sunshine_duration",
            "precipitation_hours",
            "municipality"
        ]]

        weather_frames.append(weather)


    weather_all = pd.concat(

        weather_frames,

        ignore_index=True
    )

    weather_all = weather_all.drop_duplicates(
        subset=["municipality", "date"]
    )

    weather_all.to_csv(
        "data/processed/weather_all.csv",
        index=False
    )

else:

    print("Loading existing weather files...")

    weather_files = list(
        Path("data/weather").glob("*.csv")
    )

    weather_frames = []

    for file in weather_files:

        df = pd.read_csv(file)

        weather_frames.append(df)

    weather_all = pd.concat(

        weather_frames,

        ignore_index=True
    )


# =========================================================
# WEATHER COLUMN NORMALIZATION
# =========================================================

weather_all = weather_all.rename(columns={

    "temperature_2m_mean": "temperature",
    "precipitation_sum": "precipitation",
    "wind_speed_10m_max": "wind_speed"

})

# Keep only standardized columns

weather_all = weather_all[[
    "date",
    "municipality",
    "temperature",
    "precipitation",
    "wind_speed",
    "sunshine_duration",
    "precipitation_hours"
]]

# Remove duplicate rows

weather_all = weather_all.drop_duplicates()

# =========================================================
# DATE FIX
# =========================================================

municipality_daily["date"] = (

    pd.to_datetime(
        municipality_daily["datetime"]
    )
    .dt.date
    .astype(str)

)

weather_all["date"] = (
    weather_all["date"]
    .astype(str)
)


# =========================================================
# WEATHER MERGE
# =========================================================

municipality_daily = municipality_daily.merge(

    weather_all,

    on=[
        "municipality",
        "date"
    ],

    how="left"

)


# =========================================================
# FEATURE ENGINEERING
# =========================================================

municipality_daily = municipality_daily.sort_values(
    [
        "municipality",
        "datetime"
    ]
)

municipality_daily["previous_day_traffic"] = (

    municipality_daily
    .groupby("municipality")["daily_count"]
    .shift(1)

)

municipality_daily["traffic_one_week_earlier"] = (

    municipality_daily
    .groupby("municipality")["daily_count"]
    .shift(7)

)

municipality_daily["year"] = (

    pd.to_datetime(
        municipality_daily["datetime"]
    ).dt.year

)

municipality_daily["month"] = (

    pd.to_datetime(
        municipality_daily["datetime"]
    ).dt.month

)

municipality_daily["day_of_week"] = (

    pd.to_datetime(
        municipality_daily["datetime"]
    ).dt.dayofweek

)


# =========================================================
# DROP NANS
# =========================================================

municipality_daily = municipality_daily.dropna()


# =========================================================
# SAVE PROCESSED FILES
# =========================================================

daily.to_csv(
    "data/processed/daily_counts.csv",
    index=False
)

municipality_daily.to_csv(
    "data/processed/municipality_daily_counts.csv",
    index=False
)

spatial.to_csv(
    "data/processed/spatial_analysis.csv",
    index=False
)

hourly.to_csv(
    "data/processed/hourly_analysis.csv",
    index=False
)


# =========================================================
# MODEL
# =========================================================

if RUN_MODEL:

    features = [

        "year",
        "month",
        "day_of_week",

        "previous_day_traffic",
        "traffic_one_week_earlier",

        "temperature",
        "precipitation",
        "wind_speed",
        "sunshine_duration",
        "precipitation_hours"

    ]

    X = municipality_daily[features]

    y = municipality_daily["daily_count"]


    # =====================================================
    # CHRONOLOGICAL SPLIT
    # =====================================================

    split_index = int(
        len(X) * 0.8
    )

    X_train = X.iloc[:split_index]
    X_test = X.iloc[split_index:]

    y_train = y.iloc[:split_index]
    y_test = y.iloc[split_index:]


    # =====================================================
    # MODEL TRAINING
    # =====================================================

    model = RandomForestRegressor(

        n_estimators=100,

        random_state=42,

        n_jobs=-1

    )

    model.fit(
        X_train,
        y_train
    )

    predictions = model.predict(
        X_test
    )

    from sklearn.metrics import (
        mean_absolute_error,
        mean_squared_error,
        r2_score
    )

    mae = mean_absolute_error(
        y_test,
        predictions
    )

    mse = mean_squared_error(
        y_test,
        predictions
    )

    rmse = mse ** 0.5

    r2 = r2_score(
        y_test,
        predictions
    )

    print("\nMODEL PERFORMANCE")
    print("------------------")
    print(f"MAE  : {mae:.2f}")
    print(f"RMSE : {rmse:.2f}")
    print(f"R²   : {r2:.3f}")

    # =====================================================
    # SAVE PREDICTIONS
    # =====================================================

    prediction_df = pd.DataFrame({
        "actual": y_test,
        "predicted": predictions

    })

    prediction_df.to_csv(
        "data/processed/predictions.csv",
        index=False
    )


    # =====================================================
    # FEATURE IMPORTANCE
    # =====================================================

    importance = pd.DataFrame({

        "feature": features,
        "importance": model.feature_importances_

    })

    importance.to_csv(
        "data/processed/feature_importance.csv",
        index=False
    )

print("\nPipeline completed successfully.")
