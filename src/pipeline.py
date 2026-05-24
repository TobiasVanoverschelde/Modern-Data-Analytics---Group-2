"""
End-to-end data pipeline as an importable class.

Replaces the manual orchestration in notebooks/01_data_preparation.ipynb:
download → load → clean → features → weather → daily aggregation → save.

Runnable headless as:
    python -m src.pipeline
"""

from pathlib import Path

import pandas as pd
import requests

from src.loaders import COLUMN_NAMES, load_sites
from src.cleaning import clean_counts, merge_with_sites
from src.features import (
    add_calendar_features, add_holidays, add_covid_period, add_cyclical_encoding,
)
from src.weather import fetch_open_meteo


AWV_BASE = "https://opendata.apps.mow.vlaanderen.be/fietstellingen/"

# One central Flanders point 
# weather would mean ~75× the API calls; the spatial variation is small enough that
# one central pull is a reasonable compromise.
FLANDERS_LAT, FLANDERS_LON = 50.88, 4.70

DAILY_CONSTANT_COLS = [
    "day_of_week", "month", "year", "is_weekend", "is_holiday", "covid_period",
    "day_of_week_sin", "day_of_week_cos", "month_sin", "month_cos",
    "lat", "lon",
]


class CyclingDataPipeline:

    def __init__(
        self,
        raw_dir: str = "data/raw",
        processed_dir: str = "data/processed",
        start_date: str = "2019-08-01",
        end_date: str = "2026-04-30",
    ):
        self.raw_dir = Path(raw_dir)
        self.processed_dir = Path(processed_dir)
        self.raw_dir.mkdir(parents=True, exist_ok=True)
        self.processed_dir.mkdir(parents=True, exist_ok=True)
        self.start_date = start_date
        self.end_date = end_date

    # Download monthly count CSVs + sites + directions. Skips files that exist.
    def download(self) -> None:
        files = self._monthly_filenames() + ["sites.csv", "richtingen.csv"]
        for name in files:
            path = self.raw_dir / name
            if path.exists():
                continue
            r = requests.get(AWV_BASE + name, timeout=60)
            if r.status_code == 200 and "html" not in r.headers.get("Content-Type", ""):
                path.write_bytes(r.content)
                print(f"  OK   {name} ({len(r.content) / 1024:,.0f} KB)")
            else:
                print(f"  SKIP {name}")

    def _monthly_filenames(self) -> list[str]:
        start = pd.Timestamp(self.start_date).to_period("M")
        end = pd.Timestamp(self.end_date).to_period("M")
        return [f"data-{p.year}-{p.month:02d}.csv"
                for p in pd.period_range(start, end, freq="M")]

    # Stream-process one monthly CSV at a time. Each file is cleaned + aggregated
    # to hourly inside the loop, so we never hold all ~42M raw rows in memory.
    # Then we concat the (much smaller) hourly frames and join site metadata.
    def clean(self, sites: pd.DataFrame) -> pd.DataFrame:
        hourly = []
        for f in sorted(self.raw_dir.glob("data-*.csv")):
            raw = pd.read_csv(f, sep=",", header=None,
                              names=COLUMN_NAMES, low_memory=False)
            hourly.append(clean_counts(raw))
            print(f"  cleaned {f.name}")
        return merge_with_sites(pd.concat(hourly, ignore_index=True), sites)

    # Calendar, holidays, COVID period, cyclical encoding.
    # Spatial features (lat/lon) already attached during clean().
    def add_features(self, df: pd.DataFrame) -> pd.DataFrame:
        df = add_calendar_features(df)
        df = add_holidays(df)
        df = add_covid_period(df)
        return add_cyclical_encoding(df)

    # Pull hourly weather once for central Flanders and merge on timestamp.
    def add_weather(self, df: pd.DataFrame) -> pd.DataFrame:
        weather = fetch_open_meteo(
            FLANDERS_LAT, FLANDERS_LON, self.start_date, self.end_date,
        )
        return df.merge(weather, on="timestamp", how="left")

    # Hourly → daily. count: sum, precip: sum, others: mean.
    # Calendar/spatial features are constant per day so taken via .first().
    def daily_aggregate(self, hourly: pd.DataFrame) -> pd.DataFrame:
        df = hourly.assign(date=hourly["timestamp"].dt.normalize())
        agg = {
            "count": "sum",
            "temperature_2m": "mean",
            "precipitation": "sum",
            "wind_speed_10m": "mean",
            "cloud_cover": "mean",
            **{col: "first" for col in DAILY_CONSTANT_COLS},
        }
        return (
            df.groupby(["date", "site_id", "gemeente"], as_index=False)
              .agg(agg)
              .dropna(subset=["temperature_2m", "lat", "lon"])
        )

    def save(self, hourly: pd.DataFrame, daily: pd.DataFrame) -> None:
        hourly.to_parquet(self.processed_dir / "cycling_features.parquet", index=False)
        daily.to_parquet(self.processed_dir / "daily_for_modeling.parquet", index=False)

    def run(self) -> tuple[pd.DataFrame, pd.DataFrame]:
        self.download()
        sites = load_sites(self.raw_dir)
        hourly = self.add_weather(self.add_features(self.clean(sites)))
        daily = self.daily_aggregate(hourly)
        self.save(hourly, daily)
        return hourly, daily


if __name__ == "__main__":
    hourly, daily = CyclingDataPipeline().run()
    print(f"\nHourly rows: {len(hourly):,}")
    print(f"Daily rows:  {len(daily):,}")
    print(f"Saved to data/processed/")
