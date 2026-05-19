import pandas as pd
from pathlib import Path


class Preprocessor:

    def __init__(self):

        self.data_dir = Path("data/raw")

    # =====================================================
    # LOAD RICHTINGEN
    # =====================================================

    def load_richtingen(self):

        richtingen = pd.read_csv(
            self.data_dir / "richtingen.csv",
            sep=";"
        )

        return richtingen

    # =====================================================
    # LOAD MONTHLY DATA
    # =====================================================

    def load_monthly_data(self):

        csv_files = sorted(
            Path(".").rglob("data-*.csv")
        )

        print("FOUND MONTHLY FILES:")

        for f in csv_files:
            print(f)

        frames = []

        for file in csv_files:

            print(f"Loading {file.name}")

            df = pd.read_csv(
                file,
                header=None
            )

            df.columns = [
                "site_id",
                "direction",
                "type",
                "start_time",
                "end_time",
                "count"
            ]

            frames.append(df)

        combined = pd.concat(
            frames,
            ignore_index=True
        )

        return combined

    # =====================================================
    # MERGE EVERYTHING
    # =====================================================

    def merge_datasets(self):

        cycling = self.load_monthly_data()

        richtingen = self.load_richtingen()

        df = cycling.merge(
            richtingen,
            on=["site_id", "direction"],
            how="left"
        )

        df["datetime"] = pd.to_datetime(
            df["start_time"],
            errors="coerce"
        )

        df["count"] = pd.to_numeric(
            df["count"],
            errors="coerce"
        ).fillna(0)

        return df

    # =====================================================
    # DAILY AGGREGATION
    # =====================================================

    def create_daily_aggregations(self, df):

        df["date"] = df["datetime"].dt.date

        daily = (
            df.groupby("date")["count"]
            .sum()
            .reset_index()
        )

        daily.columns = [
            "datetime",
            "daily_count"
        ]

        municipality_daily = (
            df.groupby([
                "municipality",
                "date"
            ])["count"]
            .sum()
            .reset_index()
        )

        municipality_daily.columns = [
            "municipality",
            "datetime",
            "daily_count"
        ]

        return daily, municipality_daily

    # =====================================================
    # SPATIAL ANALYSIS
    # =====================================================

    def create_spatial_analysis(self, df):

        spatial = (
            df.groupby([
                "municipality",
                "site_id"
            ])["count"]
            .sum()
            .reset_index()
        )

        spatial.columns = [
            "municipality",
            "site_id",
            "total_cyclists"
        ]

        return spatial

    # =====================================================
    # HOURLY ANALYSIS
    # =====================================================

    def create_hourly_analysis(self, df):

        hourly = df.copy()

        hourly["hour"] = (
            hourly["datetime"]
            .dt.hour
        )

        hourly["weekday"] = (
            hourly["datetime"]
            .dt.day_name()
        )

        result = (
            hourly.groupby([
                "weekday",
                "hour"
            ])["count"]
            .mean()
            .reset_index()
        )

        result.columns = [
            "weekday",
            "hour",
            "avg_cyclists"
        ]

        return result
