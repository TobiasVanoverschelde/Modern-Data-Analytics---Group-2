import pandas as pd
from pathlib import Path


class CyclingDataLoader:

    def __init__(self):
        self.raw_dir = Path("data/raw")

    def load_monthly_counts(self):

        files = sorted(
            self.raw_dir.glob("data-*.csv")
        )

        aggregated_frames = []

        for file in files:

            print(f"Loading {file.name}")

            df = pd.read_csv(
                file,
                header=None,
                names=[
                    "site_id",
                    "direction_id",
                    "type",
                    "start_time",
                    "end_time",
                    "count"
                ],
                usecols=[
                    0,
                    1,
                    2,
                    3,
                    5
                ],
                low_memory=True
            )

            df["site_id"] = df["site_id"].astype(str)

            df["start_time"] = pd.to_datetime(
                df["start_time"],
                errors="coerce"
            )

            df["count"] = pd.to_numeric(
                df["count"],
                errors="coerce"
            )

            df = df.dropna(
                subset=[
                    "start_time",
                    "count"
                ]
            )

            df = df[
                df["count"] >= 0
            ]

            df["start_time"] = (
                df["start_time"]
                .dt.floor("h")
            )

            hourly_site = (
                df.groupby([
                    "start_time",
                    "site_id"
                ])["count"]
                .sum()
                .reset_index()
            )

            aggregated_frames.append(
                hourly_site
            )

        counts = pd.concat(
            aggregated_frames,
            ignore_index=True
        )

        return counts

    def load_sites(self):

        return pd.read_csv(
            self.raw_dir / "sites.csv",
            header=None,
            names=[
                "site_id",
                "site_uuid",
                "longitude",
                "latitude",
                "road",
                "direction_raw",
                "site_name",
                "counter_id",
                "municipality",
                "interval_minutes",
                "install_date"
            ],
            low_memory=False
        )

    def load_directions(self):

        return pd.read_csv(
            self.raw_dir / "richtingen.csv",
            header=None,
            names=[
                "site_id",
                "direction_id",
                "direction_name"
            ],
            low_memory=False
        )

    def load_all(self):

        counts = self.load_monthly_counts()
        sites = self.load_sites()
        directions = self.load_directions()

        return counts, sites, directions
