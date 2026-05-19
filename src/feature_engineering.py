import pandas as pd


class FeatureEngineer:

    def add_time_features(self, df):

        df = df.copy()

        df["datetime"] = pd.to_datetime(
            df["datetime"]
        )

        df["year"] = (
            df["datetime"]
            .dt.year
        )

        df["month"] = (
            df["datetime"]
            .dt.month
        )

        df["day_of_month"] = (
            df["datetime"]
            .dt.day
        )

        df["day_of_week"] = (
            df["datetime"]
            .dt.weekday
        )

        return df

    def add_lag_features(self, df):

        df = df.copy()

        df = df.sort_values([
            "municipality",
            "datetime"
        ])

        df["previous_day_traffic"] = (
            df.groupby("municipality")[
                "daily_count"
            ]
            .shift(1)
        )

        df["traffic_one_week_earlier"] = (
            df.groupby("municipality")[
                "daily_count"
            ]
            .shift(7)
        )

        df["previous_day_traffic"] = (
            df["previous_day_traffic"]
            .fillna(df["daily_count"])
        )

        df["traffic_one_week_earlier"] = (
            df["traffic_one_week_earlier"]
            .fillna(df["daily_count"])
        )

        return df
