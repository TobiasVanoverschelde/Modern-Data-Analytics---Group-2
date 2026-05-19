import numpy as np
import pandas as pd
import holidays

def add_cyclical_encoding(df):
    df = df.copy()
    df["day_of_week_sin"] = np.sin(2 * np.pi * df["day_of_week"] / 7)
    df["day_of_week_cos"] = np.cos(2 * np.pi * df["day_of_week"] / 7)
    df["month_sin"] = np.sin(2 * np.pi * df["month"] / 12)
    df["month_cos"] = np.cos(2 * np.pi * df["month"] / 12)
    return df


COVID_PERIODS = {
    "pre_covid":       ("2019-01-01", "2020-03-13"),
    "first_lockdown":  ("2020-03-14", "2020-05-03"),
    "summer":          ("2020-05-04", "2020-10-18"),
    "second_lockdown": ("2020-10-19", "2021-05-08"),
    "post_covid":      ("2022-03-07", "2099-12-31"),
}


def add_calendar_features(df):
    df = df.copy()
    ts = df["timestamp"]
    df["hour"] = ts.dt.hour.astype("int8")
    df["day_of_week"] = ts.dt.dayofweek.astype("int8")
    df["month"] = ts.dt.month.astype("int8")
    df["year"] = ts.dt.year.astype("int16")
    df["is_weekend"] = df["day_of_week"] >= 5
    df["is_morning_rush"] = df["hour"].between(7, 9)
    df["is_evening_rush"] = df["hour"].between(16, 18)
    return df


def add_covid_period(df):
    df = df.copy()
    df["covid_period"] = "other"
    for label, (start, end) in COVID_PERIODS.items():
        mask = df["timestamp"].between(start, end)
        df.loc[mask, "covid_period"] = label
    return df


def add_holidays(df):
    years = df["timestamp"].dt.year.unique()
    be_holidays = holidays.country_holidays("BE", years=years)
    df = df.copy()
    df["is_holiday"] = df["timestamp"].dt.date.map(lambda d: d in be_holidays)
    return df


def add_spatial_features(df, sites_df):
    """Join WGS84 lat/lon per site so the model can learn geographic proximity
    instead of relying solely on the categorical gemeente encoding."""
    df = df.copy()
    df["site_id"] = df["site_id"].astype(str)
    coords = sites_df[["site_id", "lat", "lon"]].copy()
    coords["site_id"] = coords["site_id"].astype(str)
    return df.merge(coords, on="site_id", how="left")