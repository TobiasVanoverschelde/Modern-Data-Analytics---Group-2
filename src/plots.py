import pandas as pd
import plotly.express as px


def plot_daily_trend(df):
    grouped = df.groupby("datetime")["daily_count"].sum().reset_index()
    return px.line(grouped, x="datetime", y="daily_count", title="Total Daily Cycling Volume in Flanders")


def plot_monthly_seasonality(df):
    grouped = df.groupby("month")["daily_count"].mean().reset_index()
    return px.bar(grouped, x="month", y="daily_count", title="Average Cycling Volume by Month")


def plot_weekday_pattern(df):
    names = {0: "Monday", 1: "Tuesday", 2: "Wednesday", 3: "Thursday", 4: "Friday", 5: "Saturday", 6: "Sunday"}
    grouped = df.groupby("weekday")["daily_count"].mean().reset_index()
    grouped["weekday_name"] = grouped["weekday"].map(names)
    return px.bar(grouped, x="weekday_name", y="daily_count", title="Average Cycling Volume by Weekday")


def plot_hourly_commuting_pattern(hourly):
    grouped = hourly.groupby("hour")["hourly_count"].mean().reset_index()
    return px.line(grouped, x="hour", y="hourly_count", markers=True, title="Average Hourly Cycling Pattern")


def plot_weather_temperature(df):
    return px.scatter(df, x="temperature_mean", y="daily_count", color="is_weekend", trendline="ols", title="Temperature and Cycling Volume")


def plot_weather_rain(df):
    grouped = df.groupby("is_rainy_day")["daily_count"].mean().reset_index()
    grouped["weather"] = grouped["is_rainy_day"].map({0: "Dry day", 1: "Rainy day"})
    return px.bar(grouped, x="weather", y="daily_count", title="Average Cycling Volume on Dry vs Rainy Days")


def plot_top_municipalities(df):
    grouped = df.groupby("municipality")["daily_count"].mean().sort_values(ascending=False).head(15).reset_index()
    return px.bar(grouped, x="daily_count", y="municipality", orientation="h", title="Top Municipalities by Average Daily Cycling Volume")


def plot_map(df):
    map_df = df.groupby(["site_id", "municipality", "latitude", "longitude"])["daily_count"].mean().reset_index().dropna(subset=["latitude", "longitude"])
    return px.scatter_mapbox(
        map_df,
        lat="latitude",
        lon="longitude",
        size="daily_count",
        color="daily_count",
        hover_name="municipality",
        zoom=7,
        height=650,
        title="Spatial Distribution of Cycling Counts",
        mapbox_style="open-street-map"
    )


def plot_model_comparison(results_table):
    return px.bar(results_table, x="model", y="RMSE", title="Model Comparison: Lower RMSE is Better")


def plot_actual_vs_predicted(prediction_df):
    return px.scatter(prediction_df, x="daily_count", y="prediction", color="municipality", trendline="ols", title="Actual vs Predicted Daily Cycling Counts")


def plot_residuals(prediction_df):
    return px.scatter(prediction_df, x="prediction", y="residual", color="municipality", title="Residual Analysis")


def plot_feature_importance(importance_df):
    if importance_df.empty:
        return px.bar(title="Feature importance not available for selected model")
    top = importance_df.head(20).sort_values("importance")
    return px.bar(top, x="importance", y="feature", orientation="h", title="Top 20 Feature Importances")


def plot_permutation_importance(importance_df):
    top = importance_df.head(15).sort_values("importance_mean")
    return px.bar(top, x="importance_mean", y="feature", orientation="h", error_x="importance_std", title="Permutation Importance")


def plot_site_segments(segments):
    return px.scatter(
        segments,
        x="commuter_ratio",
        y="weekend_ratio",
        size="avg_hourly_count",
        color="segment_label",
        hover_data=["site_id"],
        title="Behavioural Segmentation of Counting Sites"
    )


def plot_anomalies(df):
    anomaly_df = df.groupby("datetime").agg(total_count=("daily_count", "sum"), anomaly_sites=("is_anomaly_day", "sum")).reset_index()
    return px.scatter(anomaly_df, x="datetime", y="total_count", size="anomaly_sites", title="Anomaly Days in Cycling Volume")
