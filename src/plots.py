import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


def plot_hourly_profile(df, savepath=None):
    profile = (
        df.groupby(["hour", "is_weekend"])["count"]
          .mean()
          .reset_index()
    )
    profile["day_type"] = np.where(profile["is_weekend"], "Weekend", "Weekday")

    fig, ax = plt.subplots(figsize=(10, 5))
    sns.lineplot(data=profile, x="hour", y="count", hue="day_type",
                 marker="o", ax=ax, linewidth=2)
    ax.set_title("Average hourly cycling count: weekday vs. weekend",
                 fontsize=14, fontweight="bold")
    ax.set_xlabel("Hour of day")
    ax.set_ylabel("Average cyclists per hour")
    ax.set_xticks(range(0, 24, 2))
    ax.legend(title="Day type")
    if savepath:
        fig.savefig(savepath, dpi=150, bbox_inches="tight")
    plt.show()


def plot_monthly_seasonality(df, savepath=None):
    daily = (
        df.groupby([df["timestamp"].dt.date, "month"], as_index=False)["count"]
          .sum()
    )
    fig, ax = plt.subplots(figsize=(10, 5))
    sns.boxplot(data=daily, x="month", y="count", ax=ax,
                palette="viridis", showfliers=False)
    ax.set_title("Daily-total cycling counts by month",
                 fontsize=14, fontweight="bold")
    ax.set_xlabel("Month")
    ax.set_ylabel("Total cyclists per day (all sites)")
    if savepath:
        fig.savefig(savepath, dpi=150, bbox_inches="tight")
    plt.show()


def plot_yearly_trend(df, savepath=None):
    yearly = df.groupby("year", as_index=False)["count"].sum()

    fig, ax = plt.subplots(figsize=(10, 5))
    sns.barplot(data=yearly, x="year", y="count", ax=ax,
                palette="rocket")
    ax.set_title("Total cycling counts per year",
                 fontsize=14, fontweight="bold")
    ax.set_xlabel("Year")
    ax.set_ylabel("Total cyclists")
    ax.bar_label(ax.containers[0], fmt="{:,.0f}", padding=3, fontsize=9)
    if savepath:
        fig.savefig(savepath, dpi=150, bbox_inches="tight")
    plt.show()


def plot_dayofweek_profile(df, savepath=None):
    profile = df.groupby("day_of_week", as_index=False)["count"].mean()
    day_labels = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]

    fig, ax = plt.subplots(figsize=(10, 5))
    sns.barplot(data=profile, x="day_of_week", y="count", ax=ax,
                palette="mako")
    ax.set_xticklabels(day_labels)
    ax.set_title("Average hourly cycling count by day of week",
                 fontsize=14, fontweight="bold")
    ax.set_xlabel("Day of week")
    ax.set_ylabel("Average cyclists per hour")
    if savepath:
        fig.savefig(savepath, dpi=150, bbox_inches="tight")
    plt.show()


def plot_heatmap_top_site(df, savepath=None):
    busiest = df.groupby("site_id")["count"].sum().idxmax()
    sub = df[df["site_id"] == busiest]
    site_name = sub["gemeente"].iloc[0]

    pivot = (
        sub.groupby(["day_of_week", "hour"])["count"]
           .mean()
           .unstack("hour")
    )
    fig, ax = plt.subplots(figsize=(14, 5))
    sns.heatmap(pivot, cmap="YlOrRd", ax=ax,
                cbar_kws={"label": "Average cyclists per hour"})
    ax.set_yticklabels(["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"],
                       rotation=0)
    ax.set_title(f"Hour × Day-of-week heatmap — busiest site: {site_name} (id={busiest})",
                 fontsize=14, fontweight="bold")
    ax.set_xlabel("Hour of day")
    ax.set_ylabel("Day of week")
    if savepath:
        fig.savefig(savepath, dpi=150, bbox_inches="tight")
    plt.show()


def plot_top_gemeenten(df, top_n=15, savepath=None):
    daily_per_gem = (
        df.groupby(["gemeente", df["timestamp"].dt.date])["count"]
          .sum()
          .reset_index()
          .groupby("gemeente")["count"]
          .mean()
          .reset_index(name="avg_daily")
          .nlargest(top_n, "avg_daily")
    )

    fig, ax = plt.subplots(figsize=(10, 7))
    sns.barplot(data=daily_per_gem, y="gemeente", x="avg_daily", ax=ax,
                palette="crest")
    ax.set_title(f"Top {top_n} gemeenten by average daily cyclists",
                 fontsize=14, fontweight="bold")
    ax.set_xlabel("Average cyclists per day")
    ax.set_ylabel("")
    if savepath:
        fig.savefig(savepath, dpi=150, bbox_inches="tight")
    plt.show()


def plot_site_map(df, top_n=50):
    import folium

    daily_per_site = (
        df.groupby(["site_id", "gemeente", "lat", "lon"])["count"]
          .agg(daily_avg=lambda s: s.sum() / max(s.index.nunique(), 1))
          .reset_index()
          .nlargest(top_n, "daily_avg")
    )

    centre = [daily_per_site["lat"].mean(), daily_per_site["lon"].mean()]
    m = folium.Map(location=centre, zoom_start=8, tiles="cartodbpositron")

    max_count = daily_per_site["daily_avg"].max()
    for _, row in daily_per_site.iterrows():
        folium.CircleMarker(
            location=[row["lat"], row["lon"]],
            radius=3 + 17 * (row["daily_avg"] / max_count),
            popup=f"<b>{row['gemeente']}</b><br>~{row['daily_avg']:.0f}/day<br>id={row['site_id']}",
            color="crimson",
            fill=True,
            fill_opacity=0.6,
        ).add_to(m)
    return m