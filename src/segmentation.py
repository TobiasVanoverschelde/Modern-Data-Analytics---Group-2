import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler


class SiteSegmenter:
    """Creates simple behavioural site segments."""

    def create_site_segments(self, hourly: pd.DataFrame, n_clusters: int = 4) -> pd.DataFrame:
        hourly = hourly.copy()
        hourly["datetime"] = pd.to_datetime(hourly["datetime"])
        hourly["hour"] = hourly["datetime"].dt.hour
        hourly["weekday"] = hourly["datetime"].dt.dayofweek
        hourly["is_weekend"] = hourly["weekday"].isin([5, 6]).astype(int)

        site_features = hourly.groupby("site_id").agg(
            avg_hourly_count=("hourly_count", "mean"),
            morning_peak=("hourly_count", lambda x: x[hourly.loc[x.index, "hour"].between(7, 9)].mean()),
            evening_peak=("hourly_count", lambda x: x[hourly.loc[x.index, "hour"].between(16, 18)].mean()),
            weekend_avg=("hourly_count", lambda x: x[hourly.loc[x.index, "is_weekend"] == 1].mean()),
            weekday_avg=("hourly_count", lambda x: x[hourly.loc[x.index, "is_weekend"] == 0].mean())
        ).reset_index()

        site_features = site_features.fillna(0)
        site_features["commuter_ratio"] = (site_features["morning_peak"] + site_features["evening_peak"]) / (site_features["avg_hourly_count"] + 1)
        site_features["weekend_ratio"] = site_features["weekend_avg"] / (site_features["weekday_avg"] + 1)

        clustering_features = ["avg_hourly_count", "commuter_ratio", "weekend_ratio"]
        X = StandardScaler().fit_transform(site_features[clustering_features])
        kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        site_features["cluster"] = kmeans.fit_predict(X)

        labels = {}
        for cluster in sorted(site_features["cluster"].unique()):
            subset = site_features[site_features["cluster"] == cluster]
            avg_commuter = subset["commuter_ratio"].mean()
            avg_weekend = subset["weekend_ratio"].mean()
            if avg_commuter > site_features["commuter_ratio"].median() and avg_weekend < site_features["weekend_ratio"].median():
                labels[cluster] = "Commuter-oriented site"
            elif avg_weekend > site_features["weekend_ratio"].median():
                labels[cluster] = "Leisure/weekend-oriented site"
            else:
                labels[cluster] = "Mixed-use site"
        site_features["segment_label"] = site_features["cluster"].map(labels)
        return site_features
