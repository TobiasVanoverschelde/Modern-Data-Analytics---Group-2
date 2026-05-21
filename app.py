from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from shiny import App, reactive, render, ui
from src.features import add_cyclical_encoding

# Load data and model
PROCESSED_DIR = Path(__file__).parent / "data" / "processed"

df_hourly = pd.read_parquet(PROCESSED_DIR / "cycling_features.parquet")
df_daily = pd.read_parquet(PROCESSED_DIR / "daily_for_modeling.parquet")
df_daily = add_cyclical_encoding(df_daily)

# Interpretation artifacts, pre-computed by src/training.py
pdp_path = PROCESSED_DIR / "pdp_results.parquet"
pdp_df = pd.read_parquet(pdp_path) if pdp_path.exists() else None

importance_path = PROCESSED_DIR / "feature_importance.parquet"
importance_df_static = pd.read_parquet(importance_path) if importance_path.exists() else None

# Model — try MLflow first, otherwise fall back to joblib
try:
    import mlflow
    mlruns_path = (Path(__file__).parent / "notebooks" / "mlruns").resolve()
    mlflow.set_tracking_uri(mlruns_path.as_uri())
    model = mlflow.sklearn.load_model("models:/cycling-flanders/latest")
    print("Loaded model from MLflow registry")
except Exception as exc:
    print(f"MLflow load failed ({exc}), falling back to joblib")
    model = joblib.load(PROCESSED_DIR / "best_model.pkl")

# Vertex AI endpoint config
USE_VERTEX = True
PROJECT_ID = "mda-cycling-flanders"
REGION = "europe-west1"
ENDPOINT_NAME_FILE = PROCESSED_DIR / "vertex_endpoint.txt"

if USE_VERTEX and ENDPOINT_NAME_FILE.exists():
    try:
        from google.cloud import aiplatform
        aiplatform.init(project=PROJECT_ID, location=REGION)
        vertex_endpoint = aiplatform.Endpoint(ENDPOINT_NAME_FILE.read_text().strip())
        print("Using Vertex AI endpoint for predictions")
    except Exception as exc:
        print(f"Vertex init failed ({exc}), using local model")
        vertex_endpoint = None
else:
    vertex_endpoint = None


def predict(input_df):
    """Predict using Vertex endpoint if available, else local model."""
    if vertex_endpoint is not None:
        instances = input_df.to_dict(orient="records")
        response = vertex_endpoint.predict(instances=instances)
        return np.array(response.predictions)
    return model.predict(input_df)

GEMEENTEN = sorted(df_hourly["gemeente"].dropna().unique())
ALL_FLANDERS = "All Flanders"
GEMEENTE_CHOICES = [ALL_FLANDERS] + list(GEMEENTEN)
DEFAULT_GEMEENTE = ALL_FLANDERS

# Average lat/lon per gemeente, used as spatial input for what-if scenarios.
# Falls back to a Flanders-centre coordinate when a gemeente has no sites with coordinates.
if {"lat", "lon"}.issubset(df_daily.columns):
    GEMEENTE_COORDS = (
        df_daily.dropna(subset=["lat", "lon"])
                .groupby("gemeente")[["lat", "lon"]]
                .mean()
                .to_dict("index")
    )
else:
    GEMEENTE_COORDS = {}
FLANDERS_CENTRE = {"lat": 51.0, "lon": 4.5}

sns.set_theme(style="whitegrid", context="notebook")


SITES = (
    df_daily.dropna(subset=["lat", "lon"])
            .groupby(["site_id", "gemeente"], as_index=False)[["lat", "lon"]]
            .mean()
)


def build_predicted_map(sites_df: pd.DataFrame, predictions) -> str:
    """Folium map with each counter coloured by its predicted daily count."""
    import folium
    import matplotlib.colors as mcolors

    centre = [sites_df["lat"].mean(), sites_df["lon"].mean()]
    m = folium.Map(location=centre, zoom_start=8, tiles="cartodbpositron")

    cmap = plt.cm.RdYlBu_r
    norm = mcolors.Normalize(vmin=float(predictions.min()), vmax=float(predictions.max()))
    for (_, row), pred in zip(sites_df.iterrows(), predictions):
        hex_color = mcolors.to_hex(cmap(norm(pred)))
        folium.CircleMarker(
            location=[row["lat"], row["lon"]],
            radius=7,
            popup=f"<b>{row['gemeente']}</b><br>Predicted: {pred:,.0f} cyclists/day",
            color=hex_color, fill=True, fill_color=hex_color, fill_opacity=0.8,
        ).add_to(m)
    return m._repr_html_()

# UI
app_ui = ui.page_navbar(
    # Tab 1: Temporal Explorer
    ui.nav_panel(
        "Temporal explorer",
        ui.layout_sidebar(
            ui.sidebar(
                ui.input_select("gemeente", "Municipality",
                                choices=GEMEENTE_CHOICES,
                                selected=DEFAULT_GEMEENTE),
                ui.input_checkbox_group(
                    "years", "Years",
                    choices=[str(y) for y in sorted(df_hourly["year"].unique())],
                    selected=[str(y) for y in sorted(df_hourly["year"].unique())[-3:]],
                ),
                ui.markdown(
                    "**Interpretation:** weekday curves show two commuting peaks "
                    "(~8 AM, ~5 PM); weekend curves show one broader afternoon peak."
                ),
            ),
            ui.h3("Hourly cycling profile"),
            ui.output_plot("hourly_plot"),
            ui.h3("Day-of-week and monthly patterns"),
            ui.output_plot("dow_month_plot"),
        ),
    ),

    # Tab 2: Model Insights
    ui.nav_panel(
        "Model insights",
        ui.layout_sidebar(
            ui.sidebar(
                ui.input_numeric("n_features", "Top N features", value=15, min=5, max=30),
                ui.markdown(
                    "**Drivers of cycling volume** are quantified using the best "
                    "model's feature importances (tree models) or absolute "
                    "coefficients (linear models)."
                ),
            ),
            ui.h3("Top feature importances"),
            ui.output_plot("importance_plot"),
            ui.h3("Predicted vs actual"),
            ui.output_plot("predvactual_plot"),
        ),
    ),

    # Tab 3: Weather Impact
    ui.nav_panel(
        "Weather impact",
        ui.layout_sidebar(
            ui.sidebar(
                ui.input_slider("temp", "Temperature (°C)",
                                min=-5, max=35, value=15, step=1),
                ui.input_slider("precip", "Precipitation (mm)",
                                min=0, max=30, value=0, step=0.5),
                ui.input_slider("wind", "Wind speed (km/h)",
                                min=0, max=60, value=10, step=1),
                ui.input_select("scenario_gemeente", "Municipality",
                                choices=GEMEENTE_CHOICES,
                                selected=DEFAULT_GEMEENTE),
                ui.input_select(
                    "scenario_day", "Day of week",
                    choices={"0": "Monday", "1": "Tuesday", "2": "Wednesday",
                             "3": "Thursday", "4": "Friday", "5": "Saturday",
                             "6": "Sunday"},
                    selected="2",
                ),
                ui.markdown(
                    "**What-if scenarios:** adjust the sliders to see how the model "
                    "responds to weather conditions for a given gemeente and weekday."
                ),
            ),
            ui.h3("Model prediction for this scenario"),
            ui.output_text("scenario_prediction"),
            ui.h3("Effect of temperature, sweeping all values"),
            ui.output_plot("weather_sweep_plot"),
            ui.h3("Average effect of weather across all sites (partial dependence)"),
            ui.output_plot("pdp_weather"),
            ui.h3("Observed effect of weather"),
            ui.output_plot("weather_scatter_plot"),
        ),
    ),

    # Tab 4: Spatial
    ui.nav_panel(
        "Spatial",
        ui.layout_sidebar(
            ui.sidebar(
                ui.input_slider("spatial_temp", "Temperature (°C)",
                                min=-5, max=35, value=15, step=1),
                ui.input_slider("spatial_precip", "Precipitation (mm)",
                                min=0, max=30, value=0, step=0.5),
                ui.input_slider("spatial_wind", "Wind speed (km/h)",
                                min=0, max=60, value=10, step=1),
                ui.input_select(
                    "spatial_day", "Day of week",
                    choices={"0": "Monday", "1": "Tuesday", "2": "Wednesday",
                             "3": "Thursday", "4": "Friday", "5": "Saturday",
                             "6": "Sunday"},
                    selected="2",
                ),
                ui.markdown(
                    "**Where does the model expect cycling activity?** "
                    "Each circle is a counter location, coloured by the model's "
                    "prediction under the conditions on the left."
                ),
            ),
            ui.h3("Predicted cycling activity per counter"),
            ui.output_ui("spatial_map"),
        ),
    ),

    title="Cycling Patterns in Flanders",
)


# Server
def server(input, output, session):

    # ------------- Tab 1: Temporal Explorer -------------
    @reactive.Calc
    def filtered_hourly():
        years = [int(y) for y in input.years()]
        if not years:
            return df_hourly.iloc[0:0]
        sub = df_hourly[df_hourly["year"].isin(years)]
        if input.gemeente() != ALL_FLANDERS:
            sub = sub[sub["gemeente"] == input.gemeente()]
        return sub

    @output
    @render.plot
    def hourly_plot():
        sub = filtered_hourly()
        if sub.empty:
            fig, ax = plt.subplots()
            ax.text(0.5, 0.5, "No data for selection", ha="center", va="center")
            return fig
        profile = (
            sub.groupby(["hour", "is_weekend"])["count"]
               .mean()
               .reset_index()
        )
        profile["day_type"] = np.where(profile["is_weekend"], "Weekend", "Weekday")
        fig, ax = plt.subplots(figsize=(10, 4.5))
        sns.lineplot(data=profile, x="hour", y="count", hue="day_type",
                     marker="o", ax=ax, linewidth=2)
        ax.set_title(f"Hourly profile: {input.gemeente()}"
                     + (" (averaged across all counters)"
                        if input.gemeente() == ALL_FLANDERS else ""))
        ax.set_xticks(range(0, 24, 2))
        return fig

    @output
    @render.plot
    def dow_month_plot():
        sub = filtered_hourly()
        if sub.empty:
            fig, ax = plt.subplots()
            ax.text(0.5, 0.5, "No data", ha="center", va="center")
            return fig
        fig, axes = plt.subplots(1, 2, figsize=(14, 4))
        day_labels = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
        dow = sub.groupby("day_of_week")["count"].mean().reset_index()
        sns.barplot(data=dow, x="day_of_week", y="count", ax=axes[0],
                    palette="mako")
        axes[0].set_xticklabels(day_labels)
        axes[0].set_title("By day of week")
        mn = sub.groupby("month")["count"].mean().reset_index()
        sns.barplot(data=mn, x="month", y="count", ax=axes[1], palette="viridis")
        axes[1].set_title("By month")
        plt.tight_layout()
        return fig

    # ------------- Tab 2: Model Insights -------------
    @reactive.Calc
    def importance_df():
        # Use the saved permutation importance: model-agnostic, less biased
        # against one-hot categoricals than the tree built-in importance.
        if importance_df_static is not None:
            return importance_df_static
        # Fallback if the artifact wasn't generated (shouldn't happen in
        # normal flow — training script saves it).
        return pd.DataFrame({"feature": [], "importance": []})

    @output
    @render.plot
    def importance_plot():
        imp = importance_df().nlargest(input.n_features(), "importance")
        fig, ax = plt.subplots(figsize=(10, 0.4 * len(imp) + 1))
        sns.barplot(data=imp, y="feature", x="importance", ax=ax, palette="rocket")
        ax.set_title(f"Top {input.n_features()} features")
        plt.tight_layout()
        return fig

    @output
    @render.plot
    def predvactual_plot():
        # Last 20% of dates = held-out test set
        cutoff = df_daily["date"].quantile(0.8)
        test = df_daily[df_daily["date"] >= cutoff]
        from src.modeling import ALL_FEATURES
        from sklearn.metrics import r2_score
        y_true = test["count"]
        y_pred = model.predict(test[ALL_FEATURES])
        r2 = r2_score(y_true, y_pred)

        fig, ax = plt.subplots(figsize=(7, 7))
        ax.scatter(y_true, y_pred, alpha=0.3, s=10,
                   c=test["is_weekend"].map({True: "crimson", False: "steelblue"}))
        lim = [0, max(y_true.max(), y_pred.max())]
        ax.plot(lim, lim, "k--", linewidth=1, label="Perfect prediction")
        ax.set_xlim(lim); ax.set_ylim(lim)
        ax.set_xlabel("Actual daily cyclists at this counter")
        ax.set_ylabel("Predicted daily cyclists at this counter")
        ax.set_title(f"Predicted vs actual on held-out test set (R² = {r2:.2f})\n"
                     "One point = one (date, counter) pair. Blue = weekday, red = weekend.",
                     fontsize=11)
        ax.legend(loc="upper left")
        return fig

    # ------------- Tab 3: Weather Impact -------------
    # One row per gemeente — when "All Flanders" we predict for all sites and
    # average; for a specific gemeente this is a single row.
    @reactive.Calc
    def scenario_input():
        day = int(input.scenario_day())
        month = 6
        gemeenten = (list(GEMEENTEN) if input.scenario_gemeente() == ALL_FLANDERS
                     else [input.scenario_gemeente()])
        rows = []
        for g in gemeenten:
            coords = GEMEENTE_COORDS.get(g, FLANDERS_CENTRE)
            rows.append({
                "temperature_2m": input.temp(),
                "precipitation": input.precip(),
                "wind_speed_10m": input.wind(),
                "cloud_cover": 50.0,
                "lat": coords["lat"],
                "lon": coords["lon"],
                "day_of_week_sin": np.sin(2 * np.pi * day / 7),
                "day_of_week_cos": np.cos(2 * np.pi * day / 7),
                "month_sin": np.sin(2 * np.pi * month / 12),
                "month_cos": np.cos(2 * np.pi * month / 12),
                "gemeente": g,
                "covid_period": "post_covid",
                "is_weekend": day >= 5,
                "is_holiday": False,
            })
        return pd.DataFrame(rows)

    @output
    @render.text
    def scenario_prediction():
        preds = predict(scenario_input())
        avg = float(np.mean(preds))
        if input.scenario_gemeente() == ALL_FLANDERS:
            return f"Avg predicted daily cycling count across {len(preds)} counters: {avg:,.0f}"
        return f"Predicted daily cycling count: {avg:,.0f}"

    @output
    @render.plot
    def weather_sweep_plot():
        # Sweep temperature; for each value predict over all rows in scenario_input
        # (one row per gemeente when "All Flanders") and average.
        base = scenario_input()
        temps = np.linspace(-5, 35, 41)
        preds = []
        for t in temps:
            sweep = base.copy()
            sweep["temperature_2m"] = t
            preds.append(float(np.mean(predict(sweep))))

        fig, ax = plt.subplots(figsize=(10, 4.5))
        ax.plot(temps, preds, linewidth=2, color="steelblue")
        ax.axvline(input.temp(), color="crimson", linestyle="--",
                   label=f"Current: {input.temp()}°C")
        ax.set_xlabel("Temperature (°C)")
        ax.set_ylabel("Predicted daily count")
        suffix = " (avg across all counters)" if input.scenario_gemeente() == ALL_FLANDERS else ""
        ax.set_title(f"Predicted count vs temperature{suffix}")
        ax.legend()
        return fig

    @output
    @render.plot
    def pdp_weather():
        features = [
            ("temperature_2m", "Temperature (°C)"),
            ("precipitation", "Precipitation (mm/day)"),
            ("wind_speed_10m", "Wind speed (km/h)"),
        ]
        fig, axes = plt.subplots(1, 3, figsize=(15, 4))
        if pdp_df is None:
            for ax in axes:
                ax.text(0.5, 0.5, "Run training to generate pdp_results.parquet",
                        ha="center", va="center", transform=ax.transAxes)
                ax.axis("off")
            return fig
        for ax, (feat, xlabel) in zip(axes, features):
            sub = pdp_df[pdp_df["feature"] == feat]
            baseline = sub["prediction"].mean()
            pct = (sub["prediction"] - baseline) / baseline * 100
            ax.plot(sub["grid_value"], pct, color="steelblue", linewidth=2)
            ax.axhline(0, color="grey", linestyle=":", linewidth=1)
            ax.set_xlabel(xlabel)
            ax.set_title(feat)
        axes[0].set_ylabel("% change from mean predicted count")
        plt.tight_layout()
        return fig

    @output
    @render.plot
    def weather_scatter_plot():
        if input.scenario_gemeente() == ALL_FLANDERS:
            sub = df_daily
        else:
            sub = df_daily[df_daily["gemeente"] == input.scenario_gemeente()]
        fig, ax = plt.subplots(figsize=(10, 4.5))
        ax.scatter(sub["temperature_2m"], sub["count"], alpha=0.4, s=8)
        ax.set_xlabel("Daily average temperature (°C)")
        ax.set_ylabel("Daily count")
        ax.set_title(f"Observed: temp vs count in {input.scenario_gemeente()}")
        return fig

    # ------------- Tab 4: Spatial -------------
    @reactive.Calc
    def spatial_predictions():
        day = int(input.spatial_day())
        month = 6
        rows = [{
            "temperature_2m": input.spatial_temp(),
            "precipitation": input.spatial_precip(),
            "wind_speed_10m": input.spatial_wind(),
            "cloud_cover": 50.0,
            "lat": s["lat"],
            "lon": s["lon"],
            "day_of_week_sin": np.sin(2 * np.pi * day / 7),
            "day_of_week_cos": np.cos(2 * np.pi * day / 7),
            "month_sin": np.sin(2 * np.pi * month / 12),
            "month_cos": np.cos(2 * np.pi * month / 12),
            "gemeente": s["gemeente"],
            "covid_period": "post_covid",
            "is_weekend": day >= 5,
            "is_holiday": False,
        } for _, s in SITES.iterrows()]
        return np.asarray(predict(pd.DataFrame(rows)))

    @output
    @render.ui
    def spatial_map():
        return ui.HTML(build_predicted_map(SITES, spatial_predictions()))


app = App(app_ui, server)