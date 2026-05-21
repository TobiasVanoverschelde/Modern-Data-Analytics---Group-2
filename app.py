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
DEFAULT_GEMEENTE = "Leuven" if "Leuven" in GEMEENTEN else GEMEENTEN[0]

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


def build_site_map(daily_df: pd.DataFrame, top_n: int = 50) -> str:
    """Folium map of the top-N busiest sites; returns embeddable HTML."""
    import folium

    per_site = (
        daily_df.dropna(subset=["lat", "lon"])
                .groupby(["site_id", "gemeente", "lat", "lon"], as_index=False)["count"]
                .mean()
                .nlargest(top_n, "count")
    )
    if per_site.empty:
        return "<i>No sites with coordinates available.</i>"

    centre = [per_site["lat"].mean(), per_site["lon"].mean()]
    m = folium.Map(location=centre, zoom_start=8, tiles="cartodbpositron")

    max_count = per_site["count"].max()
    for _, row in per_site.iterrows():
        folium.CircleMarker(
            location=[row["lat"], row["lon"]],
            radius=3 + 17 * (row["count"] / max_count),
            popup=f"<b>{row['gemeente']}</b><br>~{row['count']:.0f} cyclists/day<br>site {row['site_id']}",
            color="crimson",
            fill=True,
            fill_opacity=0.6,
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
                                choices=GEMEENTEN,
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
                                choices=GEMEENTEN,
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
            ui.h3("Average effect of temperature across all sites (partial dependence)"),
            ui.output_plot("pdp_temperature"),
            ui.h3("Observed effect of weather"),
            ui.output_plot("weather_scatter_plot"),
        ),
    ),

    # Tab 4: Spatial
    ui.nav_panel(
        "Spatial",
        ui.layout_sidebar(
            ui.sidebar(
                ui.input_numeric("spatial_top_n", "Show top N sites",
                                 value=50, min=10, max=200),
                ui.markdown(
                    "**Where do people cycle?** Circle size encodes average daily "
                    "count per counting site. `lat`/`lon` are also numeric features "
                    "in the model, so the regressor can learn geographic gradients "
                    "(urban vs rural) on top of the categorical municipality."
                ),
            ),
            ui.h3("Top counting sites in Flanders"),
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
        return df_hourly[
            (df_hourly["gemeente"] == input.gemeente())
            & (df_hourly["year"].isin(years))
        ]

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
        ax.set_title(f"Hourly profile: {input.gemeente()}")
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
        # Use the last 20% of daily data as test
        cutoff = df_daily["date"].quantile(0.8)
        test = df_daily[df_daily["date"] >= cutoff]
        from src.modeling import ALL_FEATURES
        y_true = test["count"]
        y_pred = model.predict(test[ALL_FEATURES])

        fig, ax = plt.subplots(figsize=(7, 7))
        ax.scatter(y_true, y_pred, alpha=0.3, s=10)
        lim = [0, max(y_true.max(), y_pred.max())]
        ax.plot(lim, lim, "k--", linewidth=1)
        ax.set_xlim(lim); ax.set_ylim(lim)
        ax.set_xlabel("Actual"); ax.set_ylabel("Predicted")
        ax.set_title("Predicted vs actual (test set)")
        return fig

    # ------------- Tab 3: Weather Impact -------------
    @reactive.Calc
    def scenario_input():
        day = int(input.scenario_day())
        month = 6                        # neutral default
        coords = GEMEENTE_COORDS.get(input.scenario_gemeente(), FLANDERS_CENTRE)
        return pd.DataFrame([{
            "temperature_2m": input.temp(),
            "precipitation": input.precip(),
            "wind_speed_10m": input.wind(),
            "cloud_cover": 50.0,         # neutral default
            "lat": coords["lat"],
            "lon": coords["lon"],
            "day_of_week_sin": np.sin(2 * np.pi * day / 7),
            "day_of_week_cos": np.cos(2 * np.pi * day / 7),
            "month_sin": np.sin(2 * np.pi * month / 12),
            "month_cos": np.cos(2 * np.pi * month / 12),
            "gemeente": input.scenario_gemeente(),
            "covid_period": "post_covid",
            "is_weekend": day >= 5,
            "is_holiday": False,
        }])

    @output
    @render.text
    def scenario_prediction():
        pred = predict(scenario_input())[0]
        return f"Predicted daily cycling count: {pred:,.0f}"

    @output
    @render.plot
    def weather_sweep_plot():
        # Sweep temperature, hold other inputs constant
        base = scenario_input().iloc[0]
        temps = np.linspace(-5, 35, 41)
        rows = [base.copy() for _ in temps]
        for r, t in zip(rows, temps):
            r["temperature_2m"] = t
        sweep_df = pd.DataFrame(rows)
        preds = predict(sweep_df)

        fig, ax = plt.subplots(figsize=(10, 4.5))
        ax.plot(temps, preds, linewidth=2, color="steelblue")
        ax.axvline(input.temp(), color="crimson", linestyle="--",
                   label=f"Current: {input.temp()}°C")
        ax.set_xlabel("Temperature (°C)")
        ax.set_ylabel("Predicted daily count")
        ax.set_title("Predicted count as a function of temperature")
        ax.legend()
        return fig

    @output
    @render.plot
    def pdp_temperature():
        fig, ax = plt.subplots(figsize=(10, 4.5))
        if pdp_df is None:
            ax.text(0.5, 0.5,
                    "Run notebook 03 to generate pdp_results.parquet",
                    ha="center", va="center", transform=ax.transAxes)
            ax.axis("off")
            return fig
        sub = pdp_df[pdp_df["feature"] == "temperature_2m"]
        ax.plot(sub["grid_value"], sub["prediction"],
                color="steelblue", linewidth=2)
        ax.set_xlabel("Temperature (°C)")
        ax.set_ylabel("Avg predicted daily count")
        ax.set_title("Partial dependence: temperature")
        return fig

    @output
    @render.plot
    def weather_scatter_plot():
        # Observed temp vs count in the daily data
        sub = df_daily[df_daily["gemeente"] == input.scenario_gemeente()]
        fig, ax = plt.subplots(figsize=(10, 4.5))
        ax.scatter(sub["temperature_2m"], sub["count"], alpha=0.4, s=8)
        ax.set_xlabel("Daily average temperature (°C)")
        ax.set_ylabel("Daily count")
        ax.set_title(f"Observed: temp vs count in {input.scenario_gemeente()}")
        return fig

    # ------------- Tab 4: Spatial -------------
    @output
    @render.ui
    def spatial_map():
        html = build_site_map(df_daily, top_n=input.spatial_top_n())
        return ui.HTML(html)


app = App(app_ui, server)