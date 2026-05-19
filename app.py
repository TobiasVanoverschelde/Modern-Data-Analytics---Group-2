from pathlib import Path

from shiny import App, ui, render, reactive

from shinywidgets import output_widget, render_widget

import pandas as pd

import plotly.express as px

import plotly.graph_objects as go

# =========================================================

# PATHS

# =========================================================

PROCESSED_DIR = Path("data/processed")

PRECOMPUTED_DIR = Path("data/precomputed")

# =========================================================

# HELPERS

# =========================================================

def load_csv(path, required=True):

    path = Path(path)

    if path.exists():

        return pd.read_csv(path)

    if required:

        raise FileNotFoundError(f"Missing required file: {path}")

    return pd.DataFrame()

def clean_datetime(df, col):

    if col in df.columns:

        df[col] = pd.to_datetime(

            df[col],

            errors="coerce"

        )

    return df

def empty_figure(title):

    fig = go.Figure()

    fig.update_layout(

        title=title,

        template="plotly_white",

        paper_bgcolor="#f7faf7",

        plot_bgcolor="white",

        height=500

    )

    return fig

# =========================================================

# LOAD DATA

# =========================================================

municipality_daily = load_csv(

    PROCESSED_DIR / "municipality_daily_counts.csv"

)

predictions = load_csv(

    PROCESSED_DIR / "predictions.csv",

    required=False

)

importance = load_csv(

    PROCESSED_DIR / "feature_importance.csv",

    required=False

)

weekday_summary = load_csv(

    PRECOMPUTED_DIR / "weekday_summary.csv",

    required=False

)

monthly_summary = load_csv(

    PRECOMPUTED_DIR / "monthly_summary.csv",

    required=False

)

yearly_summary = load_csv(

    PRECOMPUTED_DIR / "yearly_summary.csv",

    required=False

)

weather_corr = load_csv(

    PRECOMPUTED_DIR / "weather_correlations.csv",

    required=False

)

hourly_curve_summary = load_csv(

    PRECOMPUTED_DIR / "hourly_curve_summary.csv",

    required=False

)

hourly_heatmap_summary = load_csv(

    PRECOMPUTED_DIR / "hourly_heatmap_summary.csv",

    required=False

)

municipality_kpis = load_csv(

    PRECOMPUTED_DIR / "municipality_kpis.csv",

    required=False

)

spatial_summary = load_csv(

    PRECOMPUTED_DIR / "spatial_summary.csv",

    required=False

)

weather_merged = load_csv(
    PRECOMPUTED_DIR / "weather_merged.csv"
)

# =========================================================

# CLEAN DATA

# =========================================================

municipality_daily = clean_datetime(

    municipality_daily,

    "datetime"

)

municipality_daily = municipality_daily.dropna(

    subset=["datetime"]

)

for col in ["peak_day", "start_date", "end_date"]:

    municipality_kpis = clean_datetime(

        municipality_kpis,

        col

    )

# =========================================================

# WEATHER COLUMN NORMALIZATION

# =========================================================

weather_renames = {

    "temperature_2m_mean": "temperature",

    "precipitation_sum": "precipitation",

    "wind_speed_10m_max": "wind_speed"

}

municipality_daily = municipality_daily.rename(

    columns=weather_renames

)

weather_columns = [

    "temperature",

    "precipitation",

    "wind_speed",

    "sunshine_duration",

    "precipitation_hours"

]

for col in weather_columns:

    if col not in municipality_daily.columns:

        municipality_daily[col] = pd.NA

# =========================================================

# FEATURE LABELS

# =========================================================

feature_mapping = {

    "year": "Year",

    "month": "Month",

    "day_of_week": "Day of Week",

    "previous_day_traffic": "Previous Day Traffic",

    "traffic_one_week_earlier": "Traffic One Week Earlier",

    "temperature": "Temperature",

    "precipitation": "Rainfall",

    "wind_speed": "Wind Speed",

    "sunshine_duration": "Sunshine Duration",

    "precipitation_hours": "Rain Hours"

}

if not importance.empty and "feature" in importance.columns:

    importance["feature"] = (

        importance["feature"]

        .map(feature_mapping)

        .fillna(importance["feature"])

    )

if not weather_corr.empty and "weather_variable" in weather_corr.columns:

    weather_corr["weather_variable"] = (

        weather_corr["weather_variable"]

        .map(feature_mapping)

        .fillna(weather_corr["weather_variable"])

    )

# =========================================================

# MUNICIPALITIES

# =========================================================

municipalities = sorted(

    municipality_daily["municipality"]

    .dropna()

    .unique()

)

# =========================================================

# FALLBACK KPI TABLE

# =========================================================

if municipality_kpis.empty:

    rows = []

    for municipality in municipalities:

        df = municipality_daily[

            municipality_daily["municipality"] == municipality

        ]

        if df.empty:

            continue

        peak = df.loc[

            df["daily_count"].idxmax()

        ]

        rows.append({

            "municipality": municipality,

            "avg_daily": df["daily_count"].mean(),

            "total_cyclists": df["daily_count"].sum(),

            "peak_day": peak["datetime"],

            "peak_count": peak["daily_count"],

            "start_date": df["datetime"].min(),

            "end_date": df["datetime"].max(),

            "active_days": df["datetime"].nunique()

        })

    municipality_kpis = pd.DataFrame(rows)

# =========================================================

# GLOBAL KPIS

# =========================================================

global_start = municipality_daily["datetime"].min()

global_end = municipality_daily["datetime"].max()

total_municipalities = len(municipalities)

daily_global = (

    municipality_daily
    .groupby("datetime")["daily_count"]
    .sum()

)

global_avg_daily = int(
    daily_global.mean()
)

global_daily = (

    municipality_daily
    .groupby("datetime")["daily_count"]
    .sum()
    .reset_index()

)

peak_row = global_daily.loc[
    global_daily["daily_count"].idxmax()
]

peak_day = peak_row["datetime"]

peak_count = int(
    peak_row["daily_count"]
)

valid_kpis = municipality_kpis[
    municipality_kpis["active_days"] > 365
].copy()

top_row = valid_kpis.loc[
    valid_kpis["avg_daily"].idxmax()
]

top_municipality = top_row["municipality"]

top_avg = int(

    top_row["avg_daily"]

)

# =========================================================

# STATIC FIGURES

# =========================================================

spatial_df = spatial_summary.dropna(

    subset=["latitude", "longitude"]

)

STATIC_SPATIAL_FIG = px.scatter_mapbox(

    spatial_df,

    lat="latitude",

    lon="longitude",

    size="total_cyclists",

    color="total_cyclists",

    hover_name="municipality",

    hover_data={

        "total_cyclists": ":,",

        "latitude": False,

        "longitude": False

    },

    zoom=7,

    height=850,

    color_continuous_scale="Greens",

    size_max=45

)

STATIC_SPATIAL_FIG.update_layout(

    mapbox_style="carto-positron",

    margin=dict(

        l=0,

        r=0,

        t=50,

        b=0

    ),

    title="Spatial Distribution of Cycling Activity",

    template="plotly_white",

    paper_bgcolor="#f7faf7"

)

prediction_sample = predictions.sample(

    min(3000, len(predictions)),

    random_state=42

)

STATIC_PREDICTION_FIG = px.scatter(

    prediction_sample,

    x="actual",

    y="predicted",

    opacity=0.45

)

STATIC_PREDICTION_FIG.update_traces(

    marker=dict(color="#2d6a4f")

)

STATIC_PREDICTION_FIG.add_trace(

    go.Scatter(

        x=prediction_sample["actual"],

        y=prediction_sample["actual"],

        mode="lines",

        line=dict(

            color="#081c15",

            width=3

        ),

        name="Perfect Prediction"

    )

)

STATIC_PREDICTION_FIG.update_layout(

    title="Actual vs Predicted Cycling Counts",

    xaxis_title="Actual Counts",

    yaxis_title="Predicted Counts",

    template="plotly_white",

    paper_bgcolor="#f7faf7",

    plot_bgcolor="white",

    height=550

)

residual_sample = prediction_sample.copy()

residual_sample["residual"] = (

    residual_sample["actual"]

    - residual_sample["predicted"]

)

residual_sample["absolute_error"] = (

    residual_sample["residual"].abs()

)

STATIC_RESIDUAL_FIG = px.scatter(

    residual_sample,

    x="predicted",

    y="residual",

    opacity=0.45

)

STATIC_RESIDUAL_FIG.update_traces(

    marker=dict(color="#2d6a4f")

)

STATIC_RESIDUAL_FIG.add_hline(

    y=0,

    line_dash="dash",

    line_color="#081c15"

)

STATIC_RESIDUAL_FIG.update_layout(

    title="Residuals vs Predicted Counts",

    xaxis_title="Predicted Counts",

    yaxis_title="Residual",

    template="plotly_white",

    paper_bgcolor="#f7faf7",

    plot_bgcolor="white",

    height=550

)

STATIC_ERROR_DIST_FIG = px.histogram(

    residual_sample,

    x="absolute_error",

    nbins=40

)

STATIC_ERROR_DIST_FIG.update_traces(

    marker_color="#2d6a4f"

)

STATIC_ERROR_DIST_FIG.update_layout(

    title="Prediction Error Distribution",

    xaxis_title="Absolute Error",

    yaxis_title="Frequency",

    template="plotly_white",

    paper_bgcolor="#f7faf7",

    plot_bgcolor="white",

    height=550

)

mae = (

    predictions["actual"]

    - predictions["predicted"]

).abs().mean()

rmse = (

    (

        (

            predictions["actual"]

            - predictions["predicted"]

        ) ** 2

    ).mean()

) ** 0.5

STATIC_MODEL_ERROR_FIG = px.bar(

    pd.DataFrame({

        "Metric": [

            "Mean Absolute Error",

            "Root Mean Squared Error"

        ],

        "Value": [

            mae,

            rmse

        ]

    }),

    x="Metric",

    y="Value"

)

STATIC_MODEL_ERROR_FIG.update_traces(

    marker_color="#1b4332"

)

STATIC_MODEL_ERROR_FIG.update_layout(

    title="Model Error Summary",

    xaxis_title="Metric",

    yaxis_title="Error",

    template="plotly_white",

    paper_bgcolor="#f7faf7",

    plot_bgcolor="white",

    height=550

)

STATIC_IMPORTANCE_FIG = px.bar(

    importance.sort_values(

        "importance",

        ascending=True

    ),

    x="importance",

    y="feature",

    orientation="h"

)

STATIC_IMPORTANCE_FIG.update_traces(

    marker_color="#1b4332"

)

STATIC_IMPORTANCE_FIG.update_layout(

    title="Most Important Predictive Features",

    xaxis_title="Importance Score",

    yaxis_title="Feature",

    template="plotly_white",

    paper_bgcolor="#f7faf7",

    plot_bgcolor="white",

    height=650

)

# =========================================================

# UI

# =========================================================

app_ui = ui.page_fluid(

    ui.tags.head(

        ui.tags.style("""

            body {

                background-color: #f7faf7;

                font-family: Inter, Arial, sans-serif;

                color: #1b4332;

            }

            .card {

                border-radius: 24px;

                border: none;

                background: white;

                padding: 22px;

                box-shadow: 0 8px 24px rgba(0,0,0,0.055);

                margin-bottom: 16px;

            }

            .small-card {

                border-radius: 18px;

                border: none;

                background: white;

                padding: 16px;

                box-shadow: 0 6px 18px rgba(0,0,0,0.045);

                margin-bottom: 12px;

            }

            h1 {

                color: #1b4332;

                font-weight: 850;

                font-size: 42px;

            }

            h2, h3 {

                color: #1b4332;

                font-weight: 750;

            }

            h5 {

                color: #52796f;

                font-weight: 650;

            }

            p {

                color: #52796f;

            }

            .nav-link {

                color: #2d6a4f !important;

                font-weight: 650;

                border-radius: 12px;

            }

            .nav-link.active {

                background-color: #d8f3dc !important;

            }

        """)

    ),

    ui.div(

        ui.h1(

            "🚴 Cycling Behaviour Analytics Dashboard"

        ),

        ui.p(

            "Municipality-level cycling analytics enriched with weather, temporal behaviour, spatial analysis, prediction diagnostics, and feature importance."

        )

    ),

    ui.layout_sidebar(

        ui.sidebar(

            ui.input_select(

                "municipality",

                "Municipality",

                choices=["All"] + municipalities,

                selected="All"

            ),

            width=260

        ),

        ui.div(

            ui.row(

                ui.column(

                    3,

                    ui.div(

                        ui.h5("Municipalities"),

                        ui.h2(f"{total_municipalities:,}"),

                        ui.p(

                            f"{global_start.date()} → {global_end.date()}"

                        ),

                        class_="card"

                    )

                ),

                ui.column(

                    3,

                    ui.div(

                        ui.h5("Average Daily Traffic"),

                        ui.h2(f"{global_avg_daily:,}"),

                        ui.p("Mean across municipalities"),

                        class_="card"

                    )

                ),

                ui.column(

                    3,

                    ui.div(

                        ui.h5("Peak Day"),

                        ui.h3(

                            peak_day.strftime("%d %B %Y")

                        ),

                        ui.p(f"{peak_count:,} cyclists"),

                        class_="card"

                    )

                ),

                ui.column(

                    3,

                    ui.div(

                        ui.h5("Highest Avg Daily Traffic"),

                        ui.h3(top_municipality),

                        ui.p(f"{top_avg:,} cyclists/day"),

                        class_="card"

                    )

                )

            ),

            ui.navset_tab(

                ui.nav_panel(

                    "Executive Summary",

                    output_widget("overview_plot")

                ),

                ui.nav_panel(

                    "Weather Impact",

                    output_widget("weather_corr_plot"),

                    output_widget("temperature_plot"),

                    output_widget("rain_plot"),

                    output_widget("sunshine_plot"),

                    output_widget("wind_plot")

                ),

                ui.nav_panel(

                    "Temporal Behaviour",

                    output_widget("weekday_plot"),

                    output_widget("monthly_plot"),

                    output_widget("yearly_plot")

                ),

                ui.nav_panel(

                    "Hourly Behaviour",

                    output_widget("hourly_curve_plot"),

                    output_widget("hourly_heatmap_plot")

                ),

                ui.nav_panel(

                    "Spatial Analysis",

                    output_widget("interactive_map_plot")

                ),

                ui.nav_panel(

                    "Model Performance",

                    output_widget("prediction_plot"),

                    output_widget("residual_plot"),

                    output_widget("error_distribution_plot"),

                    output_widget("model_error_plot")

                ),

                ui.nav_panel(

                    "Feature Importance",

                    output_widget("importance_plot")

                )

            )

        )

    )

)

# =========================================================

# SERVER

# =========================================================

def server(input, output, session):

    @reactive.calc

    def selected_daily():

        if input.municipality() == "All":

            return municipality_daily

        return municipality_daily[

            municipality_daily["municipality"]

            == input.municipality()

        ]

    @render_widget

    def overview_plot():

        df = selected_daily()

        if df.empty:

            return empty_figure(

                "No executive summary data available"

            )

        daily_plot = (

            df.groupby("datetime")[["daily_count"]]

            .sum()

            .reset_index()

            .sort_values("datetime")

        )

        daily_plot["rolling_avg"] = (

            daily_plot["daily_count"]

            .rolling(30, min_periods=1)

            .mean()

        )

        fig = go.Figure()

        fig.add_trace(

            go.Scatter(

                x=daily_plot["datetime"].dt.strftime("%Y-%m-%d"),

                y=daily_plot["daily_count"],

                mode="lines",

                line=dict(

                    color="#95d5b2",

                    width=1

                ),

                name="Daily Traffic"

            )

        )

        fig.add_trace(

            go.Scatter(

                x=daily_plot["datetime"].dt.strftime("%Y-%m-%d"),

                y=daily_plot["rolling_avg"],

                mode="lines",

                line=dict(

                    color="#1b4332",

                    width=4

                ),

                name="30-Day Average"

            )

        )

        fig.update_layout(

            title="Cycling Activity Over Time",

            xaxis_title="Date",

            yaxis_title="Cyclists Per Day",

            template="plotly_white",

            paper_bgcolor="#f7faf7",

            plot_bgcolor="white",

            height=650

        )

        return fig

    @render_widget

    def weather_corr_plot():

        df = weather_corr.copy()

        if input.municipality() != "All":

            df = df[

                df["municipality"]

                == input.municipality()

            ]

        fig = px.bar(

            df,

            x="correlation",

            y="weather_variable",

            orientation="h",

        )

        fig.update_traces(
            marker_color="#1b4332"
        )

        fig.update_layout(

            title=f"Weather Correlations ({input.municipality()})",

            template="plotly_white",

            paper_bgcolor="#f7faf7",

            height=450

        )

        return fig

    def weather_scatter(x_col, title, x_title):

        df = weather_merged.copy()

        if input.municipality() != "All":

            df = df[
                df["municipality"]
                == input.municipality()
            ]

        df[x_col] = pd.to_numeric(

            df[x_col],

            errors="coerce"

        )

        df["daily_count"] = pd.to_numeric(

            df["daily_count"],

            errors="coerce"

        )

        df = df.dropna(

            subset=[x_col, "daily_count"]

        )

        if len(df) > 5000:

            df = df.sample(

                5000,

                random_state=42

            )

        fig = px.scatter(

            df,

            x=x_col,

            y="daily_count",

            opacity=0.25

        )

        fig.update_traces(

            marker=dict(

                color="#2d6a4f",

                size=6

            )

        )

        fig.update_layout(

            title=title,

            xaxis_title=x_title,

            yaxis_title="Daily Cyclists",

            template="plotly_white",

            paper_bgcolor="#f7faf7",

            plot_bgcolor="white",

            height=500

        )

        return fig

    @render_widget

    def temperature_plot():

        return weather_scatter(

            "temperature",

            "Temperature vs Cycling",

            "Mean Daily Temperature (°C)"

        )

    @render_widget

    def rain_plot():

        return weather_scatter(

            "precipitation",

            "Rainfall vs Cycling",

            "Daily Rainfall (mm)"

        )

    @render_widget

    def sunshine_plot():

        df = weather_merged.copy()

        if input.municipality() != "All":

            df = df[
                df["municipality"]
                == input.municipality()
            ]

        df["sunshine_hours"] = (

            pd.to_numeric(

                df["sunshine_duration"],

                errors="coerce"

            ) / 3600

        )

        df["daily_count"] = pd.to_numeric(

            df["daily_count"],

            errors="coerce"

        )

        df = df.dropna(

            subset=["sunshine_hours", "daily_count"]

        )

        if len(df) > 5000:

            df = df.sample(

                5000,

                random_state=42

            )

        fig = px.scatter(

            df,

            x="sunshine_hours",

            y="daily_count",

            opacity=0.25

        )

        fig.update_traces(

            marker=dict(

                color="#2d6a4f",

                size=6

            )

        )

        fig.update_layout(

            title="Sunshine vs Cycling",

            xaxis_title="Sunshine Duration (hours)",

            yaxis_title="Daily Cyclists",

            template="plotly_white",

            paper_bgcolor="#f7faf7",

            plot_bgcolor="white",

            height=500

        )

        return fig

    @render_widget

    def wind_plot():

        return weather_scatter(

            "wind_speed",

            "Wind Speed vs Cycling",

            "Maximum Daily Wind Speed (km/h)"

        )

    @render_widget

    def weekday_plot():

        df = weekday_summary.copy()

        if input.municipality() != "All":

            df = df[

                df["municipality"]

                == input.municipality()

            ]

        df = (

            df.groupby(

                "weekday",

                as_index=False

            )["daily_count"]

            .mean()

        )

        order = [

            "Monday",

            "Tuesday",

            "Wednesday",

            "Thursday",

            "Friday",

            "Saturday",

            "Sunday"

        ]

        df["weekday"] = pd.Categorical(

            df["weekday"],

            categories=order,

            ordered=True

        )

        df = df.sort_values("weekday")


        fig = px.bar(
            df,
            x="weekday",
            y="daily_count"
        )

        fig.update_traces(
            marker_color="#1b4332"
        )

        fig.update_layout(

            title=f"Average Cycling Activity by Weekday ({input.municipality()})",

            template="plotly_white",

            paper_bgcolor="#f7faf7",

            height=550

        )

        return fig

    @render_widget

    def monthly_plot():

        df = monthly_summary.copy()

        if input.municipality() != "All":

            df = df[

                df["municipality"]

                == input.municipality()

            ]

        df = (

            df.groupby(

                "month",

                as_index=False

            )["daily_count"]

            .mean()

        )

        order = [

            "January",

            "February",

            "March",

            "April",

            "May",

            "June",

            "July",

            "August",

            "September",

            "October",

            "November",

            "December"

        ]

        df["month"] = pd.Categorical(

            df["month"],

            categories=order,

            ordered=True

        )

        df = df.sort_values("month")

        fig = px.bar(

            df,

            x="month",

            y="daily_count",

        )

        fig.update_traces(
            marker_color="#1b4332"
        )

        fig.update_layout(

            title=f"Average Cycling Activity by Month ({input.municipality()})",

            template="plotly_white",

            paper_bgcolor="#f7faf7",

            height=550

        )

        return fig

    @render_widget

    def yearly_plot():

        df = yearly_summary.copy()

        if input.municipality() != "All":

            df = df[

                df["municipality"]

                == input.municipality()

            ]

        df = (

            df.groupby(

                "year",

                as_index=False

            )["daily_count"]

            .mean()

        )

        fig = px.line(

            df,

            x="year",

            y="daily_count",

            markers=True

        )

        fig.update_traces(

            line_color="#1b4332",

            marker_color="#1b4332"

        )

        fig.update_layout(

            title=f"Average Cycling Activity by Year ({input.municipality()})",

            template="plotly_white",

            paper_bgcolor="#f7faf7",

            height=550

        )

        return fig

    @render_widget
    def hourly_curve_plot():

        df = hourly_curve_summary.copy()

        if (
            "municipality" in df.columns
            and input.municipality() != "All"
        ):

            df = df[
                df["municipality"]
                == input.municipality()
            ]

        if df.empty:

            return empty_figure(
                "No hourly data available"
            )

        df = (
            df.groupby("hour", as_index=False)[
                "avg_cyclists"
            ].mean()
        )

        fig = px.line(

            df,

            x="hour",
            y="avg_cyclists",

            markers=True
        )

        fig.update_traces(

            line_color="#1b4332",

            marker_color="#1b4332"

        )

        fig.update_layout(

            title=f"Hourly Cycling Behaviour ({input.municipality()})",

            template="plotly_white",

            paper_bgcolor="#f7faf7",

            height=500
        )

        return fig

    @render_widget
    def hourly_heatmap_plot():

        df = hourly_heatmap_summary.copy()

        if (
            "municipality" in df.columns
            and input.municipality() != "All"
        ):

            df = df[
                df["municipality"]
                == input.municipality()
            ]

        if df.empty:

            return empty_figure(
                "No hourly heatmap data available"
            )

        heat = df.pivot_table(

            index="weekday",

            columns="hour",

            values="avg_cyclists",

            aggfunc="mean"
        )

        order = [

            "Monday",
            "Tuesday",
            "Wednesday",
            "Thursday",
            "Friday",
            "Saturday",
            "Sunday"

        ]

        heat = heat.reindex(order)

        fig = px.imshow(

            heat,

            color_continuous_scale="Greens",

            aspect="auto"
        )

        fig.update_layout(

            title=f"Weekday × Hour Heatmap ({input.municipality()})",

            template="plotly_white",

            paper_bgcolor="#f7faf7",

            height=650
        )

        return fig

    @render_widget

    def interactive_map_plot():

        return STATIC_SPATIAL_FIG

    @render_widget

    def prediction_plot():

        return STATIC_PREDICTION_FIG

    @render_widget

    def residual_plot():

        return STATIC_RESIDUAL_FIG

    @render_widget

    def error_distribution_plot():

        return STATIC_ERROR_DIST_FIG

    @render_widget

    def model_error_plot():

        return STATIC_MODEL_ERROR_FIG

    @render_widget

    def importance_plot():

        return STATIC_IMPORTANCE_FIG

# =========================================================

# APP

# =========================================================

app = App(app_ui, server)
