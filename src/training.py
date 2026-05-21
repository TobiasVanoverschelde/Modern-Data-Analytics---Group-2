"""Headless training pipeline

This file exists to read the daily parquet, run the time aware split, trains multiple sklearn pipelines with TimeSeriesSplit CV, logs each one to MLflow, picks the best by test MAE and saves that best model as a pkl file + model_metadata

Runnable as a script too."""


from pathlib import Path
import json
import joblib
import pandas as pd

from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor, HistGradientBoostingRegressor

from src.modeling import time_aware_split, build_pipeline, ALL_FEATURES
from src.tracking import setup_mlflow, fit_with_tracking
from src.interpretation import compute_permutation_importance, compute_pdp

# Weather features we want PDP curves for in the dashboard
PDP_FEATURES = ["temperature_2m", "precipitation", "wind_speed_10m", "cloud_cover"]

PROCESSED_DIR = Path("data/processed")
RANDOM_STATE = 42
TEST_QUANTILE = 0.8
MLRUNS_URI = "file:./notebooks/mlruns"   # matches what app.py loads from

# Model families to compare. Same grids as notebook 03 cells 9-11.
MODEL_GRIDS = {
    "ridge": (
        Ridge(random_state=RANDOM_STATE),
        {"regressor__alpha": [0.1, 1.0, 10.0, 100.0]},
    ),
    "random_forest": (
        RandomForestRegressor(random_state=RANDOM_STATE, n_jobs=-1),
        {
            "regressor__n_estimators": [100],
            "regressor__max_depth": [15],
            "regressor__min_samples_leaf": [5],
        },
    ),
    "hist_gradient_boosting": (
        HistGradientBoostingRegressor(random_state=RANDOM_STATE),
        {
            "regressor__max_iter": [200, 500],
            "regressor__max_depth": [5, 7, None],
            "regressor__learning_rate": [0.05, 0.1],
        },
    ),
}


def load_daily() -> pd.DataFrame:
    df = pd.read_parquet(PROCESSED_DIR / "daily_for_modeling.parquet")
    return df


# Temporal split at the given date quantile. Returns X/y train+test + cutoff string.
def split(daily: pd.DataFrame, quantile: float = TEST_QUANTILE):
    daily = daily.sort_values("date").reset_index(drop=True)
    cutoff = pd.Timestamp(daily["date"].quantile(quantile)).strftime("%Y-%m-%d")
    train_df, test_df = time_aware_split(daily, cutoff_date=cutoff)
    X_train, y_train = train_df[ALL_FEATURES], train_df["count"]
    X_test, y_test = test_df[ALL_FEATURES], test_df["count"]
    return X_train, X_test, y_train, y_test, cutoff


# Fit every model in MODEL_GRIDS. Returns {name: {model, params, metrics}}.
def train_all(X_train, y_train, X_test, y_test) -> dict:
    results = {}
    for name, (regressor, grid) in MODEL_GRIDS.items():
        pipe = build_pipeline(regressor)
        gs, metrics = fit_with_tracking(
            pipe, X_train, y_train, X_test, y_test,
            grid, model_name=name,
        )
        results[name] = {
            "model": gs.best_estimator_,
            "params": gs.best_params_,
            "metrics": metrics,
        }
        print(f"  {name}: MAE={metrics['MAE']:.2f}  R2={metrics['R2']:.3f}")
    return results


# Lowest test MAE wins.
def pick_best(results: dict) -> tuple[str, dict]:
    name = min(results, key=lambda n: results[n]["metrics"]["MAE"])
    return name, results[name]


# Permutation importance + PDP. Saved so the dashboard reads parquet, not the model.
def save_interpretation(model, X_test, y_test) -> None:
    importance = (
        compute_permutation_importance(model, X_test, y_test)
        .rename(columns={"importance_mean": "importance"})
    )
    importance.to_parquet(PROCESSED_DIR / "feature_importance.parquet", index=False)

    pdp = compute_pdp(model, X_test, PDP_FEATURES)
    pdp.to_parquet(PROCESSED_DIR / "pdp_results.parquet", index=False)


# Write best_model.pkl + model_metadata.json.
def save_artifacts(name: str, result: dict, cutoff: str, n_train: int, n_test: int) -> None:
    joblib.dump(result["model"], PROCESSED_DIR / "best_model.pkl")
    metadata = {
        "model_name": name,
        "best_params": {k: str(v) for k, v in result["params"].items()},
        "test_metrics": result["metrics"],
        "features": ALL_FEATURES,
        "cutoff_date": cutoff,
        "n_train": n_train,
        "n_test": n_test,
    }
    (PROCESSED_DIR / "model_metadata.json").write_text(json.dumps(metadata, indent=2))


def train_and_save() -> None:
    setup_mlflow(tracking_uri=MLRUNS_URI)
    daily = load_daily()
    X_train, X_test, y_train, y_test, cutoff = split(daily)
    print(f"Cutoff: {cutoff} | Train: {len(X_train):,} | Test: {len(X_test):,}\n")
    results = train_all(X_train, y_train, X_test, y_test)
    name, best = pick_best(results)
    save_artifacts(name, best, cutoff, len(X_train), len(X_test))
    print(f"\nBest: {name} | MAE={best['metrics']['MAE']:.2f} R2={best['metrics']['R2']:.3f}")
    print("Computing permutation importance + PDP...")
    save_interpretation(best["model"], X_test, y_test)
    print(f"Saved best_model.pkl + model_metadata.json + interpretation parquets to {PROCESSED_DIR}/")


if __name__ == "__main__":
    train_and_save()
