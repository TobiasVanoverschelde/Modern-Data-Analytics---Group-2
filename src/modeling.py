import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GridSearchCV, TimeSeriesSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

RANDOM_STATE = 42

NUMERIC_FEATURES = ["temperature_2m", "precipitation", "wind_speed_10m", "cloud_cover", "lat", "lon"]
CYCLICAL_FEATURES = ["day_of_week_sin", "day_of_week_cos", "month_sin", "month_cos"]
CATEGORICAL_FEATURES = ["gemeente", "covid_period"]
BOOL_FEATURES = ["is_weekend", "is_holiday"]

ALL_FEATURES = NUMERIC_FEATURES + CYCLICAL_FEATURES + CATEGORICAL_FEATURES + BOOL_FEATURES


def time_aware_split(df, cutoff_date, time_col="date"):
    cutoff = pd.Timestamp(cutoff_date)
    train = df[df[time_col] < cutoff].copy()
    test = df[df[time_col] >= cutoff].copy()

    if len(train) == 0 or len(test) == 0:
        raise ValueError(f"Empty train or test with cutoff {cutoff}")

    print(f"Cutoff: {cutoff.date()}")
    print(f"Train: {train[time_col].min().date()} -> {train[time_col].max().date()} ({len(train):,} rows)")
    print(f"Test:  {test[time_col].min().date()} -> {test[time_col].max().date()} ({len(test):,} rows)")
    return train, test


def build_preprocessor():
    return ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), NUMERIC_FEATURES + CYCLICAL_FEATURES),
            ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False),
             CATEGORICAL_FEATURES),
            ("bool", "passthrough", BOOL_FEATURES),
        ],
        remainder="drop",
    )


def build_pipeline(regressor):
    return Pipeline([
        ("preprocess", build_preprocessor()),
        ("regressor", regressor),
    ])


def evaluate(model, X_test, y_test):
    y_pred = model.predict(X_test)
    return {
        "MAE": mean_absolute_error(y_test, y_pred),
        "RMSE": np.sqrt(mean_squared_error(y_test, y_pred)),
        "R2": r2_score(y_test, y_pred),
    }


def fit_with_time_cv(pipeline, X_train, y_train, param_grid, n_splits=5):
    tscv = TimeSeriesSplit(n_splits=n_splits)
    gs = GridSearchCV(
        pipeline,
        param_grid=param_grid,
        cv=tscv,
        scoring="neg_mean_absolute_error",
        n_jobs=-1,
        verbose=1,
    )
    gs.fit(X_train, y_train)
    print(f"Best params: {gs.best_params_}")
    print(f"Best CV MAE: {-gs.best_score_:.2f}")
    return gs