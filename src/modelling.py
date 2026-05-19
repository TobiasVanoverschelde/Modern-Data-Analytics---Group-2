import pandas as pd

from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression

from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score
)


class ModelTrainer:

    def prepare_features(
        self,
        df,
        target_col="daily_count"
    ):

        df = df.copy()

        df = df.dropna(
            subset=[target_col]
        )

        numeric_cols = [
            col for col in df.columns
            if pd.api.types.is_numeric_dtype(df[col])
        ]

        if target_col in numeric_cols:
            numeric_cols.remove(target_col)

        X = df[numeric_cols].copy()

        X = X.fillna(0)

        y = df[target_col].copy()

        self.feature_names = list(X.columns)

        return X, y

    def temporal_split(
        self,
        X,
        y,
        test_size=0.2
    ):

        split_index = int(
            len(X) * (1 - test_size)
        )

        split_index = max(
            1,
            split_index
        )

        split_index = min(
            split_index,
            len(X) - 1
        )

        X_train = X.iloc[:split_index]
        X_test = X.iloc[split_index:]

        y_train = y.iloc[:split_index]
        y_test = y.iloc[split_index:]

        return (
            X_train,
            X_test,
            y_train,
            y_test
        )

    def compare_models(
        self,
        X_train,
        X_test,
        y_train,
        y_test
    ):

        models = {
            "Linear Regression": LinearRegression(),
            "Random Forest": RandomForestRegressor(
                n_estimators=80,
                random_state=42,
                n_jobs=-1
            )
        }

        results = []

        for name, model in models.items():

            model.fit(
                X_train,
                y_train
            )

            preds = model.predict(
                X_test
            )

            mae = mean_absolute_error(
                y_test,
                preds
            )

            rmse = (
                mean_squared_error(
                    y_test,
                    preds
                ) ** 0.5
            )

            r2 = r2_score(
                y_test,
                preds
            )

            results.append({
                "model": name,
                "MAE": mae,
                "RMSE": rmse,
                "R2": r2
            })

        return pd.DataFrame(results)

    def train_best_model(
        self,
        X_train,
        y_train
    ):

        model = RandomForestRegressor(
            n_estimators=120,
            random_state=42,
            n_jobs=-1
        )

        model.fit(
            X_train,
            y_train
        )

        return model

    def generate_predictions(
        self,
        model,
        X_test,
        y_test
    ):

        preds = model.predict(
            X_test
        )

        predictions = pd.DataFrame({
            "actual": y_test.values,
            "predicted": preds
        })

        return predictions
