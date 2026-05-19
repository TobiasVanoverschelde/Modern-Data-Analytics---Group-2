import pandas as pd
from sklearn.inspection import partial_dependence, permutation_importance


def compute_permutation_importance(model, X_test, y_test,
                                   n_repeats=5, sample_size=5000,
                                   random_state=42):
    """Robust feature importance by shuffling one feature at a time and
    measuring the drop in R². Unbiased w.r.t. cardinality, unlike the
    tree-built-in importance which inflates one-hot encoded categoricals
    such as gemeente.

    Operates on the raw feature space, so importances are reported per
    logical feature instead of per one-hot column.
    """
    n_sample = min(sample_size, len(X_test))
    sample = X_test.sample(n=n_sample, random_state=random_state)
    sample_y = y_test.loc[sample.index]

    result = permutation_importance(
        model, sample, sample_y,
        n_repeats=n_repeats,
        random_state=random_state,
        n_jobs=-1,
    )

    return (
        pd.DataFrame({
            "feature": X_test.columns,
            "importance_mean": result.importances_mean,
            "importance_std": result.importances_std,
        })
        .sort_values("importance_mean", ascending=False)
        .reset_index(drop=True)
    )


def compute_pdp(model, X, features, sample_size=5000, grid_resolution=50,
                random_state=42):
    """Partial dependence per feature: vary one feature across its range
    while averaging over the observed distribution of the others, to see
    the marginal effect on the prediction.

    Directly answers proposal question 3 ("how much does a degree of
    temperature change affect cycling counts"). Returns a long-format
    DataFrame with feature / grid_value / prediction columns so it can
    be stored as parquet and consumed by the dashboard.
    """
    n_sample = min(sample_size, len(X))
    sample = X.sample(n=n_sample, random_state=random_state)

    rows = []
    for feat in features:
        pdp = partial_dependence(
            model, sample, features=[feat],
            kind="average", grid_resolution=grid_resolution,
        )
        # sklearn renamed "values" to "grid_values" in 1.5; support both.
        grid = (pdp.get("grid_values") or pdp.get("values"))[0]
        preds = pdp["average"][0]
        for g, p in zip(grid, preds):
            rows.append({
                "feature": feat,
                "grid_value": float(g),
                "prediction": float(p),
            })

    return pd.DataFrame(rows)
