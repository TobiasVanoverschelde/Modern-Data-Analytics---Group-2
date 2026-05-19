import pandas as pd
from sklearn.inspection import permutation_importance


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
