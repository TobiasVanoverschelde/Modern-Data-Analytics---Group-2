import pandas as pd


class ModelExplainer:

    def get_tree_feature_importance(
        self,
        model,
        feature_names
    ):

        if not hasattr(
            model,
            "feature_importances_"
        ):

            return pd.DataFrame({
                "feature": [],
                "importance": []
            })

        importance = pd.DataFrame({
            "feature": feature_names,
            "importance": model.feature_importances_
        })

        importance = importance.sort_values(
            "importance",
            ascending=False
        )

        return importance
