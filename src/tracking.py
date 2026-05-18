import mlflow
import mlflow.sklearn

from src.modeling import fit_with_time_cv, evaluate


def setup_mlflow(experiment_name="cycling-flanders", tracking_uri="file:./mlruns"):
    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment(experiment_name)
    print(f"MLflow tracking URI: {tracking_uri}")
    print(f"Experiment: {experiment_name}")


def fit_with_tracking(pipeline, X_train, y_train, X_test, y_test,
                      param_grid, model_name,
                      registered_model_name="cycling-flanders"):
    
    with mlflow.start_run(run_name=model_name):
        gs = fit_with_time_cv(pipeline, X_train, y_train, param_grid)

        # Hyperparameters best estimator
        for param, value in gs.best_params_.items():
            mlflow.log_param(param, value)

        # Cross-validation score
        mlflow.log_metric("cv_mae", -gs.best_score_)

        # Test set metrics
        test_metrics = evaluate(gs.best_estimator_, X_test, y_test)
        for name, value in test_metrics.items():
            mlflow.log_metric(f"test_{name.lower()}", value)

        # Log fitted pipeline as MLflow Model + register in Model Registry
        mlflow.sklearn.log_model(
            gs.best_estimator_,
            artifact_path="model",
            registered_model_name=registered_model_name,
        )

        # Tag the run for easy filtering
        mlflow.set_tag("model_family", model_name)
        mlflow.set_tag("n_train_rows", len(X_train))

        print(f"Logged run '{model_name}' to MLflow")
        return gs, test_metrics