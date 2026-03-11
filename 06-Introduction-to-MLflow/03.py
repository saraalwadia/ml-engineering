import mlflow

filter_str = "metrics.f1_score > 0.6"
runs_df = mlflow.search_runs(
    experiment_names=["Insurance Experiment"],
    filter_string=filter_str,
    order_by=["metrics.precision_score DESC"]
)

print(runs_df)