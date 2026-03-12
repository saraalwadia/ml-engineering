# Eval Data
eval_data = X_test
eval_data["sex"] = y_test

# Log the lr_class model using Scikit-Learn Flavor
mlflow.sklearn.log_model(lr_class, "model")

# Get run id
run = mlflow.last_active_run()
run_id = run.info.run_id

# Evaluate the logged model with eval_data data
mlflow.evaluate(
    f"runs:/{run_id}/model", 
    data=eval_data, 
    targets="sex",
    model_type="classifier"
)