# Register the first (2022) model
mlflow.register_model("model_2022", "Insurance")

# Register the second (2023) model
mlflow.register_model(f"runs:/{run_id}/model_2023", "Insurance")