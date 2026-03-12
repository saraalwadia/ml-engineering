# Register the first (2022) model
mlflow.register_model("model_2022", "Insurance")

# Register the second (2023) model
mlflow.register_model(f"runs:/{run_id}/model_2023", "Insurance")

# Log the model using scikit-learn flavor
mlflow.sklearn.log_model(lr, "model", registered_model_name="Insurance")

insurance_filter_string = "name = 'Insurance'"

# Search for Insurance models
print(client.search_registered_models(filter_string=insurance_filter_string))