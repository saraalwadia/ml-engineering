# Import MLflow Client from MLflow module
from mlflow import MlflowClient

# Create an instance of MLflow Client Class named client
client = MlflowClient()

# Create new model
client.create_registered_model("Insurance")

# Insurance filter string
insurance_filter_string = "name LIKE 'Insurance%'"

# Search for Insurance models
print(client.search_registered_models(filter_string=insurance_filter_string))

# Not Insurance filter string
not_insurance_filter_string = "name != 'Insurance'"

# Search for non Insurance models
print(client.search_registered_models(filter_string=not_insurance_filter_string))