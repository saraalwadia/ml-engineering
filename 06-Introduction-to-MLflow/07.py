# Create Python Class
class CustomPredict(mlflow.pyfunc.PythonModel):
    
    # Set method for loading model
    def load_context(self, context):
        self.model = mlflow.sklearn.load_model("./lr_model/")
    
    # Set method for custom inference     
    def predict(self, context, model_input):
        predictions = self.model.predict(model_input)
        decoded_predictions = []  
        for prediction in predictions:
            if prediction == 0:
                decoded_predictions.append("female")
            else:
                decoded_predictions.append("male")
        return decoded_predictions


# Log the pyfunc model 
mlflow.pyfunc.log_model(
    artifact_path="lr_pyfunc", 
    # Set model to use CustomPredict Class
    python_model=CustomPredict(), 
    artifacts={"lr_model": "lr_model"}
)

# Get the last run
run = mlflow.last_active_run()
run_id = run.info.run_id

# Load the model in python_function format
loaded_model = mlflow.pyfunc.load_model(f"runs:/{run_id}/lr_pyfunc")