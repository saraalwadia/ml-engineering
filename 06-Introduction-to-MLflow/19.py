# Set run method to model_engineering
model_engineering = mlflow.projects.run(
    uri='./',
    # Set entry point to model_engineering
    entry_point='model_engineering',
    experiment_name='Insurance',
    # Set the parameters for n_jobs and fit_intercept
    parameters={
        'n_jobs': 2, 
        'fit_intercept': False
    },
    env_manager='local'
)

# Set Run ID of model training to be passed to Model Evaluation step
model_engineering_run_id = model_engineering.run_id
print(model_engineering_run_id)