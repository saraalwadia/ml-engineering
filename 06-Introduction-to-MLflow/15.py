import mlflow

# Run the MLflow Project
mlflow.projects.run(
    uri='./',
    entry_point='main',
    experiment_name='Insurance',
    env_manager='local',
    # Set parameters for n_jobs and fit_intercept
    parameters={
        'n_jobs_param': 2,
        'fit_intercept_param': False
    }
)