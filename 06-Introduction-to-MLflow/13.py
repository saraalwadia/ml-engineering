import mlflow

mlflow.projects.run(
    uri="./",              # current working directory
    entry_point="main",    # main entry point from MLproject
    experiment_name="Insurance",  # experiment name for tracking
    env_manager="local",
    synchronous=True,
)