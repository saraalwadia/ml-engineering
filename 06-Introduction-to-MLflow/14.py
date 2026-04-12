"""
name: insurance_model
python_env: python_env.yaml
entry_points:
  main:
    parameters:
      # Parameter for number of jobs
      n_jobs:
        type: int
        default: 1
      # Parameter for fit_intercept
      fit_intercept:
        type: bool
        default: True
    command: "python3.9 train_model.py {n_jobs} {fit_intercept}"
    """