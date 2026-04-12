"""
name: insurance_model
python_env: python_env.yaml
entry_points:
  # Set the entry point
  model_engineering:
    parameters: 
      # Set n_jobs 
      n_jobs:
        type: int
        default: 1
      # Set fit_intercept
      fit_intercept:
        type: bool
        default: True
    # Pass the parameters to the command
    command: "python3.9 train_model.py {n_jobs} {fit_intercept}"
"""