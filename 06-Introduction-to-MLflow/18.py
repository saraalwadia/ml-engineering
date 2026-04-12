"""
  # Set the model_evaluation entry point
  model_evaluation:
    parameters:
      # Set run_id parameter
      run_id:
        type: str 
        default: None
    # Set the parameters in the command
    command: "python3.9 evaluate.py {run_id}"
"""
