# Create a filter string for R-squared score
r_squared_filter = "metrics.r2_score > .70"

# Search runs
mlflow.search_runs(experiment_names=["Unicorn Sklearn Experiments", "Unicorn Other Experiments"], 
                   filter_string=r_squared_filter, 
                   order_by=["metrics.r2_score DESC"])