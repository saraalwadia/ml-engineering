# Run this cell to import the modules you require
import pandas as pd
import numpy as np
import mlflow
import mlflow.sklearn
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor

# Read in the data
weather = pd.read_csv("london_weather.csv")

# Start coding here
# Use as many cells as you like

# Drop missing values
weather = weather.dropna()

# Convert date
weather["date"] = pd.to_datetime(weather["date"], format="%Y%m%d")

# Feature engineering
weather["year"] = weather["date"].dt.year
weather["month"] = weather["date"].dt.month
weather["day"] = weather["date"].dt.day

# Extra useful feature
weather["temp_range"] = weather["max_temp"] - weather["min_temp"]

# Define target
y = weather["mean_temp"]

# Define features
X = weather.drop(columns=["mean_temp", "date"])

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

mlflow.set_experiment("london_weather_experiments")

with mlflow.start_run():
    model = LinearRegression()
    model.fit(X_train_scaled, y_train)

    preds = model.predict(X_test_scaled)
    rmse = np.sqrt(mean_squared_error(y_test, preds))

    mlflow.log_param("model", "LinearRegression")
    mlflow.log_metric("rmse", rmse)

    mlflow.sklearn.log_model(model, "model")

for depth in [5, 10, 20]:
    with mlflow.start_run():
        model = DecisionTreeRegressor(max_depth=depth, random_state=42)
        model.fit(X_train, y_train)

        preds = model.predict(X_test)
        rmse = np.sqrt(mean_squared_error(y_test, preds))

        mlflow.log_param("model", "DecisionTree")
        mlflow.log_param("max_depth", depth)
        mlflow.log_metric("rmse", rmse)

        mlflow.sklearn.log_model(model, "model")


for n in [100, 200]:
    for depth in [10, 20, None]:
        with mlflow.start_run():
            model = RandomForestRegressor(
                n_estimators=n,
                max_depth=depth,
                random_state=42
            )

            model.fit(X_train, y_train)

            preds = model.predict(X_test)
            rmse = np.sqrt(mean_squared_error(y_test, preds))

            mlflow.log_param("model", "RandomForest")
            mlflow.log_param("n_estimators", n)
            mlflow.log_param("max_depth", depth)
            mlflow.log_metric("rmse", rmse)

            mlflow.sklearn.log_model(model, "model")

experiment_results = mlflow.search_runs()