# first_mlflow.py
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

import mlflow
import mlflow.sklearn
import os

sns.set(style="whitegrid")

# -------------------------
# Tracking URI & Experiment
# -------------------------
mlruns_path = os.path.join(os.getcwd(), "mlruns")  # مجلد mlruns في نفس مكان السكريبت
mlflow.set_tracking_uri("sqlite:///mlflow_db/mlflow.db")
mlflow.set_experiment("iris_experiment")

# -------------------------
#  Load Dataset
# -------------------------
iris = load_iris()
X = pd.DataFrame(iris.data, columns=iris.feature_names)
y = pd.Series(iris.target, name="target")

print("Data preview:")
print(X.head())
print("\nTarget distribution:")
print(y.value_counts())

# -------------------------
# Split Dataset
# -------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# -------------------------
# Feature Scaling
# -------------------------
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# -------------------------
# Logistic Regression
# -------------------------
with mlflow.start_run(run_name="Logistic_Regression"):
    mlflow.log_param("model_type", "LogisticRegression")
    mlflow.log_param("scaling", "StandardScaler")

    lr = LogisticRegression()
    lr.fit(X_train_scaled, y_train)
    y_pred_lr = lr.predict(X_test_scaled)

    acc_lr = accuracy_score(y_test, y_pred_lr)
    print("\nLogistic Regression Accuracy:", acc_lr)
    print(classification_report(y_test, y_pred_lr))

    # Log metrics
    mlflow.log_metric("accuracy", acc_lr)

    # Log model
    mlflow.sklearn.log_model(lr, "model")

    # Confusion Matrix plot
    cm = confusion_matrix(y_test, y_pred_lr)
    plt.figure(figsize=(5,5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
    plt.title("Confusion Matrix - Logistic Regression")
    cm_path = "cm_lr.png"
    plt.savefig(cm_path)
    mlflow.log_artifact(cm_path)
    plt.close()

# -------------------------
# Random Forest
# -------------------------
with mlflow.start_run(run_name="Random_Forest"):
    mlflow.log_param("model_type", "RandomForestClassifier")
    mlflow.log_param("n_estimators", 100)

    rf = RandomForestClassifier(n_estimators=100, random_state=42)
    rf.fit(X_train, y_train)
    y_pred_rf = rf.predict(X_test)

    acc_rf = accuracy_score(y_test, y_pred_rf)
    print("\nRandom Forest Accuracy:", acc_rf)
    print(classification_report(y_test, y_pred_rf))

    # Log metrics
    mlflow.log_metric("accuracy", acc_rf)

    # Log model
    mlflow.sklearn.log_model(rf, "model")

    # Confusion Matrix plot
    cm = confusion_matrix(y_test, y_pred_rf)
    plt.figure(figsize=(5,5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Greens")
    plt.title("Confusion Matrix - Random Forest")
    cm_path = "cm_rf.png"
    plt.savefig(cm_path)
    mlflow.log_artifact(cm_path)
    plt.close()

# -------------------------
# Compare Results
# -------------------------
results = pd.DataFrame({
    "Model": ["Logistic Regression", "Random Forest"],
    "Accuracy": [acc_lr, acc_rf]
})
print("\nComparison Results:")
print(results)