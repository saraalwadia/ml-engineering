# All required libraries are imported here for you.
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn import metrics

# Load the dataset
crops = pd.read_csv("soil_measures.csv")

# Write your code
X = crops[['N', 'P', 'K', 'ph']]
y = crops['crop']

best_score = 0
best_feature = None

for col in X.columns:
    X_train, X_test, y_train, y_test = train_test_split(X[[col]], y, test_size=0.3, random_state=42)
    
    model = LogisticRegression(max_iter=200)
    model.fit(X_train, y_train)
    
    y_pred = model.predict(X_test)
    score = metrics.accuracy_score(y_test, y_pred)
    
    if score > best_score:
        best_score = score
        best_feature = col

best_predictive_feature = {best_feature: best_score}

print(best_predictive_feature)