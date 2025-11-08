from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report, confusion_matrix
import pandas as pd

# 1 Create fruit dataset
data = {
    'Weight': [150, 170, 140, 130, 200, 210, 190, 180],
    'Color_score': [0.8, 0.7, 0.9, 0.85, 0.4, 0.5, 0.45, 0.55],
    'Label': ['Apple', 'Apple', 'Apple', 'Apple', 'Orange', 'Orange', 'Orange', 'Orange']
}

df = pd.DataFrame(data)

# 2 Split data into features (X) and labels (y)
X = df[['Weight', 'Color_score']]
y = df['Label']

# 3 Split data into training and test sets (70% train, 30% test)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# 4 Standardize features (important for KNN)
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# 5 Create KNN model with k=3
knn = KNeighborsClassifier(n_neighbors=3)

# 6 Train the model
knn.fit(X_train, y_train)

# 7 Predict on test data
y_pred = knn.predict(X_test)

# 8 Evaluate the model
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))
print("\nModel Accuracy:", knn.score(X_test, y_test))
