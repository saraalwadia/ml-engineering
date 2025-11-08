from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report, confusion_matrix

# 1 Load the Iris dataset
iris = load_iris()
X = iris.data         # 4 features: sepal length, sepal width, petal length, petal width
y = iris.target       # 3 classes: 0=setosa, 1=versicolor, 2=virginica

# 2 Split the data: 70% train, 30% test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# 3 Standardize features (Normalization)
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# 4 Create the KNN model with k=5
knn = KNeighborsClassifier(n_neighbors=5)

# 5 Train (fit) the model on the training data
knn.fit(X_train, y_train)

# 6 Predict the classes on the test data
y_pred = knn.predict(X_test)

# 7 Evaluate the model
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))
