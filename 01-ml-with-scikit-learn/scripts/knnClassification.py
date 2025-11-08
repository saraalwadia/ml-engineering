from sklearn.neighbors import KNeighborsClassifier
import numpy as np

# Features (study hours, sleep hours)
X = np.array([[8, 6], [7, 5], [3, 8], [4, 9]])  

# Labels (pass or fail)
y = np.array(['Pass', 'Pass', 'Fail', 'Fail'])  

# Create the KNN model with k = 3
knn = KNeighborsClassifier(n_neighbors=3)

# Train (fit) the model on the data
knn.fit(X, y)

# New student data: study hours = 6, sleep hours = 6.5
new_student = np.array([[6, 6.5]])

# Predict the class for the new student
prediction = knn.predict(new_student)

print("Model prediction:", prediction[0])
