from sklearn.model_selection import cross_val_score, KFold
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
import matplotlib.pyplot as plt

# Create models dictionary
models = {
    "Logistic Regression": LogisticRegression(),
    "KNN": KNeighborsClassifier(),
    "Decision Tree Classifier": DecisionTreeClassifier()
}

results = []

# Loop through the models' values
for model in models.values():
    
    # Instantiate a KFold object
    kf = KFold(n_splits=6, random_state=12, shuffle=True)
    
    # Perform cross-validation
    cv_results = cross_val_score(model, X_train_scaled, y_train, cv=kf)
    
    # Append the results
    results.append(cv_results)

# Create a box plot of the results
plt.boxplot(results, labels=models.keys())
plt.ylabel("Accuracy")
plt.title("Cross-Validation Results for Classification Models")
plt.show()
