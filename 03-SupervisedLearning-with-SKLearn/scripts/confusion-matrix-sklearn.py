# Import libraries
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns
import matplotlib.pyplot as plt

# Example: True labels vs Predicted labels
y_true = [0, 0, 0, 1, 1, 1, 0, 0, 1, 0]  # 0 = Legitimate, 1 = Fraudulent
y_pred = [0, 0, 0, 1, 0, 1, 0, 0, 1, 0]

# 1️⃣ Compute Confusion Matrix
cm = confusion_matrix(y_true, y_pred)
print("Confusion Matrix:\n", cm)

# 2️⃣ Visualize Confusion Matrix
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=['Legitimate', 'Fraudulent'],
            yticklabels=['Legitimate', 'Fraudulent'])
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix')
plt.show()

# 3️⃣ Detailed Classification Report (Precision, Recall, F1-score)
report = classification_report(y_true, y_pred, target_names=['Legitimate', 'Fraudulent'])
print("Classification Report:\n", report)
