# Import necessary libraries
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score


# 1. Load the MNIST dataset
print("Loading MNIST dataset...")
mnist = fetch_openml('mnist_784', version=1, as_frame=False, parser='auto')
X, y = mnist.data, mnist.target

# Convert labels to integers (they are loaded as strings)
y = y.astype(np.int8)

#   Explore and preprocess
# Normalize pixel values from [0, 255] to [0, 1] to improve SVM convergence
X = X / 255.0

# Split into training (60k) and test (10k) sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=10000, random_state=42, stratify=y
)
# SVMs are sensitive to feature scales; standardization improves performance.
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

#  Model selection and training

print("Training SVM classifier...")
svm_clf = SVC(kernel='rbf', C=5, gamma='scale', random_state=42)
svm_clf.fit(X_train_scaled, y_train)

#  Evaluation on the test set
y_pred = svm_clf.predict(X_test_scaled)
accuracy = accuracy_score(y_test, y_pred)
print(f"Test accuracy: {accuracy * 100:.2f}%")

# Detailed performance per digit
print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# Confusion matrix to visualize misclassifications
conf_mat = confusion_matrix(y_test, y_pred)
print("Confusion Matrix:")
print(conf_mat)
