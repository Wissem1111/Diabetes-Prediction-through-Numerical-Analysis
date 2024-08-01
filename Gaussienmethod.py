import numpy as np
import pandas as pd

# Read the data from CSV file
data = pd.read_csv("diabetes.csv")

# Extract features and target
X = data.drop('Outcome', axis=1).values
y = data['Outcome'].values.reshape(-1, 1)

# Add bias term
X = np.hstack((np.ones((X.shape[0], 1)), X))

# Gaussian Elimination with Partial Pivoting method
def gaussian_elimination(A, b):
    n = len(b)
    for i in range(n):
        # Partial pivoting
        max_index = np.argmax(abs(A[i:, i])) + i
        A[[i, max_index]] = A[[max_index, i]]
        b[[i, max_index]] = b[[max_index, i]]
        
        # Elimination
        for j in range(i+1, n):
            ratio = A[j, i] / A[i, i]
            A[j, i:] -= ratio * A[i, i:]
            b[j] -= ratio * b[i]

    # Back substitution
    x = np.zeros(n)
    for i in range(n - 1, -1, -1):
        x[i] = (b[i] - np.dot(A[i, i+1:], x[i+1:])) / A[i, i]

    return x

# Split data into training and testing sets
def train_test_split(X, y, test_size=0.2):
    n = len(X)
    indices = np.random.permutation(n)
    split = int(n * (1 - test_size))
    train_indices, test_indices = indices[:split], indices[split:]
    return X[train_indices], X[test_indices], y[train_indices], y[test_indices]

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y)

# Solve for coefficients using Gaussian Elimination with Partial Pivoting
coefficients = gaussian_elimination(np.dot(X_train.T, X_train), np.dot(X_train.T, y_train))

# Predict
def predict(X, coefficients):
    z = np.dot(X, coefficients)
    h = 1 / (1 + np.exp(-z))
    return (h >= 0.5).astype(int)

# Test accuracy
predictions = predict(X_test, coefficients)
accuracy = np.mean(predictions == y_test)
print("Accuracy:", accuracy)

