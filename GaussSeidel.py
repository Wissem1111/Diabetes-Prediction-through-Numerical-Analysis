import numpy as np
import pandas as pd

# Read the data from CSV file
data = pd.read_csv("diabetes.csv")

# Extract features and target
X = data.drop('Outcome', axis=1).values
y = data['Outcome'].values.reshape(-1, 1)

# Add bias term
X = np.hstack((np.ones((X.shape[0], 1)), X))

# Define Gauss-Seidel method for solving linear system of equations
def gauss_seidel_method(A, b, max_iterations=1000, tolerance=1e-6):
    n = len(b)
    x = np.zeros_like(b)

    for _ in range(max_iterations):
        x_new = np.copy(x)
        for i in range(n):
            sigma = np.dot(A[i, :i], x_new[:i]) + np.dot(A[i, i+1:], x[i+1:])
            x_new[i] = (b[i] - sigma) / A[i, i]

        if np.allclose(x, x_new, atol=tolerance):
            break

        x = np.copy(x_new)

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

# Solve for coefficients using Gauss-Seidel method
coefficients = gauss_seidel_method(np.dot(X_train.T, X_train), np.dot(X_train.T, y_train))

# Predict
def predict(X, coefficients):
    z = np.dot(X, coefficients)
    h = 1 / (1 + np.exp(-z))
    return (h >= 0.5).astype(int)

# Test accuracy
predictions = predict(X_test, coefficients)
accuracy = np.mean(predictions == y_test)
print("Accuracy:", accuracy)
