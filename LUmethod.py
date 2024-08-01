import numpy as np
import pandas as pd

# Read the data from CSV file
data = pd.read_csv("diabetes.csv")

# Extract features and target
X = data.drop('Outcome', axis=1).values
y = data['Outcome'].values.reshape(-1, 1)

# Add bias term
X = np.hstack((np.ones((X.shape[0], 1)), X))

# LU decomposition without using numpy's linalg.lu function
def lu_decomposition(A):
    n = len(A)
    L = np.eye(n)
    U = np.zeros((n, n))

    for i in range(n):
        # Upper triangular matrix
        for j in range(i, n):
            U[i, j] = A[i, j] - np.dot(L[i, :i], U[:i, j])

        # Lower triangular matrix
        for j in range(i+1, n):
            L[j, i] = (A[j, i] - np.dot(L[j, :i], U[:i, i])) / U[i, i]

    return L, U

# Solve linear system using LU decomposition
def solve_lu(L, U, b):
    # Solve Ly = b
    n = len(L)
    y = np.zeros(n)
    for i in range(n):
        y[i] = b[i] - np.dot(L[i, :i], y[:i])

    # Solve Ux = y
    x = np.zeros(n)
    for i in range(n - 1, -1, -1):
        x[i] = (y[i] - np.dot(U[i, i+1:], x[i+1:])) / U[i, i]

    return x

# Logistic regression function
def logistic_regression(X, y, num_iterations=1000, learning_rate=0.1):
    m, n = X.shape
    theta = np.zeros((n, 1))

    for _ in range(num_iterations):
        # Compute hypothesis
        z = np.dot(X, theta)
        h = 1 / (1 + np.exp(-z))

        # Compute gradient
        gradient = np.dot(X.T, (h - y)) / m

        # Update parameters
        theta -= learning_rate * gradient

    return theta

# Split data into training and testing sets
def train_test_split(X, y, test_size=0.2):
    n = len(X)
    indices = np.random.permutation(n)
    split = int(n * (1 - test_size))
    train_indices, test_indices = indices[:split], indices[split:]
    return X[train_indices], X[test_indices], y[train_indices], y[test_indices]

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y)

# LU decomposition
L, U = lu_decomposition(np.dot(X_train.T, X_train))

# Solve for coefficients
coefficients = solve_lu(L, U, np.dot(X_train.T, y_train))

# Logistic regression
theta = logistic_regression(X_train, y_train)

# Predict
def predict(X, theta):
    z = np.dot(X, theta)
    h = 1 / (1 + np.exp(-z))
    return (h >= 0.5).astype(int)

# Test accuracy
predictions = predict(X_test, theta)
accuracy = np.mean(predictions == y_test)
print("Accuracy:", accuracy)
