import tkinter as tk
from tkinter import ttk
import numpy as np
import pandas as pd
import scipy
from Gaussienmethod import gaussian_elimination
def load_data():
    data = pd.read_csv("diabetes.csv")
    X = data.drop('Outcome', axis=1).values
    y = data['Outcome'].values.reshape(-1, 1)
    X = np.hstack((np.ones((X.shape[0], 1)), X))
    return X, y

def gaussian_elimination_solver(A, b):
    coefficients = gaussian_elimination(A, b)
    return coefficients

def lu_decomposition_solver(A):
    P, L, U = scipy.linalg.lu(A)
    return L, U

def solve_lu(L, U, b):
    y = np.linalg.solve(L, b)
    x = np.linalg.solve(U, y)
    return x

def gauss_seidel_solver(A, b, max_iterations=1000, tolerance=1e-6):
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

def jacobi_solver(A, b, max_iterations=1000, tolerance=1e-6):
    n = len(b)
    x = np.zeros_like(b)
    x_new = np.zeros_like(x)

    for _ in range(max_iterations):
        for i in range(n):
            sigma = 0
            for j in range(n):
                if j != i:
                    sigma += A[i, j] * x[j]
            x_new[i] = (b[i] - sigma) / A[i, i]

        if np.allclose(x, x_new, atol=tolerance):
            break

        x = np.copy(x_new)

    return x

def logistic_regression(X, y, num_iterations=1000, learning_rate=0.1):
    m, n = X.shape
    theta = np.zeros((n, 1))

    for _ in range(num_iterations):
        z = np.dot(X, theta)
        h = 1 / (1 + np.exp(-z))
        gradient = np.dot(X.T, (h - y)) / m
        theta -= learning_rate * gradient

    return theta

def predict(X, coefficients):
    z = np.dot(X, coefficients)
    h = 1 / (1 + np.exp(-z))
    return (h >= 0.5).astype(int)

def calculate_accuracy(X_test, y_test, coefficients):
    predictions = predict(X_test, coefficients)
    accuracy = np.mean(predictions == y_test)
    return accuracy

def train_test_split(X, y, test_size=0.2, random_seed=10): 
    np.random.seed(random_seed)
    n = len(X)
    indices = np.random.permutation(n)
    split = int(n * (1 - test_size))
    train_indices, test_indices = indices[:split], indices[split:]
    return X[train_indices], X[test_indices], y[train_indices], y[test_indices]


def run_method():
    method = method_var.get()
    X_train, X_test, y_train, y_test = train_test_split(X, y,random_seed=10)  # Fixing random seed     X_train, X_test, y_train, y_test = train_test_split(X, y,random_seed=10)
    if method == "Gaussian Elimination":
        coefficients = gaussian_elimination_solver(np.dot(X_train.T, X_train), np.dot(X_train.T, y_train))
    elif method == "LU Decomposition":
        L, U = lu_decomposition_solver(np.dot(X_train.T, X_train))
        coefficients = solve_lu(L, U, np.dot(X_train.T, y_train))
    elif method == "Gauss-Seidel Method":
        coefficients = gauss_seidel_solver(np.dot(X_train.T, X_train), np.dot(X_train.T, y_train))
    elif method == "Jacobi Method":
        coefficients = jacobi_solver(np.dot(X_train.T, X_train), np.dot(X_train.T, y_train))
    accuracy = calculate_accuracy(X_test, y_test, coefficients)
    accuracy_label.config(text=f"Accuracy: {accuracy}")


X, y = load_data()

root = tk.Tk()
root.title("Diabetes Prediction")

method_var = tk.StringVar()
method_label = ttk.Label(root, text="Select Method:")
method_label.grid(row=0, column=0, padx=10, pady=5, sticky="e")
method_combobox = ttk.Combobox(root, textvariable=method_var, values=["Gaussian Elimination", "LU Decomposition", "Gauss-Seidel Method", "Jacobi Method"])
method_combobox.grid(row=0, column=1, padx=10, pady=5, sticky="w")
method_combobox.current(0)

run_button = ttk.Button(root, text="Run Method", command=run_method)
run_button.grid(row=1, column=0, columnspan=2, pady=10)

accuracy_label = ttk.Label(root, text="Accuracy: ")
accuracy_label.grid(row=2, column=0, columnspan=2)

root.mainloop()
