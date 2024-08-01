import tkinter as tk
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
import scipy.linalg  # Import the scipy module for LU decomposition
import scipy
import sys
from sklearn.linear_model import LogisticRegression
from scipy.sparse.linalg import cg

# Interface Graphique 

def set_v(value):
    global v
    v = value
    root.destroy()  # Close the window when an option is clicked

root = tk.Tk()
root.title("Choisir une Méthode de Décomposition:")

# Set up button styles
button_style = {"font": ("Arial", 12), "padx": 60, "pady": 30}

# Create buttons
button1 = tk.Button(root, text="Pivot Du Gauss", command=lambda: set_v(1), **button_style)
button1.pack(side=tk.LEFT, padx=30, pady=50)

button2 = tk.Button(root, text="Méthode LU", command=lambda: set_v(2), **button_style)
button2.pack(side=tk.LEFT, padx=10)

button3 = tk.Button(root, text="Méthode QR", command=lambda: set_v(3), **button_style)
button3.pack(side=tk.LEFT, padx=10)
button4 = tk.Button(root, text="Descente de Coordonnées", command=lambda: set_v(4), **button_style)
button4.pack(side=tk.LEFT, padx=10)
button5 = tk.Button(root, text="Méthode QR", command=lambda: set_v(5), **button_style)
button5.pack(side=tk.LEFT, padx=30)

# Run the Tkinter event loop
v = 0
root.mainloop()

if v==0:
    print("Thanks for wasting my time.");
    sys.exit()


# DEBUT

# Load Titanic dataset
diabetes_df = pd.read_csv("diabetes.csv")

# Preprocess data
diabetes_df = diabetes_df.dropna()  # Remove rows with missing values
X = diabetes_df.drop(['Outcome'], axis=1)
y = diabetes_df['Outcome']
X = pd.get_dummies(X, drop_first=True)  # One-hot encoding for categorical variables
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Split data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Regularized linear regression using LU decomposition

def regularized_linear_regression(X, y, alpha=0.1):
  n, p = X.shape
  A = np.dot(X.T, X) + alpha * np.identity(p)  # Regularized normal equation matrix
  b = np.dot(X.T, y)
  if v==1:
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

  elif v==2:
       P, L, U = scipy.linalg.lu(A)
       x = np.linalg.solve(U, np.linalg.solve(L, np.dot(P, b)))
  elif v==3:
         
        n = len(b)
        x = np.zeros_like(b)
        max_iterations=1000
        tolerance=1e-6
        for _ in range(max_iterations):
          x_new = np.copy(x)
          for i in range(n):
            sigma = np.dot(A[i, :i], x_new[:i]) + np.dot(A[i, i+1:], x[i+1:])
            x_new[i] = (b[i] - sigma) / A[i, i]

          if np.allclose(x, x_new, atol=tolerance):
            break

          x = np.copy(x_new)

  elif v==4:
        x = np.zeros(p)  # Initialize weights
        for _ in range(max_iterations):
         x_new = np.copy(x)
         for i in range(n):
            sigma = np.dot(A[i, :i], x_new[:i]) + np.dot(A[i, i+1:], x[i+1:])
            x_new[i] = (b[i] - sigma) / A[i, i]

         if np.allclose(x, x_new, atol=tolerance):
            break

        x = np.copy(x_new)
  elif v==5:
         x = np.linalg.solve(A, np.dot(X.T, y))
  return x


# Train the model
weights = regularized_linear_regression(X_train, y_train)

# Predictions with thresholding
predictions = (np.dot(X_test, weights) >= 0.5).astype(int)

# Evaluate accuracy (or consider other classification metrics)
accuracy = accuracy_score(y_test, predictions)
print("Accuracy:", accuracy)