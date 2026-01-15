#include "linearRegression.h"
#include <cmath>
#include <stdexcept>

// Constructor
LinearRegression::LinearRegression() : coefficients(0, 0), intercept(0.0f), trained(false) {}

// Fit the model to training data using normal equation
void LinearRegression::fit(const Matrix& X, const Matrix& y) {
    if (X.rows != y.rows) {
        throw std::invalid_argument("X and y must have the same number of samples");
    }

    int m = X.rows;
    int n = X.columns;

    // Create augmented feature matrix with bias term (add column of 1s)
    Matrix X_augmented(m, n + 1);
    for (int i = 0; i < m; i++) {
        X_augmented(i, 0) = 1.0f;  // Bias column
        for (int j = 0; j < n; j++) {
            X_augmented(i, j + 1) = X(i, j);
        }
    }

    // Compute normal equation: β = (X^T X)^(-1) X^T y
    // Step 1: Compute X^T
    Matrix X_T = transpose(X_augmented);

    // Step 2: Compute X^T X
    Matrix XTX = mul(X_T, X_augmented);

    // Step 3: Compute X^T y
    Matrix XTy = mul(X_T, y);

    // Step 4: Solve (X^T X) β = X^T y using Gaussian elimination
    coefficients = gaussianElimination(XTX, XTy);
    intercept = coefficients(0, 0);

    trained = true;
}

// Make predictions on new data
Matrix LinearRegression::predict(const Matrix& X) {
    if (!trained) {
        throw std::runtime_error("Model must be trained before making predictions");
    }

    int m = X.rows;
    Matrix predictions(m, 1);

    for (int i = 0; i < m; i++) {
        float pred = intercept;
        for (int j = 0; j < X.columns; j++) {
            pred += X(i, j) * coefficients(j + 1, 0);
        }
        predictions(i, 0) = pred;
    }

    return predictions;
}

// Calculate Mean Squared Error
float LinearRegression::calculateMSE(const Matrix& y_true, const Matrix& y_pred) {
    if (y_true.rows != y_pred.rows) {
        throw std::invalid_argument("y_true and y_pred must have same number of samples");
    }

    float mse = 0.0f;
    for (int i = 0; i < y_true.rows; i++) {
        float error = y_true(i, 0) - y_pred(i, 0);
        mse += error * error;
    }

    return mse / y_true.rows;
}

// Get model coefficients
Matrix LinearRegression::getCoefficients() const {
    if (!trained) {
        throw std::runtime_error("Model not trained yet");
    }
    return coefficients;
}

// Get intercept value
float LinearRegression::getIntercept() const {
    return intercept;
}

// Gaussian elimination to solve linear system Ax = b
Matrix LinearRegression::gaussianElimination(Matrix A, Matrix b) {
    int n = A.rows;

    // Forward elimination
    for (int i = 0; i < n; i++) {
        // Find pivot
        int maxRow = i;
        for (int k = i + 1; k < n; k++) {
            if (std::abs(A(k, i)) > std::abs(A(maxRow, i))) {
                maxRow = k;
            }
        }

        // Swap rows
        swapRows(A, i, maxRow);
        swapRows(b, i, maxRow);

        // Make diagonal element 1
        float pivot = A(i, i);
        if (std::abs(pivot) < 1e-9f) {
            throw std::runtime_error("Singular matrix encountered");
        }

        for (int j = i; j < n; j++) {
            A(i, j) /= pivot;
        }
        b(i, 0) /= pivot;

        // Eliminate column below
        for (int k = i + 1; k < n; k++) {
            float factor = A(k, i);
            for (int j = i; j < n; j++) {
                A(k, j) -= factor * A(i, j);
            }
            b(k, 0) -= factor * b(i, 0);
        }
    }

    // Back substitution
    Matrix x(n, 1);
    for (int i = n - 1; i >= 0; i--) {
        x(i, 0) = b(i, 0);
        for (int j = i + 1; j < n; j++) {
            x(i, 0) -= A(i, j) * x(j, 0);
        }
    }

    return x;
}

// Helper function to swap rows in a matrix
void LinearRegression::swapRows(Matrix& mat, int row1, int row2) {
    for (int j = 0; j < mat.columns; j++) {
        float temp = mat(row1, j);
        mat(row1, j) = mat(row2, j);
        mat(row2, j) = temp;
    }
}
