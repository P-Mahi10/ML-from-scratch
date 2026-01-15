#pragma once
#include "..\Math\matrix.h"
#include "..\Math\matrixOps.h"

class LinearRegression {
private:
    Matrix coefficients;  // Stores the learned parameters (weights)
    float intercept;      // Bias term
    bool trained;         // Flag to track if model has been trained

    // Helper functions
    Matrix gaussianElimination(Matrix A, Matrix b);
    void swapRows(Matrix& mat, int row1, int row2);

public:
    // Constructor initializes the model
    LinearRegression();

    // Fit the model to training data using normal equation
    // X: feature matrix (n_samples x n_features)
    // y: target vector (n_samples x 1)
    void fit(const Matrix& X, const Matrix& y);

    // Make predictions on new data
    // X: feature matrix (n_samples x n_features)
    // Returns: predictions (n_samples x 1)
    Matrix predict(const Matrix& X);

    // Calculate Mean Squared Error between predictions and actual values
    float calculateMSE(const Matrix& y_true, const Matrix& y_pred);

    // Get model coefficients
    Matrix getCoefficients() const;

    // Get intercept value
    float getIntercept() const;
};
