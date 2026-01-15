#include "linearRegression.h"
#include <iostream>
#include <iomanip>
#include <cmath>

void printTestResults(const std::string& testName, const Matrix& y_true, const Matrix& y_pred, float mse) {
    std::cout << "\n========================================\n";
    std::cout << testName << "\n";
    std::cout << "========================================\n";
    std::cout << "Actual vs Predicted:\n";
    std::cout << std::setw(12) << "Actual" << std::setw(12) << "Predicted" << std::setw(12) << "Error\n";
    std::cout << "----------------------------------------\n";
    
    for (int i = 0; i < y_true.rows; i++) {
        float error = y_true(i, 0) - y_pred(i, 0);
        std::cout << std::setw(12) << std::fixed << std::setprecision(4) 
                  << y_true(i, 0) << std::setw(12) << y_pred(i, 0) 
                  << std::setw(12) << error << "\n";
    }
    
    std::cout << "\nMean Squared Error: " << mse << "\n";
    std::cout << "RMSE: " << std::sqrt(mse) << "\n";
}

// Test Case 1: Perfect Linear Relationship
// y = 2x + 3 - Linear Regression should work EXCELLENT
void test_perfect_linear() {
    std::cout << "\n\n***** TEST 1: PERFECT LINEAR RELATIONSHIP *****\n";
    std::cout << "Data follows: y = 2x + 3\n";
    std::cout << "Expected: Linear regression should work EXCELLENT\n";
    
    // Training data: y = 2x + 3
    Matrix X_train(10, 1);
    Matrix y_train(10, 1);
    
    for (int i = 0; i < 10; i++) {
        float x = i * 1.0f;
        X_train(i, 0) = x;
        y_train(i, 0) = 2.0f * x + 3.0f;  // Perfect linear relationship
    }
    
    // Train model
    LinearRegression model;
    model.fit(X_train, y_train);
    
    // Test data
    Matrix X_test(5, 1);
    Matrix y_test(5, 1);
    
    for (int i = 0; i < 5; i++) {
        float x = i * 2.0f + 0.5f;  // Different test points
        X_test(i, 0) = x;
        y_test(i, 0) = 2.0f * x + 3.0f;
    }
    
    // Predict
    Matrix y_pred = model.predict(X_test);
    float mse = model.calculateMSE(y_test, y_pred);
    
    printTestResults("TEST 1: PERFECT LINEAR", y_test, y_pred, mse);
    
    std::cout << "\nLearned coefficients: ";
    std::cout << "Intercept = " << model.getIntercept() 
              << ", Slope = " << model.getCoefficients()(1, 0) << "\n";
    std::cout << "Expected: Intercept ≈ 3, Slope ≈ 2\n";
    
    if (mse < 0.001f) {
        std::cout << "✓ RESULT: EXCELLENT - Model fits perfectly!\n";
    }
}

// Test Case 2: Noisy Linear Relationship
// y = 0.5x + 2 with some noise - Linear Regression should work REASONABLY WELL
void test_noisy_linear() {
    std::cout << "\n\n***** TEST 2: NOISY LINEAR RELATIONSHIP *****\n";
    std::cout << "Data follows: y ≈ 0.5x + 2 (with noise)\n";
    std::cout << "Expected: Linear regression should work REASONABLY WELL\n";
    
    // Training data: y = 0.5x + 2 with noise
    Matrix X_train(15, 1);
    Matrix y_train(15, 1);
    
    float noise[] = {0.3f, -0.2f, 0.5f, -0.4f, 0.1f, -0.3f, 0.4f, -0.1f, 
                     0.2f, -0.5f, 0.3f, 0.1f, -0.2f, 0.4f, -0.3f};
    
    for (int i = 0; i < 15; i++) {
        float x = i * 0.5f;
        X_train(i, 0) = x;
        y_train(i, 0) = 0.5f * x + 2.0f + noise[i];  // Linear with noise
    }
    
    // Train model
    LinearRegression model;
    model.fit(X_train, y_train);
    
    // Test data (actual linear relationship without noise)
    Matrix X_test(5, 1);
    Matrix y_test(5, 1);
    
    for (int i = 0; i < 5; i++) {
        float x = i * 1.5f + 1.0f;
        X_test(i, 0) = x;
        y_test(i, 0) = 0.5f * x + 2.0f;  // True relationship
    }
    
    // Predict
    Matrix y_pred = model.predict(X_test);
    float mse = model.calculateMSE(y_test, y_pred);
    
    printTestResults("TEST 2: NOISY LINEAR", y_test, y_pred, mse);
    
    std::cout << "\nLearned coefficients: ";
    std::cout << "Intercept = " << model.getIntercept() 
              << ", Slope = " << model.getCoefficients()(1, 0) << "\n";
    std::cout << "Expected: Intercept ≈ 2, Slope ≈ 0.5 (approximately)\n";
    
    if (mse < 0.5f) {
        std::cout << "✓ RESULT: GOOD - Model captures general trend despite noise!\n";
    }
}

// Test Case 3: Non-linear Quadratic Relationship
// y = x^2 + x + 1 - Linear Regression should work POORLY
void test_quadratic_nonlinear() {
    std::cout << "\n\n***** TEST 3: NON-LINEAR QUADRATIC RELATIONSHIP *****\n";
    std::cout << "Data follows: y = x² + x + 1\n";
    std::cout << "Expected: Linear regression should work POORLY\n";
    
    // Training data: y = x^2 + x + 1 (quadratic)
    Matrix X_train(12, 1);
    Matrix y_train(12, 1);
    
    for (int i = 0; i < 12; i++) {
        float x = (i - 6.0f) * 0.5f;  // Range from -3 to 2.5
        X_train(i, 0) = x;
        y_train(i, 0) = x * x + x + 1.0f;  // Quadratic relationship
    }
    
    // Train model
    LinearRegression model;
    model.fit(X_train, y_train);
    
    // Test data
    Matrix X_test(6, 1);
    Matrix y_test(6, 1);
    
    for (int i = 0; i < 6; i++) {
        float x = (i - 3.0f);  // Range from -3 to 2
        X_test(i, 0) = x;
        y_test(i, 0) = x * x + x + 1.0f;
    }
    
    // Predict
    Matrix y_pred = model.predict(X_test);
    float mse = model.calculateMSE(y_test, y_pred);
    
    printTestResults("TEST 3: QUADRATIC NON-LINEAR", y_test, y_pred, mse);
    
    std::cout << "\nLearned coefficients: ";
    std::cout << "Intercept = " << model.getIntercept() 
              << ", Slope = " << model.getCoefficients()(1, 0) << "\n";
    std::cout << "Note: Linear model tries to fit a line to curved data\n";
    
    if (mse > 1.0f) {
        std::cout << "✗ RESULT: POOR - Model cannot capture quadratic relationship!\n";
        std::cout << "Linear regression is insufficient for this non-linear data.\n";
    }
}

int main() {
    std::cout << "========================================\n";
    std::cout << "   LINEAR REGRESSION MODEL TESTING     \n";
    std::cout << "========================================\n";
    
    try {
        // Test 1: Perfect linear - should work excellently
        test_perfect_linear();
        
        // Test 2: Noisy linear - should work reasonably well
        test_noisy_linear();
        
        // Test 3: Non-linear quadratic - should work poorly
        test_quadratic_nonlinear();
        
        std::cout << "\n\n========================================\n";
        std::cout << "         TEST SUMMARY                   \n";
        std::cout << "========================================\n";
        std::cout << "Test 1 (Perfect Linear): Expected to work EXCELLENT\n";
        std::cout << "Test 2 (Noisy Linear):   Expected to work REASONABLY WELL\n";
        std::cout << "Test 3 (Quadratic):      Expected to work POORLY\n";
        std::cout << "\nAll tests completed successfully!\n";
        
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }
    
    return 0;
}
