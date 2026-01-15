#include "matrixOps.h"

Matrix add(const Matrix& a, const Matrix& b){
    if (a.rows != b.rows || a.columns != b.columns) {
        throw std::invalid_argument("Matrix dimensions must match for addition");
    }

    Matrix c(a.rows, a.columns);

    for(int i = 0; i < a.rows; i++){
        for(int j = 0; j < a.columns; j++){
            c(i, j) = a(i, j) + b(i, j);
        }
    }

    return c;
}

Matrix mul(const Matrix& a, const Matrix& b) {
    if (a.columns != b.rows) {
        throw std::invalid_argument("Matrix multiplication cannot take place");
    }

    Matrix c(a.rows, b.columns);

    for (int i = 0; i < a.rows; i++) {
        for (int j = 0; j < b.columns; j++) {
            float sum = 0.0;
            for (int k = 0; k < a.columns; k++) {
                sum += a(i, k) * b(k, j);
            }
            c(i, j) = sum;
        }
    }

    return c;
}

Matrix hadamard(const Matrix& a, const Matrix& b){
    if (a.rows != b.rows || a.columns != b.columns) {
        throw std::invalid_argument("Matrix dimensions must match for addition");
    }

    Matrix c(a.rows, a.columns);

    for(int i = 0; i < a.rows; i++){
        for(int j = 0; j < a.columns; j++){
            c(i, j) = a(i, j) * b(i, j);
        }
    }

    return c;
}

Matrix transpose(const Matrix& a){
    Matrix c(a.columns, a.rows);

    for(int i = 0; i < a.rows; i++){
        for(int j = 0; j < a.columns; j++){
            c(j, i) = a(i, j);
        }
    }

    return c;
}