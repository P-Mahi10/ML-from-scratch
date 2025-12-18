#include "matrixOps.h"

matrix add(const matrix& a, const matrix& b){
    //checking size
    if (a.rows != b.rows || a.columns != b.columns) {
        throw std::invalid_argument("Matrix dimensions must match for addition");
    }

    //addition process:
    matrix c(a.rows, a.columns);

    for(int i = 0; i <= a.rows; i++){
        for(int j = 0; j <= a.columns; j++){
            c(i, j) = a(i, j) + b(i, j);
        }
    }

    return c;
}

matrix mul(const matrix& a, const matrix& b){
    if(a.columns != b.rows){
        throw std::invalid_argument("Matrix multiplication cannot take place");
    }

    matrix c(a.columns, b.rows);
    //yet to complete
}