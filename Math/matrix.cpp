#include "matrix.h"

Matrix::Matrix(int r, int c): rows(r), columns(c), data(r*c + 0.0f){}

float& Matrix::operator()(int i, int j){
    return data[i * columns + j];
}

const float& Matrix::operator()(int i, int j) const{
    return data[i * columns + j];
}