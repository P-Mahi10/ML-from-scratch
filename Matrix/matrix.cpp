#include "matrix.h"

matrix::matrix(int r, int c): rows(r), columns(c), data(r*c + 0.0f){}

float& matrix::operator()(int i, int j){
    return data[i * columns + j];
}

const float& matrix::operator()(int i, int j) const{
    return data[i * columns + j];
}