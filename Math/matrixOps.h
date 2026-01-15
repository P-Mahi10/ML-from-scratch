#pragma once
#include "matrix.h"

Matrix add(const Matrix& a, const Matrix& b);
Matrix mul(const Matrix& a, const Matrix& b);
Matrix hadamard(const Matrix& a, const Matrix& b);
Matrix transpose(const Matrix& a);

