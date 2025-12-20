#pragma once
#include<vector>
#include <stdexcept>

class Matrix{
    public:
        int rows, columns;
        std::vector<float> data;

        Matrix(int r, int c);

        float& operator()(int i, int j);
        const float& operator()(int i, int j) const;
};