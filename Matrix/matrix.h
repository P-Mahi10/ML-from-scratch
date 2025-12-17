#pragma once
#include<vector>
#include <stdexcept>

class matrix{
    public:
        int rows, columns;
        std::vector<float> data;

        matrix(int r, int c);

        float& operator()(int i, int j);
        const float& operator()(int i, int j) const;
};