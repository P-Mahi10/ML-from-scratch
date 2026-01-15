#pragma once
#include "matrixOps.h"

class Tensor{
    public:
        std::vector<Matrix> a;
        int dim = 0;
        Tensor(); //allowing the user to make a dynamic tensor
        Tensor(int size);//for having fixed sized tensors

        float& operator()(int d, int i, int j);
        const float& operator()(int d, int i, int j) const;
        Tensor operator[](std::pair<int,int> drange) const;
        Matrix operator[](int value) const;

        int getDim();
        
        Matrix getM(int pos);
        Tensor slice(int start, int end);
};