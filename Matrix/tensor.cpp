#include "tensor.h"

Tensor::Tensor(){}
Tensor::Tensor(int size):dim(size){}

float& Tensor::operator()(int d, int i, int j){
    return a[d](i, j);
}
const float& Tensor::operator()(int d, int i, int j) const{
    return a[d](i, j);
}

Matrix Tensor::getM(int pos){
    return a[pos];
}
Tensor Tensor::operator[](std::pair<int,int> drange) const{
    Tensor b;
    for(int j = drange.first; j < drange.second; j++) {
        b.a.push_back(a[j]);
    }
    b.dim = drange.second - drange.first;
    return b;
}

Matrix Tensor::operator[](int value) const{
    return a[value];
}

int Tensor::getDim(){
    return dim;
}

Tensor Tensor::slice(int start, int end){
    Tensor b;
    for(int j = start; j < end; j++) {
        b.a.push_back(a[j]);
    }
    b.dim = end - start;
    return b;
}
