#include "tensor.h"

Tensor::Tensor(){}//default
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
    for(int j = drange.first; j < drange.second; j++) b.a[j - drange.first] = a[j];
    return b;
}

int Tensor::getDim(){
    return a.size();
}

Tensor Tensor::slice(int start, int end){
    Tensor b;
    for(int j = start; j < end; j++) b.a[j - start] = a[j];
    return b;
}
