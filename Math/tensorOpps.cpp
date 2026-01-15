#include "tensorOpps.h"
#include "matrixOps.h"

Tensor sum(Tensor A, Tensor B){
    if(A.getDim() != B.getDim() || A[0].columns != B[0].columns || A[0].rows != B[0].rows){
        throw std::invalid_argument("Dimensions dont match");
    }
    Tensor C;
    for(int i = 0; i < A.getDim(); i++){
        C.a.push_back(add(A[i], B[i]));
    }
    C.dim = A.getDim();
    return C;
}

Tensor sum(Tensor A){
    for(int i = 1; i < A.getDim(); i++){
        A[0] = add(A[0], A[i]);
    }
    return A;
}

Tensor mul(Tensor A, Tensor B){
    if(A.getDim() != B.getDim() || A[0].columns != B[0].columns || A[0].rows != B[0].rows){
        throw std::invalid_argument("Dimensions dont match");
    }

    Tensor C;
    for(int i = 0; i < A.getDim(); i++){
        C.a.push_back(mul(A[i], B[i]));
    }
    C.dim = A.getDim();

    return C;
}

Tensor transpose(Tensor A){
    Tensor result;
    for(int i = 0; i < A.getDim(); i++){
        result.a.push_back(transpose(A[i]));
    }
    result.dim = A.getDim();
    return result;
}
