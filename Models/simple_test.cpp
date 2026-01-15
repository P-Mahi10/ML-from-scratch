#include "..\Math\matrix.h"
#include <iostream>

int main() {
    Matrix m(2, 2);
    m(0, 0) = 1;
    m(0, 1) = 2;
    m(1, 0) = 3;
    m(1, 1) = 4;
    
    std::cout << "Matrix created successfully!\n";
    std::cout << "m(0,0) = " << m(0, 0) << "\n";
    
    return 0;
}
