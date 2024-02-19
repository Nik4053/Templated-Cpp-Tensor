#include <iostream>
#include "tensorlib/tensor.hpp"
#include "tensorlib/cpu/tensoralloc.hpp"
#include "tensorlib/cpu/tensorhelper.hpp"


void allocateTensors(){
    const size_t dim1 =4;
    const size_t dim2 =3;
    const size_t dim3 =1;

    // possible ways to allocate a Tensor
    // 1.
    TensorAllocator<float,dim3,dim2,dim1> allocator; // will free data on destruction
    auto t = allocator.createTensor();
    // 2.
    float data[Tensor<float,dim3,dim2,dim1>::SIZE];
    Tensor<float,dim3,dim2,dim1> t1(data);
    // 3.
    float *data2 = new float[Tensor<float,dim3,dim2,dim1>::SIZE];
    Tensor<float,dim3,dim2,dim1> t2(data2);
    free(data2);
    // WARNING: Tensor<float,dim3,dim2,dim1> t2(data2); will not free data2 on destruction!
    // t2 is now operating on a dangling pointer!!!
}

void printTensor(){
    const size_t dim1 =2;
    const size_t dim2 =2;
    const size_t dim3 =2;
    TensorAllocator<float,dim3,dim2,dim1> allocator;
    auto t = allocator.createTensor();
    t.set(3.2);

    // Tensor.print() 
    t.print();
    // using std::cout
    std::cout << t << std::endl;
}

void modifyTensor(){
    const size_t dim1 =2;
    const size_t dim2 =2;
    const size_t dim3 =2;
    TensorAllocator<float,dim3,dim2,dim1> allocator;
    auto t = allocator.createTensor();
    t.set(1);

    TensorAllocator<float,dim3,dim2,dim1> allocator2;
    auto t2 = allocator2.createTensor();
    t2.set(2.0);

    // Tensor.operator+=()
    t += 10;
    t += t2;
    t *= t;
    t.print();

    // reshape
    auto t3 = t.reshape<dim3*dim2,dim1>();
    t3.print();

    // flatten
    auto t4 = t.flatten();
    t4.print();
}

void specialMathOperations(){
    const size_t dim1 =2;
    const size_t dim2 =2;
    TensorAllocator<float,dim2,dim1> allocator;
    auto A = allocator.createTensor();
    A.set(1.0);

    TensorAllocator<float,dim2,dim1> allocator2;
    auto B = allocator2.createTensor();
    B.set(2.0);
    B[0][0] = 3.0;

    TensorAllocator<float,dim2,dim1> allocator3;
    auto C = allocator3.createTensor();
    // t2.set(2.0);

    // A @ B = C
    TensorHelper::Matmul(A,B,C);
    A.print();
    B.print();
    C.print();

    // Transpose
    TensorHelper::Transpose(C);
    C.print();

}


int main(){
    std::cout << "allocateTensors()" << std::endl;
    allocateTensors();
    std::cout << "printTensor()" << std::endl;
    printTensor();
    std::cout << "modifyTensor()" << std::endl;
    modifyTensor();
    std::cout << "specialMathOperations()" << std::endl;
    specialMathOperations();

}
