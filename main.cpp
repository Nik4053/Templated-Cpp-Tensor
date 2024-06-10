#include <iostream>
#include "tensorlib/tensor.hpp"
#include "tensorlib/cpu/tensoralloc.hpp"

using namespace tel;

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
    t.set(2);

    TensorAllocator<float,dim3,dim2,dim1> allocator2;
    auto t2 = allocator2.createTensor();
    t[1][1][1] = -23;
    auto expr = t+ 3*t/2; // performance warning: t+ 3.0*t/2.0 slower than t+ 3.0/2.0*t as 3/2 is a simple non array division
    t2.set(expr); 
    t2.print();
    t.print();
    std::cout << "expr: " << expr << std::endl;
    std::cout << "tensor: " << t2 << std::endl;
    auto dot = (t*t2).sum();
    std::cout << "dot: " << dot << std::endl;
    std::cout << "dot2: " << t.dot(t2) << std::endl;

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
    Matmul(A,B,C);
    A.print();
    B.print();
    C.print();

    // Transpose
    Transpose(C);
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
