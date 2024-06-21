#include <iostream>
#include "tensor.hpp"
#include "batchedtensor.hpp"

using namespace tel;

void allocateTensors(){
    const size_t dim1 =4;
    const size_t dim2 =3;
    const size_t dim3 =1;

    // possible ways to allocate a Tensor
    // 1.
    float data[Tensor<float,dim3,dim2,dim1>::SIZE];
    Tensor<float,dim3,dim2,dim1> t1(data);
    // 2.
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
    float data[Tensor<float,dim3,dim2,dim1>::SIZE];

    Tensor<float,dim3,dim2,dim1> t(data);
    t.set(3.2);

    // using std::cout
    std::cout << t << std::endl;
}

void modifyTensor(){
    const size_t dim1 =2;
    const size_t dim2 =2;
    const size_t dim3 =2;
    float data[Tensor<float,dim3,dim2,dim1>::SIZE];
    Tensor<float,dim3,dim2,dim1> t(data);
    t.set(2);

    float data2[Tensor<float,dim3,dim2,dim1>::SIZE];
    Tensor<float,dim3,dim2,dim1> t2(data);
    t[1][1][1] = -23;
    auto expr = t+ 3*t/2; // performance warning: t+ 3.0*t/2.0 slower than t+ 3.0/2.0*t as 3/2 is a simple non array division
    t2.set(expr); 
    std::cout << t2 << std::endl;
    std::cout << t << std::endl;
    std::cout << "expr: " << expr << std::endl;
    std::cout << "tensor: " << t2 << std::endl;
    auto dot = (t*t2).sum();
    std::cout << "dot: " << dot << std::endl;
    std::cout << "dot2: " << t.dot(t2) << std::endl;

    // Tensor.operator+=()
    t += 10;
    t += t2;
    t *= t;
    std::cout << t << std::endl;

    // reshape
    auto t3 = t.reshape<dim3*dim2,dim1>();
    std::cout << t3 << std::endl;

    // flatten
    auto t4 = t.flatten();
    std::cout << t4 << std::endl;
}

void specialMathOperations(){
    const size_t dim1 =2;
    const size_t dim2 =2;
    float dataA[Tensor<float,dim2,dim1>::SIZE];
    Tensor<float,dim2,dim1> A(dataA);
    A.set(1.0);

    float dataB[Tensor<float,dim2,dim1>::SIZE];
    Tensor<float,dim2,dim1> B(dataB);
    B.set(2.0);
    B[0][0] = 3.0;

    float dataC[Tensor<float,dim2,dim1>::SIZE];
    Tensor<float,dim2,dim1> C(dataC);
    // t2.set(2.0);

    // A @ B = C
    Matmul(A,B,C);
    std::cout << A << std::endl;
    std::cout << B << std::endl;
    std::cout << C << std::endl;

    // Transpose
    Transpose(C);
    std::cout << C << std::endl;

}

void batchedTensor(){
    const size_t dim1 =2;
    const size_t dim2 =2;
    const size_t batch_size = 2;
    float data[batch_size*dim2*dim1] = {1,2,3,4,5,6,7,8};

    BatchedTensor<float,dim2,dim1> bt(batch_size,data);
    bt(1,1,1) = 3;
    std::cout << bt << std::endl;
    auto t2 = bt[1];
    std::cout << t2 << std::endl;
    auto t3 = bt.reshape<dim2,dim1>();
    std::cout << t3 << std::endl;
    auto t4 = bt.reshape<dim2,dim1*2>(1);
    std::cout << t4 << std::endl;
}


int main(){
    std::cout << "\nallocateTensors()" << std::endl;
    allocateTensors();
    std::cout << "\nprintTensor()" << std::endl;
    printTensor();
    std::cout << "\nmodifyTensor()" << std::endl;
    modifyTensor();
    std::cout << "\nspecialMathOperations()" << std::endl;
    specialMathOperations();
    std::cout << "batchedTensor()" << std::endl;
    batchedTensor();

}
