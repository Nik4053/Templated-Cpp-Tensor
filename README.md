
# Tensor Class using Expression Templates in C++

This class is designed to show the capabilities of C++ template metaprogramming and is not intended for production use.
While it has the potential to be very efficient, it can also contribute to very long compile times and unreadable error messages.

## Introduction
After finding myself writing the same tensor class over and over again, I decided to create a reusable C++ tensor class. 
This class is designed to be simple and easy to use, while still being efficient. 
The class is implemented using the C++ standard library and is header-only, meaning that it can be included in any C++ project without the need for any additional libraries. 

What first started as a simple class, quickly turned into a project to explore the capabilities of C++ template metaprogramming.
I started with a simple tensor class with a template parameter for the data type, then extended it to also include the number of dimensions as a template parameter.
And finally, I added support for arbitrary tensor shapes using variadic templates.

 ```cpp
 Tensor<float> tensor;           // first version  <dtype>
 Tensor<float, 3> tensor;        // second version <dtype,RANK>
 Tensor<float, 2, 2, 2> tensor;  // final version  <dtype,SHAPE>
 ```

## Base implementation
I wanted a simple storage class that allows me to easily access the data and perform basic operations such as addition, subtraction, multiplication, and division.

The class itself does therefore not contain any data, but instead contains a pointer to the data.
This also allows me to easily create sub-tensors by simply creating a new tensor object with a different pointer to the same data.


```cpp
// basic tensor definition
template <typename T, std::size_t... DIMS>
class Tensor;
 
// specialization for 1 dimensions
template <typename T, std::size_t DIM>
class Tensor<T, DIM> {
public:
    T* data;
    constexpr static size_t SIZE = DIM;
    constexpr static size_t RANK = 1;
    Tensor(T* data) : data(data) {}
};


template <typename T, std::size_t DIM, std::size_t... DIMS>
class Tensor<T,DIM,DIMS...> {
public:
    T* data;
    constexpr static size_t SIZE = DIM * (DIMS * ...);
    constexpr static size_t RANK = 1 + sizeof...(DIMS);
    Tensor(T *data) : data(data) {}
};
```

Example of the operations this tensor allows:
```cpp
Tensor<float, 4,3,2> tensor;                    // create a tensor with 3 dimensions
tensor[0][0][0] = 5;                            // access the data
tensor(0,0,0) = 5;                              // access the data
tensor += 1;                                    // multiply all elements by 2
tensor +=tensor;                                // add 1 to all elements
Tensor<float, 3,2> sub = tensor[0];             // create a sub-tensor 
Tensor<float, 4*3*2> flat = tensor.flatten();   // flatten the tensor
cout << tensor << endl;                         // print the tensor
```

## Helper functions
I also added some helper functions to make the class more user-friendly. 


```cpp
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
```

## Expression Templates

Example of the new operations this tensor allows:
```cpp
Tensor<float, 4,3,2> A,B,C,D;                         // create a tensor with 3 dimensions
D = A + B / C;                                        // add tensors
D = A + B + Tval<D.SIZE>(4.2f);                       // use constants
auto expr = A + B / C;                                // create an expression.
float value = expr[0];                                // evaluate the expression at a specific index 
value = expr[A.idx(1,1,1)];                           // evaluate the expression at a specific index

Tensor<float, 4,3,1> E;
E = A + B / C;                                         // ERROR: SIZE does not match

Tensor<float, 4*3*2> F;
F = A + B / C;                                         // Works: SIZE matches
```