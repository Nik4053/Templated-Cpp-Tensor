
# Tensor Class using Expression Templates in C++

This class is designed to show the capabilities of C++ template metaprogramming and is not intended for production use.
While it has the potential to be very efficient, it can also contribute to very long compile times and unreadable error messages.

## Installation
Simply include the file `tensorlib/tensor.hpp` in your project.

## Example usage
```cpp
#include "tensorlib/tensor.hpp"
int main(){
    float data[4*3];
    Tensor<float, 4,3> t(data);
    t.set(3.2); // set all elements to 3.2
    t[1][1] = 100.0; // set element at index 1,1 to 2.0
    t.print(); // print the tensor
    std::cout << t << std::endl; // print the tensor
    t += 2 * t + 3; // perform elementwise operations
    auto expr = t + 2 * t + 3; // create an expression
    t.set(expr); // set the tensor to the expression
    std::cout << expr << std::endl; // print the expression
    float sum = expr.sum(); // sum all elements of expression
    auto t2 = t.reshape<3,4>(); // reshape the tensor
}
```


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
See `main.cpp`

## Expression Templates

Example of the new operations this tensor allows:
```cpp
Tensor<float, 4,3,2> A,B,C,D;                         // create a tensor with 3 dimensions
D = A + sin(B / C);                                        // add tensors
D = A + B + 4.2f;                                     // use constants
auto expr = A + B / C.max();                                // create an expression.
float value = expr[0];                                // evaluate the expression at a specific index 
value = expr[A.idx(1,1,1)];                           // evaluate the expression at a specific index

Tensor<float, 4,3,1> E;
E = A + B / C;                                         // ERROR: SIZE does not match

Tensor<float, 4*3*2> F;
F = A + B / C;                                         // Works: SIZE matches
```