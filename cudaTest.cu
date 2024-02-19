#include "tensorlib/tensor.hpp"
#include "tensorlib/cpu/tensoralloc.hpp"
#include "tensorlib/cuda/tensor_cuda.hpp"
#include "tensorlib/cuda/tensoralloc_cuda.hpp"
#include "tensorlib/cpu/batchedtensor.hpp"
// CUDA includes
#include <cuda_runtime.h>


__global__ void cuda_hello(){
    printf("Hello World from GPU!\n");
}
template <typename T, std::size_t DIM3, std::size_t DIM2, std::size_t DIM1>
__global__ void cuda_hello_tensor(Tensor<float,DIM3,DIM2,DIM1> t){
    t(0,0,0) = 5;
    printf("Hello World from GPU! %lu\n", t.RANK);
}

template <typename T, std::size_t DIM3, std::size_t DIM2, std::size_t DIM1>
__global__ void cuda_hello_tensor(TensorCUDA<float,DIM3,DIM2,DIM1> t){
    t(0,0,0) = 5;
    printf("Hello World from GPU! %lu\n", t.RANK);
}


class Layer{
    TensorAllocator<float,3,3,3> talloc;
    Tensor<float,3,3,3> input = talloc.createTensor();
};
int main() {
    const size_t dim1 =4;
    const size_t dim2 =3;
    const size_t dim3 =1;
    float *data;
    cudaMallocManaged(&data, dim1*dim2*dim3*sizeof(float));
    Tensor<float,dim3,dim2,dim1> t(data);
    cuda_hello<<<1,1>>>(); 
    cudaDeviceSynchronize();
    cuda_hello_tensor<float,dim3,dim2,dim1><<<1,1>>>(t);
    cudaDeviceSynchronize();
    // std::cout << t << std::endl;
    TensorCUDA<float,dim3,dim2,dim1> t2(data);
    // t2[0][0][1] = 7;
    t2.print();

    float dataHOST[dim1*dim2*dim3];
    for (size_t i = 0; i < dim1*dim2*dim3; i++)
    {
        dataHOST[i] = i;
    }
    Tensor<float,dim3,dim2,dim1> tH(dataHOST);
    tH.print();
    float *dataDEVICE;
    cudaMalloc(&dataDEVICE, dim1*dim2*dim3*sizeof(float));
    TensorCUDA<float,dim3,dim2,dim1> tD2(dataDEVICE);
    tD2.print();
    tD2.copyFromHost(tH);
    cuda_hello_tensor<float,dim3,dim2,dim1><<<1,1>>>(tD2);
    cudaDeviceSynchronize();
    tD2.print();
    tD2.copyToHost(tH);
    
    TensorAllocator<float,3,3,3> talloc;
    auto t3 = talloc.createTensor();
    t3.set(5);
    TensorAllocatorCUDA<float,3,3,3> tallocCUDA;
    auto t4 = tallocCUDA.createDeviceTensor();

    t4.copyFromHost(t3);
    t4.print();


    // std::cout << t2 << std::endl;
    Layer *l = new Layer();
    return 0;
}