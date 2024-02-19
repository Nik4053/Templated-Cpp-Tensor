/* Copyright (C) 2024 Niklas Heidenreich - All Rights Reserved */
#pragma once
#include "../tensor.hpp"
#include "cudaerrcheck.cuh"
// using namespace std;
// #ifndef NDEBUG
// #warning "TensorCUDA class compiled in DEBUG mode! Performance may be slow!"
// #endif
template <typename T, std::size_t... DIMS>
class TensorCUDA;

/*
    Internal helpers to get the TensorCUDA class working
*/
namespace internal{
  template <typename T, std::size_t ... DIMS>
  struct genTenCUDA;
  
  template <typename T, std::size_t DIM>
  struct genTenCUDA<T,DIM>
  {
      using type = TensorCUDA<T,DIM>;
  };
  template <typename T, std::size_t... DIMS>
  struct genTenCUDA
  {
      using type = TensorCUDA<T,DIMS...>;
  };
  template <typename T, std::size_t DIM>
  __global__ void printTensorCUDA(const TensorCUDA<T,DIM> t) { 
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if(idx == 0) {
      for(size_t i=0; i < DIM; i++) {
        printf("%e",t.data[i]);
        if(i != DIM-1) printf(", ");
      }
    }
  }

};

template <typename T, std::size_t DIM>
class TensorCUDA<T, DIM>{
private:
public:
  T *data = nullptr;
  constexpr static size_t SIZE = DIM;
  constexpr static size_t RANK = 1;
  
  template <std::size_t ...NDIMS>
  TensorCUDA(const TensorCUDA<float,NDIMS...>& other){ static_assert(SIZE == TensorCUDA<float,NDIMS...>::SIZE); this->data = other.data;};

  TensorCUDA(T *data) : data(data) {}
  constexpr static size_t idx(size_t x) { assert(x < DIM); return x; }
  __device__ T operator[](size_t x) const {assert(x < DIM); return data[x];}
  __device__ T& operator[](size_t x) {assert(x < DIM); return data[x];}
  __device__ T operator()(size_t x) const {assert(x < DIM); return data[idx(x)];}
  __device__ T& operator()(size_t x) {assert(x < DIM); return data[idx(x)];}
  template <std::size_t ...NDIMS>
  TensorCUDA<T,NDIMS...> reshape() const {static_assert(TensorCUDA<T,NDIMS...>::SIZE == SIZE,"SIZE of reshaped TensorCUDA has to stay the same!"); TensorCUDA<T,NDIMS...> t(data); return t; }
  void print(int precision=5, bool newline=true){ 
    std::cout<< "[";
    internal::printTensorCUDA<<<1,1>>>(*this);
    gpuErrchk( cudaPeekAtLastError() );
    gpuErrchk( cudaDeviceSynchronize() );
    std::cout << "]";
    if(newline) std::cout << std::endl;
  }
  void copyFromHost(const Tensor<T,DIM>& src) { gpuErrchk(cudaMemcpy(data, src.data, SIZE*sizeof(T), cudaMemcpyHostToDevice)); }
  void copyToHost(Tensor<T,DIM>& dst) const { gpuErrchk(cudaMemcpy(dst.data, data, SIZE*sizeof(T), cudaMemcpyDeviceToHost)); }

};


template <typename T, std::size_t DIM, std::size_t... DIMS>
class TensorCUDA<T,DIM,DIMS...> {
private:
public:
	T *data = nullptr;
	constexpr static size_t SIZE = DIM * (DIMS * ...);
	constexpr static size_t RANK = 1 + sizeof...(DIMS);
	TensorCUDA(T *data) : data(data) {}
	template <typename ...Args>
	constexpr static size_t idx(size_t x, Args ... args) { assert(x < DIM); return x*internal::genTenCUDA<T,DIMS...>::type::SIZE + internal::genTenCUDA<T,DIMS...>::type::idx(args...) ; }
	__host__ __device__ typename internal::genTenCUDA<T,DIMS...>::type operator[](size_t x) const { assert(x < DIM);  typename internal::genTenCUDA<T,DIMS...>::type t(data+x*internal::genTenCUDA<T,DIMS...>::type::SIZE); return t; }
	template <typename ...Args>
  __host__ __device__ T operator()(size_t x, Args ... args) const {assert(x < DIM); return data[idx(x,args...)];}
	template <typename ...Args>
  __host__ __device__ T& operator()(size_t x, Args ... args) {assert(x < DIM); return data[idx(x, args...)];}
  TensorCUDA<T,SIZE> flatten() const {TensorCUDA<T,SIZE> t(data); return t; }
  template <std::size_t ...NDIMS>
  TensorCUDA<T,NDIMS...> reshape() const {static_assert(TensorCUDA<T,NDIMS...>::SIZE == SIZE,"SIZE of reshaped TensorCUDA has to stay the same!"); TensorCUDA<T,NDIMS...> t(data); return t; }
  void print(int precision=5, bool newline=true){ 
    if(newline) std::cout << "DEVICE: ";
    std::cout<< "[";
    for(size_t d=0; d < DIM; d++){
      if(d + 1 == DIM) this->operator[](d).print(precision,false);
      else  { 
        this->operator[](d).print(precision,false) ;
        std::cout << ",";
      }
    }
    std::cout << "]";
    if(newline) std::cout << std::endl;
  }
  void copyFromHost(const Tensor<T,DIM,DIMS...>& src) { flatten().copyFromHost(src); }
  void copyToHost(Tensor<T,DIM,DIMS...>& dst) const { auto tmp = dst.flatten(); flatten().copyToHost(tmp); }
};


// class TensorCUDAAllocator{
// private:
//   void* data = nullptr;
//   size_t size = 0;
// public:
//   TensorCUDAAllocator(size_t size) : size(size) { gpuErrchk(cudaMalloc(&data, size)); }
//   ~TensorCUDAAllocator(){ gpuErrchk(cudaFree(data)); }
//   void* get() const { return data; }
//   size_t getSize() const { return size; }
// };