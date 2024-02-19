/* Copyright (C) 2024 Niklas Heidenreich - All Rights Reserved */
#pragma once
#include "../tensor.hpp"
#include "tensor_cuda.hpp"  


template <typename T, std::size_t... DIMS>
class TensorAllocatorManaged;

template <typename T, std::size_t DIM>
class TensorAllocatorManaged<T,DIM>{
private:
  T* data = nullptr;
public:
  constexpr static size_t SIZE = DIM;
  constexpr static size_t RANK = 1;
  TensorAllocatorManaged()  { gpuErrchk(cudaMallocManaged(&data, SIZE*sizeof(T))); }
  ~TensorAllocatorManaged(){ gpuErrchk(cudaFree(data)); }
  T* get() const { return data; }
  TensorAllocatorManaged(const TensorAllocatorManaged<T, DIM>& other) = delete; // copy constructor
	TensorAllocatorManaged(TensorAllocatorManaged<T, DIM>&& other) noexcept = delete; // move constructor
  TensorAllocatorManaged<T, DIM>& operator=(const TensorAllocatorManaged<T, DIM>& other) = delete; // copy assignment
  TensorAllocatorManaged<T, DIM>& operator=(TensorAllocatorManaged<T, DIM>&& other) noexcept = delete; // move assignment
  Tensor<T,DIM> createTensor() const { Tensor<T,DIM> t(data); return t; }
  TensorCUDA<T,DIM> createDeviceTensor() const { TensorCUDA<T,DIM> t(data); return t; }
};

template <typename T, std::size_t DIM, std::size_t... DIMS>
class TensorAllocatorManaged<T,DIM,DIMS...>{
private:
  T* data = nullptr;
public:
	constexpr static size_t SIZE = DIM * (DIMS * ...);
	constexpr static size_t RANK = 1 + sizeof...(DIMS);
  TensorAllocatorManaged()  { gpuErrchk(cudaMallocManaged(&data, SIZE*sizeof(T))); }
  ~TensorAllocatorManaged(){ gpuErrchk(cudaFree(data)); }
  T* get() const { return data; }
  TensorAllocatorManaged(const TensorAllocatorManaged<T, DIM, DIMS...>& other) = delete; // copy constructor
	TensorAllocatorManaged(TensorAllocatorManaged<T, DIM, DIMS...>&& other) noexcept = delete; // move constructor
  TensorAllocatorManaged<T, DIM, DIMS...>& operator=(const TensorAllocatorManaged<T, DIM, DIMS...>& other) = delete; // copy assignment
  TensorAllocatorManaged<T, DIM, DIMS...>& operator=(TensorAllocatorManaged<T, DIM, DIMS...>&& other) noexcept = delete; // move assignment
  Tensor<T,DIM,DIMS...> createTensor() const { Tensor<T,DIM,DIMS...> t(data); return t; }
  TensorCUDA<T,DIM,DIMS...> createDeviceTensor() const { TensorCUDA<T,DIM,DIMS...> t(data); return t; }
};
