/* Copyright (C) 2024 Niklas Heidenreich - All Rights Reserved */
#pragma once
#include "tensor_cuda.hpp"  


template <typename T, std::size_t... DIMS>
class TensorAllocatorCUDA;

template <typename T, std::size_t DIM>
class TensorAllocatorCUDA<T,DIM>{
private:
  T* data = nullptr;
public:
  constexpr static size_t SIZE = DIM;
  constexpr static size_t RANK = 1;
  TensorAllocatorCUDA()  { gpuErrchk(cudaMalloc(&data, SIZE*sizeof(T))); }
  ~TensorAllocatorCUDA(){ gpuErrchk(cudaFree(data)); }
  T* get() const { return data; }
  TensorAllocatorCUDA(const TensorAllocatorCUDA<T, DIM>& other) = delete; // copy constructor
	TensorAllocatorCUDA(TensorAllocatorCUDA<T, DIM>&& other) noexcept = delete; // move constructor
  TensorAllocatorCUDA<T, DIM>& operator=(const TensorAllocatorCUDA<T, DIM>& other) = delete; // copy assignment
  TensorAllocatorCUDA<T, DIM>& operator=(TensorAllocatorCUDA<T, DIM>&& other) noexcept = delete; // move assignment
  TensorCUDA<T,DIM> createDeviceTensor() const { TensorCUDA<T,DIM> t(data); return t; }
};

template <typename T, std::size_t DIM, std::size_t... DIMS>
class TensorAllocatorCUDA<T,DIM,DIMS...>{
private:
  T* data = nullptr;
public:
	constexpr static size_t SIZE = DIM * (DIMS * ...);
	constexpr static size_t RANK = 1 + sizeof...(DIMS);
  TensorAllocatorCUDA()  { gpuErrchk(cudaMalloc(&data, SIZE*sizeof(T))); }
  ~TensorAllocatorCUDA(){ gpuErrchk(cudaFree(data)); }
  T* get() const { return data; }
  TensorAllocatorCUDA(const TensorAllocatorCUDA<T, DIM, DIMS...>& other) = delete; // copy constructor
	TensorAllocatorCUDA(TensorAllocatorCUDA<T, DIM, DIMS...>&& other) noexcept = delete; // move constructor
  TensorAllocatorCUDA<T, DIM, DIMS...>& operator=(const TensorAllocatorCUDA<T, DIM, DIMS...>& other) = delete; // copy assignment
  TensorAllocatorCUDA<T, DIM, DIMS...>& operator=(TensorAllocatorCUDA<T, DIM, DIMS...>&& other) noexcept = delete; // move assignment
  TensorCUDA<T,DIM,DIMS...> createDeviceTensor() const { TensorCUDA<T,DIM,DIMS...> t(data); return t; }
};
