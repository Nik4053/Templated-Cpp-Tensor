/* Copyright (C) 2024 Niklas Heidenreich - All Rights Reserved */
#pragma once
#include "../tensor.hpp" 


namespace tel{
  template <typename T, std::size_t... DIMS>
  class TensorAllocator;

  template <typename T, std::size_t DIM>
  class TensorAllocator<T,DIM>{
  private:
    T* data = nullptr;
  public:
    constexpr static size_t SIZE = DIM;
    constexpr static size_t RANK = 1;
    TensorAllocator()  { data= new T[SIZE]; }
    ~TensorAllocator(){ delete[] data; }
    T* get() const { return data; }
    TensorAllocator(const TensorAllocator<T, DIM>& other) = delete; // copy constructor
    TensorAllocator(TensorAllocator<T, DIM>&& other) noexcept = delete; // move constructor
    TensorAllocator<T, DIM>& operator=(const TensorAllocator<T, DIM>& other) = delete; // copy assignment
    TensorAllocator<T, DIM>& operator=(TensorAllocator<T, DIM>&& other) noexcept = delete; // move assignment
    Tensor<T,DIM> createTensor() const { Tensor<T,DIM> t(data); return t; }
  };

  template <typename T, std::size_t DIM, std::size_t... DIMS>
  class TensorAllocator<T,DIM,DIMS...>{
  private:
    T* data = nullptr;
  public:
    constexpr static size_t SIZE = DIM * (DIMS * ...);
    constexpr static size_t RANK = 1 + sizeof...(DIMS);
    TensorAllocator()  { data= new T[SIZE]; }
    ~TensorAllocator(){ delete[] data; }
    T* get() const { return data; }
    TensorAllocator(const TensorAllocator<T, DIM, DIMS...>& other) = delete; // copy constructor
    TensorAllocator(TensorAllocator<T, DIM, DIMS...>&& other) noexcept = delete; // move constructor
    TensorAllocator<T, DIM, DIMS...>& operator=(const TensorAllocator<T, DIM, DIMS...>& other) = delete; // copy assignment
    TensorAllocator<T, DIM, DIMS...>& operator=(TensorAllocator<T, DIM, DIMS...>&& other) noexcept = delete; // move assignment
    Tensor<T,DIM,DIMS...> createTensor() const { Tensor<T,DIM,DIMS...> t(data); return t; }
  };
} // namespace tel