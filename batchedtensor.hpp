/* Copyright (C) 2024 Niklas Heidenreich - All Rights Reserved */
#pragma once
#include "tensor.hpp"


namespace tel{
    template <typename T, std::size_t... DIMS>
    class BatchedTensor {
    private:
        T *data = nullptr;
    public:
        const size_t batch_size;
        size_t SIZE = Tensor<T,DIMS...>::SIZE * batch_size;
        BatchedTensor(size_t batch_size, T *data) : data(data), batch_size(batch_size) {	}
        template <typename ...Args>
        size_t idx(size_t b, Args ... args) const { assert(b < batch_size); return b*Tensor<T,DIMS...>::SIZE + Tensor<T,DIMS...>::idx(args...); }
        template <typename ...Args>
        T operator()(size_t b, Args ... args) const { assert(b < batch_size); return data[b*Tensor<T,DIMS...>::SIZE + Tensor<T,DIMS...>::idx(args...)]; }
        template <typename ...Args>
        T& operator()(size_t b, Args ... args) {assert(b < batch_size); return data[b*Tensor<T,DIMS...>::SIZE + Tensor<T,DIMS...>::idx(args...)]; }
        typename internal::genTen<T,DIMS...>::type operator[](size_t b) const { assert(b < batch_size); typename internal::genTen<T,DIMS...>::type t(data+b*internal::genTen<T,DIMS...>::type::SIZE); return t; }
        // will only reshape the internal Tensor. batch_size will stay the same
        template <std::size_t ...NDIMS>
        BatchedTensor<T,NDIMS...> reshape(){static_assert(Tensor<T,DIMS...>::SIZE == Tensor<T,NDIMS...>::SIZE,"SIZE of reshaped Tensor has to stay the same!"); BatchedTensor<T,NDIMS...> t(batch_size,data); return t; }
        template <std::size_t ...NDIMS>
        BatchedTensor<T,NDIMS...> reshape(size_t new_batch_size){ assert(this->SIZE == (new_batch_size * Tensor<T,NDIMS...>::SIZE)); BatchedTensor<T,NDIMS...> t(new_batch_size,data); return t; }
        void print(int precision=5){
            std::cout <<std::fixed<<std::setprecision(precision)<< *this <<std::endl;
        }
        
    };

    template <typename T, std::size_t... DIMS>
    std::ostream& operator<<(std::ostream& os, const BatchedTensor<T,DIMS...>& t)
    {
        os << "[";
        for(size_t d=0; d < t.batch_size; d++){
            if(d + 1 == t.batch_size) os <<  t[d] ;
            else  os <<  t[d] << ",";
        }
        os << "]";
        return os;
    }
} // namespace tel