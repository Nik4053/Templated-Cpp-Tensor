/* Copyright (C) 2024 Niklas Heidenreich - All Rights Reserved */
#pragma once
#include "../tensor.hpp"

namespace TensorHelper
{
    template <typename T, std::size_t DIM_Ay, std::size_t DIM_Ax, std::size_t DIM_Bx>
    static void Matmul( const Tensor<T,DIM_Ay,DIM_Ax>& A, const Tensor<T,DIM_Ax,DIM_Bx>& B, Tensor<T,DIM_Ay,DIM_Bx>& C, bool zeroC = true)
    {   
        if(zeroC) C.setZero();
        // for each element in A
        for(size_t ay=0; ay < DIM_Ay; ay++){
            for (size_t ax = 0; ax < DIM_Ax; ax++)
            {
                T a = A(ay,ax);
                // multiply with each element in Bs corresponding row
                for (size_t bx = 0; bx < DIM_Bx; bx++)
                {
                    C(ay,bx) += a * B(ax,bx);
                }
            }           
        }
    }
    template <typename T, std::size_t DIM>
    static void Transpose(Tensor<T,DIM,DIM>& t)
    {
        for(size_t i=0; i < DIM; i++){
            for(size_t j=i; j < DIM; j++){
                T temp = t(i,j);
                t(i,j) = t(j,i);
                t(j,i) = temp;
            }
        }
    }
    template <typename T, std::size_t DIM>
    static T Dot(const Tensor<T,DIM>& A, const Tensor<T,DIM>& B)
    {
        T c = static_cast<T>(0);
        for(size_t i=0; i < DIM; i++){
            c += A(i) * B(i);
        }
        return c;
    }
    
    // template <typename T, std::size_t DIM>
    // static void Dot(const Tensor<T,DIM>& A, const Tensor<T,DIM>& B, T& C)
    // {
    //     C = 0;
    //     for(size_t i=0; i < DIM; i++){
    //         C += A(i);
    //         C *= B(i);
    //     }
    // }
}

