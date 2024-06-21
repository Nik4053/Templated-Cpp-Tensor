#include <benchmark/benchmark.h>
#include "tensor.hpp"

using namespace tel;

#define DIM 127
float lhs[Tensor<float,DIM,DIM>::SIZE];
float rhs[Tensor<float,DIM,DIM>::SIZE];
float result[Tensor<float,DIM,DIM>::SIZE];

static void BM_Reference(benchmark::State& state) {

  for (auto _ : state){
    for(size_t i=0; i < DIM; i++){
        for(size_t j=0; j < DIM; j++){
            float tmp = 0;
            for(size_t k=0; k < DIM; k++){
               tmp += lhs[i*DIM+k] * rhs[k*DIM+j];
            }
            result[i*DIM+j] = tmp;
        }
    }
  }
}
// Register the function as a benchmark
BENCHMARK(BM_Reference);

static void BM_Reference_line(benchmark::State& state) {

  for (auto _ : state){
    for(size_t i=0; i < DIM; i++){
        for(size_t k=0; k < DIM; k++){
            float tmp = lhs[i*DIM+k];
            for(size_t j=0; j < DIM; j++){
               result[i*DIM+j] += tmp * rhs[k*DIM+j];
            }
        }
    }
  }
}
// Register the function as a benchmark
BENCHMARK(BM_Reference_line);

static void BM_Tensor(benchmark::State& state) {

    Tensor<float,DIM,DIM> lhs_t(lhs);
    Tensor<float,DIM,DIM> rhs_t(rhs);
    Tensor<float,DIM,DIM> result_t(result);
  for (auto _ : state){
    Matmul(result_t,lhs_t,rhs_t);
  }
}
BENCHMARK(BM_Tensor);

static void BM_Tensor_line(benchmark::State& state) {

    Tensor<float,DIM,DIM> lhs_t(lhs);
    Tensor<float,DIM,DIM> rhs_t(rhs);
    Tensor<float,DIM,DIM> result_t(result);
  for (auto _ : state){
    for(size_t i=0; i < DIM; i++){
        for(size_t k=0; k < DIM; k++){
            float tmp = lhs_t(i,k);
            for(size_t j=0; j < DIM; j++){
               result_t(i,j) += tmp * rhs_t(k,i);
            }
        }
    }
  }
}
BENCHMARK(BM_Tensor_line);

static void BM_Tensor_lineV2(benchmark::State& state) {

    Tensor<float,DIM,DIM> lhs_t(lhs);
    Tensor<float,DIM,DIM> rhs_t(rhs);
    Tensor<float,DIM,DIM> result_t(result);
  for (auto _ : state){
    for(size_t i=0; i < DIM; i++){
        for(size_t k=0; k < DIM; k++){
            float tmp = lhs_t[i][k];
            for(size_t j=0; j < DIM; j++){
               result_t[i][j] += tmp * rhs_t[k][j];
            }
        }
    }
  }
}
BENCHMARK(BM_Tensor_lineV2);

BENCHMARK_MAIN();