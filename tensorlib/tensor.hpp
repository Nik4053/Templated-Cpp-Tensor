/* Copyright (C) 2024 Niklas Heidenreich - All Rights Reserved */
#pragma once

// TODO find supported cpp versions
// #if __cplusplus <= 199711L
//   #error This library needs at least a C++11 compliant compiler
// #endif

// TODO test performance of expression for loops vs std::max_element, std::min_element, std::accumulate


#include <iosfwd>
#include <assert.h>
#include <iomanip>      // std::setprecision
#include <numeric>      // std::accumulate
#include <algorithm>    // std::max_element, std::min_element

#include <cmath>        // std::sin, std::cos, std::tan, std::asin, std::acos, std::atan, std::atan2, std::sinh, std::cosh, std::tanh, std::exp, std::log, std::log10, std::sqrt, std::abs, std::pow, std::fmod, std::remainder, std::copysign, std::nextafter
// using namespace std;
// #ifndef NDEBUG
// #warning "Tensor class compiled in DEBUG mode! Performance may be slow!"
// #endif

#ifndef __CUDACC__
#define HOSTDEVICE
#else
#define HOSTDEVICE __host__ __device__
#endif

namespace tel{
  template <typename T, std::size_t... DIMS>
  class Tensor;
  /*
      Internal helpers to get the Tensor class working
  */
  namespace internal{
    template <typename T, std::size_t ... DIMS>
    struct genTen;
    
    template <typename T, std::size_t DIM>
    struct genTen<T,DIM>
    {
        using type = Tensor<T,DIM>;
    };
    template <typename T, std::size_t... DIMS>
    struct genTen
    {
        using type = Tensor<T,DIMS...>;
    };


    // expressions
    namespace expressions{
      // base expression
      
      template <class A, size_t SIZE> struct TExpr{
        inline const A& operator~() const{
          return static_cast<const A&>(*this);
        }
        auto operator[](int i) const { return this->operator~()[i]; }; // TODO could this be done with decltype(auto) ?
        auto max() const { auto tmp = this->operator~()[0]; for(int i=1;i<SIZE;++i) tmp=std::max(tmp,this->operator~()[i]); return tmp; }
        auto min() const { auto tmp = this->operator~()[0]; for(int i=1;i<SIZE;++i) tmp=std::min(tmp,this->operator~()[i]); return tmp; }
        auto sum() const { auto tmp = this->operator~()[0]; for(int i=1;i<SIZE;++i) tmp+=this->operator~()[i]; return tmp; }
        template <class B>
        auto dot(const TExpr<B,SIZE>& b) const { return (*this * b).sum(); }
      };

      // function support: support for sin, cos etc. functions
      template <typename T, class A,  size_t SIZE>
      class TExprFunc : public TExpr<TExprFunc<T,A,SIZE>,SIZE >{
        const A a_;
        T (*func)(T);
      public:
        TExprFunc(T (*func)(T),const A& a) : a_(a) , func(func) {}
        auto operator[](int i) const {
          return func(a_[i]); 
        };
      };

      // used for single values. Multiply array by 3 etc.
      template <typename T, size_t SIZE>
      class TValueExpr : public TExpr<TValueExpr<T,SIZE>,SIZE>{
        public:
        T val = -1;
        TValueExpr(const T val) :val(val) {}
        TValueExpr(const TValueExpr<T,1>& expr) : val(expr.val) {}
        T operator[](int i) const {
          return val; 
        };
      };

      // helper for single value expression types. Multipy A by 3 can be written "A * Tval(3)"
      template <size_t SIZE, typename T>
      inline TValueExpr<T,SIZE> Tval(T val){
        return TValueExpr<T,SIZE>(val);
      }

      // --- Implementation of different operators
      // plus
      template <class A, class B,  size_t SIZE>
      class TExprPlus : public TExpr<TExprPlus<A,B,SIZE>,SIZE >{
        const A a_; const B b_;
      public:
        TExprPlus(const A& a, const B& b) : a_(a), b_(b){}
        auto operator[](int i) const {
          return a_[i] + b_[i]; 
        };
      };

      template <class A, class B, size_t SIZE>
      inline TExprPlus<A,B,SIZE> operator+ (const TExpr<A,SIZE>& a, const TExpr<B,SIZE>& b){
        return TExprPlus<A,B,SIZE>(~a,~b);
      }

      // minus
      template <class A, class B,  size_t SIZE>
      class TExprMinus : public TExpr<TExprMinus<A,B,SIZE>,SIZE >{
        const A a_; const B b_;
      public:
        TExprMinus(const A& a, const B& b) : a_(a), b_(b){}
        auto operator[](int i) const {
          return a_[i] - b_[i]; 
        };
      };

      template <class A, class B, size_t SIZE>
      inline TExprMinus<A,B,SIZE> operator- (const TExpr<A,SIZE>& a, const TExpr<B,SIZE>& b){
        return TExprMinus<A,B,SIZE>(~a,~b);
      }

      // self minus ie.: return -a
      template <class A,  size_t SIZE>
      class TExprSelfMinus : public TExpr<TExprSelfMinus<A,SIZE>,SIZE >{
        const A a_;
      public:
        TExprSelfMinus(const A& a) : a_(a) {}
        auto operator[](int i) const {
          return -a_[i]; 
        };
      };

      template <class A, size_t SIZE>
      inline TExprSelfMinus<A,SIZE> operator- (const TExpr<A,SIZE>& a){
        return TExprSelfMinus<A,SIZE>(~a);
      }

      // mul
      template <class A, class B,  size_t SIZE>
      class TExprMul : public TExpr<TExprMul<A,B,SIZE>,SIZE >{
        const A a_; const B b_;
      public:
        TExprMul(const A& a, const B& b) : a_(a), b_(b){}
        auto operator[](int i) const {
          return a_[i] * b_[i]; 
        };
      };

      template <class A, class B, size_t SIZE>
      inline TExprMul<A,B,SIZE> operator* (const TExpr<A,SIZE>& a, const TExpr<B,SIZE>& b){
        return TExprMul<A,B,SIZE>(~a,~b);
      }

      // div
      template <class A, class B,  size_t SIZE>
      class TExprDiv : public TExpr<TExprDiv<A,B,SIZE>,SIZE >{
        const A a_; const B b_;
      public:
        TExprDiv(const A& a, const B& b) : a_(a), b_(b){}
        auto operator[](int i) const {
          return a_[i] / b_[i]; 
        };
      };

      template <class A, class B, size_t SIZE>
      inline TExprDiv<A,B,SIZE> operator/ (const TExpr<A,SIZE>& a, const TExpr<B,SIZE>& b){
        return TExprDiv<A,B,SIZE>(~a,~b);
      }

      // simple type artihmetic operators
      // plus 
      template <class T,typename = std::enable_if_t<std::is_arithmetic<T>::value>, class B, size_t SIZE>
      inline TExprPlus<TValueExpr<T,SIZE>,B,SIZE> operator+ (const T t, const TExpr<B,SIZE>& b){
        return TExprPlus<TValueExpr<T,SIZE>,B,SIZE>(~Tval<SIZE>(t),~b);
      }

      template <class A, class T,typename = std::enable_if_t<std::is_arithmetic<T>::value>, size_t SIZE>
      inline TExprPlus<A,TValueExpr<T,SIZE>,SIZE> operator+ (const TExpr<A,SIZE>& a, const T t){
        return TExprPlus<A,TValueExpr<T,SIZE>,SIZE>(~a,~Tval<SIZE>(t));
      }
      // minus 
      template <class T,typename = std::enable_if_t<std::is_arithmetic<T>::value>, class B, size_t SIZE>
      inline TExprMinus<TValueExpr<T,SIZE>,B,SIZE> operator- (const T t, const TExpr<B,SIZE>& b){
        return TExprMinus<TValueExpr<T,SIZE>,B,SIZE>(~Tval<SIZE>(t),~b);
      }

      template <class A, class T,typename = std::enable_if_t<std::is_arithmetic<T>::value>, size_t SIZE>
      inline TExprMinus<A,TValueExpr<T,SIZE>,SIZE> operator- (const TExpr<A,SIZE>& a, const T t){
        return TExprMinus<A,TValueExpr<T,SIZE>,SIZE>(~a,~Tval<SIZE>(t));
      }
      // mul
      template <class T,typename = std::enable_if_t<std::is_arithmetic<T>::value>, class B, size_t SIZE>
      inline TExprMul<TValueExpr<T,SIZE>,B,SIZE> operator* (const T t, const TExpr<B,SIZE>& b){
        return TExprMul<TValueExpr<T,SIZE>,B,SIZE>(~Tval<SIZE>(t),~b);
      }

      template <class A, class T,typename = std::enable_if_t<std::is_arithmetic<T>::value>, size_t SIZE>
      inline TExprMul<A,TValueExpr<T,SIZE>,SIZE> operator* (const TExpr<A,SIZE>& a, const T t){
        return TExprMul<A,TValueExpr<T,SIZE>,SIZE>(~a,~Tval<SIZE>(t));
      }
      // div
      template <class T,typename = std::enable_if_t<std::is_arithmetic<T>::value>, class B, size_t SIZE>
      inline TExprDiv<TValueExpr<T,SIZE>,B,SIZE> operator/ (const T t, const TExpr<B,SIZE>& b){
        return TExprDiv<TValueExpr<T,SIZE>,B,SIZE>(~Tval<SIZE>(t),~b);
      }

      template <class A, class T,typename = std::enable_if_t<std::is_arithmetic<T>::value>, size_t SIZE>
      inline TExprDiv<A,TValueExpr<T,SIZE>,SIZE> operator/ (const TExpr<A,SIZE>& a, const T t){
        return TExprDiv<A,TValueExpr<T,SIZE>,SIZE>(~a,~Tval<SIZE>(t));
      }
    }
  };

  template<class T>
  struct TFunc{
      T (*func) (T);
      TFunc(T (*func) (T)){
          this->func = func;
      }
      template <class A,  size_t SIZE>
      auto operator()(const internal::expressions::TExpr<A,SIZE>& a) const {
          return internal::expressions::TExprFunc<T,A,SIZE>(func, ~a);
      };
  };

  template <typename T, std::size_t DIM, std::size_t... DIMS>
  class Tensor<T, DIM, DIMS...> : public internal::expressions::TExpr<Tensor<T, (DIM * ... * DIMS)>, (DIM * ... * DIMS)>
  {
  private:
  public:
    T *data = nullptr;
    constexpr static size_t RANK = 1 + sizeof...(DIMS);
    constexpr static size_t SIZE = (DIM * ... * DIMS);
    Tensor(T *data) : data(data) {}
    // Tensor(const Tensor<T,DIM,DIMS...>& other){ this->data = other.data;};
    // Tensor(const Tensor<T,SIZE>& other){ this->data = other.data;};
    template <typename T2, std::size_t DIM2, std::size_t... DIMS2, typename = std::enable_if_t<SIZE == Tensor<T2,DIM2,DIMS2...>::SIZE>>
    Tensor(const Tensor<T2,DIM2,DIMS2...>& other){ this->data = other.data;};

    template <typename ...Args>
    constexpr static size_t idx(size_t x, Args ... args) { assert(x < DIM); 
      if constexpr (RANK == 1) return x;
      else return x*internal::genTen<T,DIMS...>::type::SIZE + internal::genTen<T,DIMS...>::type::idx(args...) ;
    }
    auto operator[](size_t x) const { assert(x < DIM);  
      if constexpr (RANK == 1) return data[x];
      else { typename internal::genTen<T,DIMS...>::type t(data+x*internal::genTen<T,DIMS...>::type::SIZE); return t; }
    }
    
    decltype(auto) operator[](size_t x) { assert(x < DIM); 
      if constexpr (RANK == 1) return data[x];
      else { typename internal::genTen<T,DIMS...>::type t(data+x*internal::genTen<T,DIMS...>::type::SIZE); return t; }
    }


    template <typename ...Args>
    HOSTDEVICE T operator()(size_t x, Args ... args) const {assert(x < DIM); 
      if constexpr (RANK == 1) return data[x];
      else return data[idx(x,args...)];
    }
    template <typename ...Args>
    HOSTDEVICE T& operator()(size_t x, Args ... args) {assert(x < DIM); 
      if constexpr (RANK == 1) return data[x];
      else return data[idx(x,args...)];
    }
    Tensor<T,SIZE> flatten() const {Tensor<T,SIZE> t(data); return t; }
    template <std::size_t ...NDIMS>
    Tensor<T,NDIMS...> reshape() const {static_assert(Tensor<T,NDIMS...>::SIZE == SIZE,"SIZE of reshaped Tensor has to stay the same!"); Tensor<T,NDIMS...> t(data); return t; }
    void set(const T& val){ for(size_t i=0; i < SIZE; i++) data[i] = val; }
    // void set(const Tensor<T,SIZE>& other) { this->flatten().set(other); } // moved to expressions
    template <class A>
    void set(const internal::expressions::TExpr<A,SIZE>& a) { for(int i=0;i<SIZE;++i) data[i] = (~a)[i]; }
    void setZero(){ const T zero = static_cast<T>(0); this->set(zero); }
    // TODO: Remove *= operators. Use set() instead. Reason: Confusing behavior. A = B or A = A * 2 breaks. A *= 2 works. MAYBE all assignments should be done with set() to avoid confusion.
    // +=, -=, *=, /= operators for simple types
    Tensor& operator+=(const T& rhs){ set(*this + rhs); return *this; }
    Tensor& operator-=(const T& rhs){ set(*this - rhs); return *this; }
    Tensor& operator*=(const T& rhs){ set(*this * rhs); return *this; }
    Tensor& operator/=(const T& rhs){ set(*this / rhs); return *this; }
    // +=, -=, *=, /= operators for Expressions
    template <class A>
    Tensor& operator+=(const internal::expressions::TExpr<A,SIZE>& rhs){ set(*this + rhs); return *this; }
    template <class A>
    Tensor& operator-=(const internal::expressions::TExpr<A,SIZE>& rhs){ set(*this - rhs); return *this; }
    template <class A>
    Tensor& operator*=(const internal::expressions::TExpr<A,SIZE>& rhs){ set(*this * rhs); return *this; }
    template <class A>
    Tensor& operator/=(const internal::expressions::TExpr<A,SIZE>& rhs){ set(*this / rhs); return *this; }

    void print(int precision=5){ std::cout<<std::setprecision(precision)<< *this <<std::endl; }
    /*
    DELETE COPY ASSIGNMENT OPERATOR. This is to avoid confusion. If you want to copy a Tensor use set() instead.
    Would allow A = expr (would copy all elements to A) and A = B (would only copy the pointer and discard pointer to data of A). This results in confusing behavior. Use set() instead.
    */
    void operator=(const Tensor&) = delete;

    /*
      Copies the unpadded Tensor into this Tensor. The unpadded Tensor has to be smaller or equal to this Tensor.
      The unpadded Tensor will be centered in this Tensor. The padding will be left untouched.
      Fails via assert if centering is not possible. (i.e. NewDim - OldDim is not even)
    */
    template <std::size_t ODIM, std::size_t ...ODIMS>
    void copyAsPadded(const Tensor<T,ODIM,ODIMS...>& unpaddedT) 
    {
      static_assert(DIM >= ODIM,"DIM >= ODIM");
      static_assert((DIM - ODIM) % 2 == 0,"(DIM - ODIM) % 2 == 0");
      const size_t padding = (DIM - ODIM)/2;
      for(size_t i=0; i < ODIM; i++){
        if constexpr (RANK == 1) this->operator[](padding+i) = unpaddedT[i];
        else this->operator[](padding+i).copyAsPadded(unpaddedT[i]);
      }
    }
    /*
      Same as copyAsPadded but returns a new Tensor instead of modifying this Tensor.
      The size of paddedData has to be big enough to hold the new Tensor.
    */
    template <std::size_t PDIM, std::size_t ...PDIMS>
    Tensor<T,PDIM,PDIMS...> createPaddedCopy( T* paddedData) const
    {
      static_assert(PDIM >= DIM,"PDIM >= DIM");
      static_assert((PDIM - DIM) % 2 == 0,"(PDIM - DIM) % 2 == 0");
      const size_t padding = (PDIM - DIM)/2;
      Tensor<T,PDIM,PDIMS...> pt(paddedData);
      pt.copyAsPadded(*this);
      return pt;
    }
  
    // true if the two Tensors overlap in memory. Two Tensors of size 0 never overlap.
    template <typename T2, std::size_t SIZE2>
    bool overlaps(const Tensor<T2,SIZE2>& other) const { return ( other.data >= data && other.data < data + SIZE ) || ( data >= other.data && data < other.data + other.SIZE ); }
  };

  // --- Equality operators: Check  if two TExpr/Tensors are equal. They are equal if all elements are equal.
  // equality operators for simple types
  template <class T, class A, size_t SIZE,typename = std::enable_if_t<std::is_arithmetic<T>::value>>
  bool operator==(const internal::expressions::TExpr<A,SIZE>& a, const T rhs){ for (size_t i =0; i< SIZE; i++) { if(a[i]!=rhs)return false;}; return true; }
  template <class T, class A, size_t SIZE,typename = std::enable_if_t<std::is_arithmetic<T>::value>>
  bool operator!=(const internal::expressions::TExpr<A,SIZE>& a, const T rhs){ return !(a == rhs); }
  // equality operators for simple types (reverse)
  template <class T, class A, size_t SIZE,typename = std::enable_if_t<std::is_arithmetic<T>::value>>
  bool operator==(const T rhs, const internal::expressions::TExpr<A,SIZE>& a){ for (size_t i =0; i< SIZE; i++) { if(a[i]!=rhs)return false;}; return true; }
  template <class T, class A, size_t SIZE,typename = std::enable_if_t<std::is_arithmetic<T>::value>>
  bool operator!=(const T rhs, const internal::expressions::TExpr<A,SIZE>& a){ return !(rhs == a); }
  // equality operators for expressions
  template <class A, class B, size_t SIZE>
  bool operator==(const internal::expressions::TExpr<A,SIZE>& a, const internal::expressions::TExpr<B,SIZE>& b){ for (size_t i =0; i< SIZE; i++) { if(a[i]!=b[i])return false;}; return true; }
  template <class A, class B, size_t SIZE>
  bool operator!=(const internal::expressions::TExpr<A,SIZE>& a, const internal::expressions::TExpr<B,SIZE>& b){ return !(a == b); }
  // equality operators for Tensors
  template <typename T, std::size_t SIZE>
  bool operator==(const Tensor<T, SIZE>& a, const Tensor<T, SIZE>& b){ if(a.data == b.data) return true; for (size_t i =0; i< SIZE; i++) { if(a[i]!=b[i])return false;}; return true; }
  template <typename T, std::size_t SIZE>
  bool operator!=(const Tensor<T, SIZE>& a, const Tensor<T, SIZE>& b){ return !(a == b); }


  template <typename T, std::size_t DIM, std::size_t... DIMS>
  std::ostream& operator<<(std::ostream& os, const Tensor<T,DIM,DIMS...>& t)
  {
    os << "[";
    for(size_t d=0; d < DIM; d++){
      if(d + 1 == DIM) os << t[d];
      else  os << t[d] << ",";
    }
    os << "]";
    return os;
  }

  template <typename A, std::size_t SIZE>
  std::ostream& operator<<(std::ostream& os, const internal::expressions::TExpr<A,SIZE>& expr)
  {
    os << "[";
    for(size_t d=0; d < SIZE; d++){
      if(d + 1 == SIZE) os << expr[d];
      else  os << expr[d] << ",";
    }
    os << "]";
    return os;
  }

  // --- Math operations
  // --- Simple Math operations
  // sum
  template <class A, size_t SIZE>
  auto sum(const internal::expressions::TExpr<A,SIZE>& a) { return a.sum(); }
  // max
  template <class A, size_t SIZE>
  auto max(const internal::expressions::TExpr<A,SIZE>& a) { return a.max(); }
  // min
  template <class A, size_t SIZE>
  auto min(const internal::expressions::TExpr<A,SIZE>& a) { return a.min(); }
  // dot
  template <class A, class B, size_t SIZE>
  auto dot(const internal::expressions::TExpr<A,SIZE>& a, const internal::expressions::TExpr<B,SIZE>& b) { return a.dot(b); }
  // sin
  template <class A, size_t SIZE>
  auto sin(const internal::expressions::TExpr<A,SIZE>& a) { return TFunc<float>(std::sin)(a); }
  // cos
  template <class A, size_t SIZE>
  auto cos(const internal::expressions::TExpr<A,SIZE>& a) { return TFunc<float>(std::cos)(a); }
  // tan
  template <class A, size_t SIZE>
  auto tan(const internal::expressions::TExpr<A,SIZE>& a) { return TFunc<float>(std::tan)(a); }
  // asin
  template <class A, size_t SIZE>
  auto asin(const internal::expressions::TExpr<A,SIZE>& a) { return TFunc<float>(std::asin)(a); }
  // acos
  template <class A, size_t SIZE>
  auto acos(const internal::expressions::TExpr<A,SIZE>& a) { return TFunc<float>(std::acos)(a); }
  // atan
  template <class A, size_t SIZE>
  auto atan(const internal::expressions::TExpr<A,SIZE>& a) { return TFunc<float>(std::atan)(a); }
  // atan2
  template <class A, class B, size_t SIZE>
  auto atan2(const internal::expressions::TExpr<A,SIZE>& a, const internal::expressions::TExpr<B,SIZE>& b) { return TFunc<float>(std::atan2)(a,b); }
  // sinh
  template <class A, size_t SIZE>
  auto sinh(const internal::expressions::TExpr<A,SIZE>& a) { return TFunc<float>(std::sinh)(a); }
  // cosh
  template <class A, size_t SIZE>
  auto cosh(const internal::expressions::TExpr<A,SIZE>& a) { return TFunc<float>(std::cosh)(a); }
  // tanh
  template <class A, size_t SIZE>
  auto tanh(const internal::expressions::TExpr<A,SIZE>& a) { return TFunc<float>(std::tanh)(a); }
  // exp
  template <class A, size_t SIZE>
  auto exp(const internal::expressions::TExpr<A,SIZE>& a) { return TFunc<float>(std::exp)(a); }
  // log
  template <class A, size_t SIZE>
  auto log(const internal::expressions::TExpr<A,SIZE>& a) { return TFunc<float>(std::log)(a); }
  // log10
  template <class A, size_t SIZE>
  auto log10(const internal::expressions::TExpr<A,SIZE>& a) { return TFunc<float>(std::log10)(a); }
  // sqrt
  template <class A, size_t SIZE>
  auto sqrt(const internal::expressions::TExpr<A,SIZE>& a) { return TFunc<float>(std::sqrt)(a); }
  // abs
  template <class A, size_t SIZE>
  auto abs(const internal::expressions::TExpr<A,SIZE>& a) { return TFunc<float>(std::abs)(a); }
  // pow
  template <class A, class B, size_t SIZE>
  auto pow(const internal::expressions::TExpr<A,SIZE>& a, const internal::expressions::TExpr<B,SIZE>& b) { return TFunc<float>(std::pow)(a,b); }


  // --- Complex Math operations
  // matmul expression (inefficient, use Matmul function instead)

  // template <typename T, std::size_t DIM_Ay, std::size_t DIM_Ax, std::size_t DIM_Bx>
  // class TExprMatmul : public internal::expressions::TExpr<TExprMatmul<T,DIM_Ay,DIM_Ax,DIM_Bx>,DIM_Ay*DIM_Bx >{
  //   const Tensor<T,DIM_Ay,DIM_Ax> a_; const Tensor<T,DIM_Ax,DIM_Bx> b_;
  // public:
  //   TExprMatmul(const Tensor<T,DIM_Ay,DIM_Ax>& a, const Tensor<T,DIM_Ax,DIM_Bx>& b) : a_(a), b_(b){}
  //   auto operator[](int i) const {
  //     int x = i % DIM_Bx;
  //     int y = i / DIM_Bx;
  //     T sum = 0;
  //     for(int j=0; j < DIM_Ax; j++){
  //       sum += a_[y][j] * b_[j][x];
  //     }
  //     return sum;
  //   };
  // };

  // template <typename T, std::size_t DIM_Ay, std::size_t DIM_Ax, std::size_t DIM_Bx>
  // inline TExprMatmul<T,DIM_Ay,DIM_Ax,DIM_Bx> Matmul(const Tensor<T,DIM_Ay,DIM_Ax>& a, const Tensor<T,DIM_Ax,DIM_Bx>& b){
  //   return TExprMatmul<T,DIM_Ay,DIM_Ax,DIM_Bx>(a,b);
  // }

  // matmul

  template <typename T, std::size_t DIM_Ay, std::size_t DIM_Ax, std::size_t DIM_Bx>
  static Tensor<T,DIM_Ay,DIM_Bx> Matmul( const Tensor<T,DIM_Ay,DIM_Ax>& A, const Tensor<T,DIM_Ax,DIM_Bx>& B, Tensor<T,DIM_Ay*DIM_Bx>& C, bool zeroC = true)
  {   
    if(A.overlaps(C) == true) throw std::runtime_error("Matmul: A and C overlap in memory. This is not supported.");
    if(B.overlaps(C) == true) throw std::runtime_error("Matmul: B and C overlap in memory. This is not supported.");

    if(zeroC) C.setZero();
    Tensor<T,DIM_Ay,DIM_Bx> result(C);
    // for each element in A
    for(size_t ay=0; ay < DIM_Ay; ay++){
        for (size_t ax = 0; ax < DIM_Ax; ax++)
        {
            T a = A(ay,ax);
            // multiply with each element in Bs corresponding row
            #pragma omp simd
            for (size_t bx = 0; bx < DIM_Bx; bx++)
            {
                result(ay,bx) += a * B(ax,bx);
            }
        }           
    }
    return result;
  }
  template <typename T, std::size_t DIM_Ay, std::size_t DIM_Ax, std::size_t DIM_Bx>
  static Tensor<T,DIM_Ay,DIM_Bx> Matmul( const Tensor<T,DIM_Ay,DIM_Ax>& A, const Tensor<T,DIM_Ax,DIM_Bx>& B, Tensor<T,DIM_Ay,DIM_Bx>& C, bool zeroC = true)
  {   
    Tensor<T,DIM_Ay*DIM_Bx> C_flatten(C.data);
    return Matmul(A,B,C_flatten,zeroC);
  }

  template <typename T, std::size_t DIM>
  static Tensor<T,DIM,DIM> Transpose(Tensor<T,DIM,DIM> t)
  {
      for(size_t i=0; i < DIM; i++){
          for(size_t j=i; j < DIM; j++){
              T temp = t(i,j);
              t(i,j) = t(j,i);
              t(j,i) = temp;
          }
      }
      return t;
  }

} // namespace tel
