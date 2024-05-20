/* Copyright (C) 2024 Niklas Heidenreich - All Rights Reserved */
#pragma once

// TODO find supported cpp versions
// #if __cplusplus <= 199711L
//   #error This library needs at least a C++11 compliant compiler
// #endif

// TODO test performance of expression for loops vs std::max_element, std::min_element, std::accumulate



#include <iostream>
#include <assert.h>
#include <iomanip>      // std::setprecision
#include <numeric>      // std::accumulate
#include <algorithm>    // std::max_element, std::min_element
// using namespace std;
// #ifndef NDEBUG
// #warning "Tensor class compiled in DEBUG mode! Performance may be slow!"
// #endif
template <typename T, std::size_t... DIMS>
class Tensor;

#ifndef __CUDACC__
#define HOSTDEVICE
#else
#define HOSTDEVICE __host__ __device__
#endif

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

template <typename T, std::size_t DIM>
class Tensor<T, DIM> : public internal::expressions::TExpr<Tensor<T, DIM>,DIM>{
private:
public:
  T *data = nullptr;
  constexpr static size_t SIZE = DIM;
  constexpr static size_t RANK = 1;
  template <std::size_t ...NDIMS>
  Tensor(const Tensor<float,NDIMS...>& other){ static_assert(SIZE == Tensor<float,NDIMS...>::SIZE); this->data = other.data;};
  Tensor(T *data) : data(data) {}
  constexpr static size_t idx(size_t x) { assert(x < DIM); return x; }
  HOSTDEVICE T operator[](size_t x) const {assert(x < DIM); return data[x];}
  HOSTDEVICE T& operator[](size_t x) {assert(x < DIM); return data[x];}
  HOSTDEVICE T operator()(size_t x) const {assert(x < DIM); return data[idx(x)];}
  HOSTDEVICE T& operator()(size_t x) {assert(x < DIM); return data[idx(x)];}
  template <std::size_t ...NDIMS>
  Tensor<T,NDIMS...> reshape() const{static_assert(Tensor<T,NDIMS...>::SIZE == SIZE,"SIZE of reshaped Tensor has to stay the same!"); Tensor<T,NDIMS...> t(data); return t; }
  void set(const T& val){ for(size_t i=0; i < SIZE; i++) data[i] = val; }
  // void set(const Tensor<T,DIM>& other) { for(size_t i=0; i < SIZE; i++) data[i] = other[i]; } // moved to expressions
  template <class A>
	void set(const internal::expressions::TExpr<A,DIM>& a) { for(int i=0;i<SIZE;++i) data[i] = (~a)[i]; }
  void setZero(){ const T zero = static_cast<T>(0); this->set(zero); }
  // T max() const { return *std::max_element(data, data + SIZE); } // moved to expressions
  // T min() const { return *std::min_element(data, data + SIZE); }
  // T sum() const { return std::accumulate(data, data + SIZE, 0.0); }
  bool operator==(const T& rhs){ for (size_t i =0; i< DIM; i++) { if(this->operator[](i)!=rhs)return false;}; return true; }
  bool operator==(const Tensor<T, DIM>& rhs){ for (size_t i =0; i< DIM; i++) { if(this->operator[](i)!=rhs[i])return false;}; return true; }
  Tensor<T, DIM>& operator+=(const T& rhs){ for (size_t i =0; i< DIM; i++) this->operator[](i)+=rhs; return *this; }
  Tensor<T, DIM>& operator+=(const Tensor<T, DIM>& rhs){ for (size_t i =0; i< DIM; i++) this->operator[](i)+=rhs[i]; return *this; }
  Tensor<T, DIM>& operator-=(const T& rhs){ for (size_t i =0; i< DIM; i++) this->operator[](i)-=rhs; return *this; }
  Tensor<T, DIM>& operator-=(const Tensor<T, DIM>& rhs){ for (size_t i =0; i< DIM; i++) this->operator[](i)-=rhs[i]; return *this; }
  Tensor<T, DIM>& operator*=(const T& rhs){ for (size_t i =0; i< DIM; i++) this->operator[](i)*=rhs; return *this; }
  Tensor<T, DIM>& operator*=(const Tensor<T, DIM>& rhs){ for (size_t i =0; i< DIM; i++) this->operator[](i)*=rhs[i]; return *this; }
  Tensor<T, DIM>& operator/=(const T& rhs){ for (size_t i =0; i< DIM; i++) this->operator[](i)/=rhs; return *this; }
  Tensor<T, DIM>& operator/=(const Tensor<T, DIM>& rhs){ for (size_t i =0; i< DIM; i++) this->operator[](i)/=rhs[i]; return *this; }
  template <class A>
  Tensor<T, DIM>& operator+=(const internal::expressions::TExpr<A,DIM>& rhs){ for (size_t i =0; i< DIM; i++) this->operator[](i)+=(~rhs)[i]; return *this; }
  template <class A>
  Tensor<T, DIM>& operator-=(const internal::expressions::TExpr<A,DIM>& rhs){ for (size_t i =0; i< DIM; i++) this->operator[](i)-=(~rhs)[i]; return *this; }
  template <class A>
  Tensor<T, DIM>& operator*=(const internal::expressions::TExpr<A,DIM>& rhs){ for (size_t i =0; i< DIM; i++) this->operator[](i)*=(~rhs)[i]; return *this; }
  template <class A>
  Tensor<T, DIM>& operator/=(const internal::expressions::TExpr<A,DIM>& rhs){ for (size_t i =0; i< DIM; i++) this->operator[](i)/=(~rhs)[i]; return *this; }
  /*
  DELETE COPY ASSIGNMENT OPERATOR. This is to avoid confusion. If you want to copy a Tensor use set() instead.
  Would allow A = expr (would copy all elements to A) and A = B (would only copy the pointer and discard pointer to data of A). This results in confusing behavior. Use set() instead.
  */
  void operator=(const Tensor<T, DIM>&) = delete;
  
  void print(int precision=5){ std::cout <<std::fixed<<std::setprecision(precision)<< *this <<std::endl;  }

  template <std::size_t ODIM>
  void copyAsPadded( const Tensor<T,ODIM>& unpaddedT) 
  {
    static_assert(DIM >= ODIM,"DIM >= ODIM");
    static_assert((DIM - ODIM) % 2 == 0,"(DIM - ODIM) % 2 == 0");
    const size_t padding = (DIM - ODIM)/2;
    for(size_t i=0; i < ODIM; i++){
      this->operator[](padding+i) = unpaddedT[i];
    }
  }
  template <std::size_t PDIM>
  Tensor<T,PDIM> createPaddedCopy( T* paddedData) const
  {
    static_assert(PDIM >= DIM,"PDIM >= DIM");
    static_assert((PDIM - DIM) % 2 == 0,"(PDIM - DIM) % 2 == 0");
    const size_t padding = (PDIM - DIM)/2;
    Tensor<T,PDIM> pt(paddedData);
    pt.copyAsPadded(*this);
    return pt;
  }
};


template <typename T, std::size_t DIM, std::size_t... DIMS>
class Tensor<T,DIM,DIMS...> : public internal::expressions::TExpr<Tensor<T, DIM * (DIMS * ...)>, DIM * (DIMS * ...)>{
private:
public:
	T *data = nullptr;
	constexpr static size_t SIZE = DIM * (DIMS * ...);
	constexpr static size_t RANK = 1 + sizeof...(DIMS);
	Tensor(T *data) : data(data) {}
  Tensor(const Tensor<T,DIM,DIMS...>& other){ this->data = other.data;};	
	template <typename ...Args>
	constexpr static size_t idx(size_t x, Args ... args) { assert(x < DIM); return x*internal::genTen<T,DIMS...>::type::SIZE + internal::genTen<T,DIMS...>::type::idx(args...) ; }
	typename internal::genTen<T,DIMS...>::type operator[](size_t x) const { assert(x < DIM);  typename internal::genTen<T,DIMS...>::type t(data+x*internal::genTen<T,DIMS...>::type::SIZE); return t; }
	template <typename ...Args>
  HOSTDEVICE T operator()(size_t x, Args ... args) const {assert(x < DIM); return data[idx(x,args...)];}
	template <typename ...Args>
  HOSTDEVICE T& operator()(size_t x, Args ... args) {assert(x < DIM); return data[idx(x, args...)];}
  Tensor<T,SIZE> flatten() const {Tensor<T,SIZE> t(data); return t; }
  template <std::size_t ...NDIMS>
  Tensor<T,NDIMS...> reshape() const {static_assert(Tensor<T,NDIMS...>::SIZE == SIZE,"SIZE of reshaped Tensor has to stay the same!"); Tensor<T,NDIMS...> t(data); return t; }
  void set(const T& val){ this->flatten().set(val); }
  // void set(const Tensor<T,SIZE>& other) { this->flatten().set(other); } // moved to expressions
  template <class A>
	void set(const internal::expressions::TExpr<A,SIZE>& a) { this->flatten().set(a); }
  void setZero(){ this->flatten().setZero(); }
  bool operator==(const T& rhs){ return this->flatten()==rhs; }
  bool operator==(const Tensor<T,SIZE>& rhs){ return this->flatten()==rhs;  }
  Tensor<T,DIM,DIMS...>& operator+=(const T& rhs){ this->flatten()+=rhs; return *this; }
  Tensor<T,DIM,DIMS...>& operator+=(const Tensor<T,SIZE>& rhs){ this->flatten()+=rhs; return *this; }
  Tensor<T,DIM,DIMS...>& operator-=(const T& rhs){ this->flatten()-=rhs; return *this; }
  Tensor<T,DIM,DIMS...>& operator-=(const Tensor<T,SIZE>& rhs){ this->flatten()-=rhs; return *this; }
  Tensor<T,DIM,DIMS...>& operator*=(const T& rhs){ this->flatten()*=rhs; return *this; }
  Tensor<T,DIM,DIMS...>& operator*=(const Tensor<T,SIZE>& rhs){ this->flatten()*=rhs; return *this; }
  Tensor<T,DIM,DIMS...>& operator/=(const T& rhs){ this->flatten()/=rhs; return *this; }
  Tensor<T,DIM,DIMS...>& operator/=(const Tensor<T,SIZE>& rhs){ this->flatten()/=rhs; return *this; }
  template <class A>
  Tensor<T,DIM,DIMS...>& operator+=(const internal::expressions::TExpr<A,SIZE>& rhs){ this->flatten()+=rhs; return *this; }
  template <class A>
  Tensor<T,DIM,DIMS...>& operator-=(const internal::expressions::TExpr<A,SIZE>& rhs){ this->flatten()-=rhs; return *this; }
  template <class A>
  Tensor<T,DIM,DIMS...>& operator*=(const internal::expressions::TExpr<A,SIZE>& rhs){ this->flatten()*=rhs; return *this; }
  template <class A>
  Tensor<T,DIM,DIMS...>& operator/=(const internal::expressions::TExpr<A,SIZE>& rhs){ this->flatten()/=rhs; return *this; }
  void print(int precision=5){ std::cout<<std::setprecision(precision)<< *this <<std::endl; }
  void operator=(const Tensor<T, DIM, DIMS...>&) = delete;
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
      auto tmp = unpaddedT[i];
      this->operator[](padding+i).copyAsPadded(tmp);
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
};

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
