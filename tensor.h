#ifndef DENSE_H
#define DENSE_H


#ifndef NDEBUG
#include <cstdlib>
#include <type_traits>
#include <array>
#include "shape.h"
#include "dense_base.h"
#include "map.h"
#endif


namespace tsr {

 
  template <typename T, size_t... dims>
  class Tensor: public DenseBase<Tensor<T, dims...>> {
    //friend class TensorBase<Tensor>;
    // friend class DenseBase<Tensor>;

    using Parent = DenseBase<Tensor>;

  public:
    using typename Parent::Scalar;
    using typename Parent::Shape;
    
  private:
    Scalar data[Shape::size];
    // std::array<Scalar, Shape::size> data;
    
  public:
    constexpr Scalar* get_data() {return data;}
    constexpr const Scalar* get_data() const {return data;}

    constexpr Scalar& sequence_ref(size_t n) {return data[n];}
    constexpr const Scalar& sequence_ref(size_t n) const {return data[n];}

  public:
    // Can not use initializer_list for array initialization,
    // eg: `Tensor(...): data{l} {}`;
    constexpr Tensor(const std::initializer_list<Scalar>& l) {
      Scalar* it1 = data;
      auto it2 = l.begin();
      Unroll<0, Shape::size>::map([&it1, &it2](size_t)
      {*it1++ = *it2++;});
    }
    // Default copy constructor is always synthetized!
    // This is for constructing from other TensorBases objects.
    // Note that data needs to be initialized here (until C++20),
    // See discussion on: https://stackoverflow.com/questions/59662404/legitimate-to-initialize-an-array-in-a-constexpr-constructor
    template<typename U>
    constexpr Tensor(const TensorBase<U>& t): data{} {operator=(t);}
    
    template<typename E,
             std::enable_if_t<std::is_convertible<E, Scalar>::value, int> = 0>
    constexpr Tensor(const E& e): data{e} {}
    // The following constructor is enabled only when the number
    
    // of input arguments is not equal to 1 or the input argument
    // is not a TensorBase type.
    template <typename... E,
              std::enable_if_t<sizeof...(E) != 1, int> = 0
              >
    constexpr Tensor(const E&... e): data{e...} {}

    using Parent::operator=;    
    
    constexpr auto operator[](size_t n)
    {return typename internal::map_slice<Scalar, dims...>::type
        (&data[Shape::stride * n]);}

    constexpr auto operator[](size_t n) const
    {return typename internal::map_slice<const Scalar, dims...>::type
        (&data[Shape::stride * n]);}


    // Additional class methods
    
    template<typename U>
    constexpr Tensor& operator+=(const TensorBase<U>& rhs) {
      *this = *this + rhs;
      return *this;
    }
    template<typename U>
    constexpr Tensor& operator-=(const TensorBase<U>& rhs) {
      *this = *this - rhs;
      return *this;
    }
    template<typename U>
    constexpr Tensor& operator*=(const TensorBase<U>& rhs) {
      *this = *this * rhs;
      return *this;
    }
    template<typename U>
    constexpr Tensor& operator/=(const TensorBase<U>& rhs) {
      *this = *this / rhs;
      return *this;
    }
    template<typename U>
    constexpr Tensor& operator%=(const TensorBase<U>& rhs) {
      *this = *this % rhs;
      return *this;
    }

    // Static functions for the class

    static constexpr Tensor linspace(const Scalar& start, const Scalar& step) {
      Tensor res;
#ifdef TSR_UNROLL
      Scalar value = start;
      Unroll<0, Shape::size>::map([&](size_t i) constexpr {
        res.sequence_ref(i) = value;
        value += step;
      });
#else
      Scalar value = start;
      for (size_t i = 0; i != Shape::size; ++i) {
        res.sequence_ref(i) = value;
        value += step;
      }
#endif
      return res;
    }
    
  };


  template <typename T, size_t... dims>
  struct BaseTraits<Tensor<T, dims...>> {
    using Shape = internal::ShapeType<dims...>;
    using Scalar = T;
  };
  
  
} // namespace tsr

#endif
