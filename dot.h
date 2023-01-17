#ifndef DOT_H
#define DOT_H

#ifndef NDEBUG
#include "unroll.h"
#include "shape.h"
#include "tensor.h"
#endif

namespace tsr {

  template<size_t n = 1, typename Lhs, typename Rhs>
  constexpr auto ndot(const TensorBase<Lhs>& lhs,
                     const TensorBase<Rhs>& rhs) {
    // Make sure the shape of the two tensors are correct.
    static_assert(std::is_same<
                  typename internal::shape_get_last_n_t<typename Lhs::Shape, n>,
                  typename internal::shape_get_first_n_t<typename Rhs::Shape, n>
                  >::value,
                  "tensor shape not correct for ndot");
    
    using Shape1 = typename internal::shape_get_first_n_t<
      typename Lhs::Shape, Lhs::Shape::dimension - n>;
    using Shape2 = typename internal::shape_get_last_n_t<
      typename Rhs::Shape, Rhs::Shape::dimension - n>;
    using Shape = internal::shape_cat_t<Shape1, Shape2>;
    using Scalar = decltype(typename Lhs::Scalar() *
                            typename Rhs::Scalar());
    using Res = typename internal::get_tensor_t<Scalar, Shape>;
    constexpr size_t contraction_size = internal::shape_get_last_n_t<typename Lhs::Shape, n>::size;
    
    Res res;
    
    #ifdef TSR_UNROLL
    Unroll<0, Res::Shape::size>::map([&](size_t i) constexpr {
      Scalar sum{};
      Unroll<0, contraction_size>::map([&](size_t j) constexpr {
        sum += lhs.sequence(contraction_size * (i / Shape2::size) + j) *
          rhs.sequence(Shape2::size * j + i % Shape2::size);
      });
      res.sequence_ref(i) = sum;
    });
    #else
    for (size_t i = 0; i != Res::Shape::size; ++i) {
      Scalar sum{};
      for (size_t j = 0; j != contraction_size; ++j) {
        sum += lhs.sequence(contraction_size * (i / Shape2::size) + j) *
          rhs.sequence(Shape2::size * j + i % Shape2::size);
      }
      res.sequence_ref(i) = sum;
    }
    #endif
    
    return res;
  }


  template<typename Lhs, typename Rhs>
  constexpr auto kron(const TensorBase<Lhs>& lhs,
                      const TensorBase<Rhs>& rhs) {
    return ndot<0>(lhs, rhs);
  }
  
  template<typename Lhs, typename Rhs>
  constexpr auto dot(const TensorBase<Lhs>& lhs,
                     const TensorBase<Rhs>& rhs) {
    return ndot<1>(lhs, rhs);
  }

  template<typename Lhs, typename Rhs>
  constexpr auto ddot(const TensorBase<Lhs>& lhs,
                      const TensorBase<Rhs>& rhs) {
    return ndot<2>(lhs, rhs);
  }

  
}    // end of namespace tsr
 

#endif    // end of DOT_H
