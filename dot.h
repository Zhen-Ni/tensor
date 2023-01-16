#ifndef DOT_H
#define DOT_H

#ifndef NDEBUG
#include "shape.h"
#include "tensor.h"
#endif

namespace tsr {
  
  template<typename Lhs, typename Rhs,
           // Make sure the two operands have the same data type
           // and the dimensions are correct.
           std::enable_if_t<
             std::is_same<typename Lhs::Scalar, typename Rhs::Scalar>::value &&
             internal::shape_rev_t<typename Lhs::Shape>::size0 == Rhs::Shape::size0
                            , int> = 0>
  constexpr auto dot(const TensorBase<Lhs>& lhs,
                     const TensorBase<Rhs>& rhs) {
    using Shape1 = typename internal::shape_rev_t<typename internal::shape_rev_t<typename Lhs::Shape>::ContainedType>;
    using Shape2 = typename Rhs::Shape::ContainedType;
    using Shape = internal::shape_cat_t<Shape1, Shape2>;
    using Scalar = typename Lhs::Scalar;
    using Res = typename internal::get_tensor_t<Scalar, Shape>;
    constexpr size_t squeeze_size = Rhs::Shape::size0;
    
    Res res;
    for (size_t i = 0; i != Res::Shape::size; ++i) {
      Scalar sum{};
      for (size_t j = 0; j != squeeze_size; ++j) {
        sum += lhs.sequence(squeeze_size * (i / Shape2::size) + j) *
          rhs.sequence(Shape2::size * j + i % Shape2::size);
      }
      res.sequence_ref(i) = sum;
    }
    
    return res;
  }
  
}    // end of namespace tsr
 

#endif    // end of DOT_H
