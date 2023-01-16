#ifndef DOT_H
#define DOT_H

#ifndef NDEBUG
#include "shape.h"
#include "tensor.h"
#endif

namespace tsr {
  
  template<typename Lhs, typename Rhs>
  constexpr auto dot(const TensorBase<Lhs>& lhs,
                     const TensorBase<Rhs>& rhs) {
    // Make sure the shape of the two tensors are correct.
    static_assert(internal::shape_rev_t<typename Lhs::Shape>::size0 ==
                  Rhs::Shape::size0,
                  "tensor shape not correct for dot");
    
    using Shape1 = typename internal::shape_rev_t<typename internal::shape_rev_t<typename Lhs::Shape>::ContainedType>;
    using Shape2 = typename Rhs::Shape::ContainedType;
    using Shape = internal::shape_cat_t<Shape1, Shape2>;
    using Scalar = decltype(typename Lhs::Scalar() *
                            typename Rhs::Scalar());
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
