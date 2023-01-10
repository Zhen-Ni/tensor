#ifndef CONSTANT_H
#define CONSTANT_H


#ifndef NDEBUG
#include <cstdlib>
#include <type_traits>
#include "unroll.h"
#include "tensor_base.h"
#endif

namespace tsr {
  template <typename T, size_t... dims>
  class Constant: public TensorBase<Constant<T, dims...>> {

    using Traits = TensorTraits<Constant>;
  public:
    using Shape = typename Traits::Shape;
    using Scalar = typename Traits::Scalar;

  private:
    Scalar value;
    
  public:
    constexpr Constant() {}
    constexpr Constant(const Scalar& value): value(value) {}

    constexpr Scalar sequence(size_t n) const {return value;}
    template<typename... Index,
             std::enable_if_t<
               // Make sure the number of arguments is correct
               std::integral_constant<bool,
                                      !(sizeof...(Index)-Shape::get_dimension())
                                      >::value, int> =0
             >
    constexpr const auto& operator()(Index... index) const {
      return value;
    }
  };

  template <typename T, size_t... dims>
  struct TensorTraits<Constant<T, dims...>> {
    using Shape = ShapeType<dims...>;
    using Scalar = T;
  };

}

#endif  // CONSTANT_H
