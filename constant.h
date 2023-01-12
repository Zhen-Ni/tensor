#ifndef CONSTANT_H
#define CONSTANT_H


#ifndef NDEBUG
#include "tensor"
#endif

namespace tsr {
  template <typename T, size_t... dims>
  class Constant: public TensorBase<Constant<T, dims...>> {

    using Traits = BaseTraits<Constant>;
  public:
    using Shape = typename Traits::Shape;
    using Scalar = typename Traits::Scalar;

  private:
    const Scalar value;
    
  public:
    constexpr Constant(const Scalar& value): value(value) {}
    constexpr Scalar sequence(size_t n) const {return value;}
  };

  template <typename T, size_t... dims>
  struct BaseTraits<Constant<T, dims...>> {
    using Shape = internal::ShapeType<dims...>;
    using Scalar = T;
  };

}

#endif  // CONSTANT_H
