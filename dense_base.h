#ifndef DENSE_BASE_H
#define DENSE_BASE_H


#ifndef NDEBUG
#include "shape.h"
#include "unroll.h"
#include "tensor_base.h"
#endif


namespace tsr {

  template <typename Derived>
  class DenseBase: public TensorBase<Derived> {
    using Base = TensorBase<Derived>;
    using typename Base::Traits;
  
  public:
    using Scalar = typename Traits::Scalar;
    using Shape = typename Traits::Shape;

  protected:
    using Base::get_derived;
    
  public:

    // constexpr Scalar sequence(size_t n) {return sequence_ref(n);}
    constexpr const Scalar sequence(size_t n) const {return sequence_ref(n);}

    constexpr Scalar& sequence_ref(size_t n) {return get_derived()->sequence_ref(n);}
    constexpr const Scalar& sequence_ref(size_t n) const {return get_derived()->sequence_ref(n);}

    constexpr auto operator[](size_t n) {return get_derived()->operator[](n);}
    constexpr auto operator[](size_t n) const {return get_derived()->operator[](n);}

    // Deleted operator= also participates in overload resolution,
    // see https://stackoverflow.com/questions/14085620/why-do-c11-deleted-functions-participate-in-overload-resolution.
    // Thus, we need to explicitly implement this member function,
    // by calling the template version.
    constexpr DenseBase& operator=(const DenseBase& other) {
      return operator=<Derived>(other);
    }
    
    template<typename U>
    constexpr Derived& operator=(const TensorBase<U>& other) {
      static_assert(std::is_same<Shape, typename U::Shape>::value,
                    "shapes of tensors for assigning should be the same");
#ifdef TSR_UNROLL
      Unroll<0, Shape::size>::
        map([&](size_t index, const TensorBase<U>& y) {
          sequence_ref(index)=y.sequence(index);},
          other);
#else   // #ifdef TSR_UNROLL
      for (size_t i = 0; i != Shape::size; ++i) {
        sequence_ref(i) = other.sequence(i);
      }
#endif  // #ifdef TSR_UNROLL
      return *get_derived();
    }
    
    template<typename... Index>
    constexpr auto& operator()(Index... index) {
      return sequence_ref(Shape::decode_index(index...));
    }
    template<typename... Index>
    constexpr const auto& operator()(Index... index) const {
      return sequence_ref(Shape::decode_index(index...));
    }
    
  };

  
} // namespace tsr

#endif
