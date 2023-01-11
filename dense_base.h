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

    constexpr Scalar sequence(size_t n) {return sequence_ref(n);}
    constexpr const Scalar sequence(size_t n) const {return sequence_ref(n);}

    constexpr Scalar& sequence_ref(size_t n) {return get_derived()->sequence_ref(n);}
    constexpr const Scalar& sequence_ref(size_t n) const {return get_derived()->sequence_ref(n);}

    constexpr auto operator[](size_t n) {return get_derived()->operator[](n);}
    constexpr auto operator[](size_t n) const {return get_derived()->operator[](n);}

    // Deleted operator= also participates in overload resolution,
    // see https://stackoverflow.com/questions/14085620/why-do-c11-deleted-functions-participate-in-overload-resolution.
    // Thus, we need to explicitly implement this member function,
    // by calling the template version.
    DenseBase& operator=(const DenseBase& other) {
      return operator=<Derived>(other);
    }
    
    template<typename U>
    Derived& operator=(const TensorBase<U>& other) {
      Unroll<0, Shape::get_size()>::
        map([&](size_t index, const TensorBase<U>& y)
            {sequence_ref(index)=y.sequence(index);},
          other);
      return *get_derived();
    }
    
    template<typename U>
    Derived& operator=(const DenseBase<U>& other) {
      Unroll<0, Shape::get_size()>::
        map([&](size_t index, const DenseBase<U>& y)
            {sequence_ref(index)=y.sequence_ref(index);},
          other);
      return *get_derived();
    } 
    
    template<typename... Index,
             // Make sure the number of arguments is correct
             std::enable_if_t<
               std::integral_constant<bool,
                                      !(sizeof...(Index)-Shape::get_dimension())
                                      >::value, int> =0
             >
    constexpr auto& operator()(Index... index) {
      return sequence_ref(internal::decode_index<Shape>(index...));
    }
    template<typename... Index,
             // Make sure the number of arguments is correct
             std::enable_if_t<
               std::integral_constant<bool,
                                      !(sizeof...(Index)-Shape::get_dimension())
                                      >::value, int> =0
             >
    constexpr const auto& operator()(Index... index) const {
      return sequence_ref(internal::decode_index<Shape>(index...));
    }
    
  };

  
} // namespace tsr

#endif
