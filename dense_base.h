#ifndef DENSE_BASE_H
#define DENSE_BASE_H


#ifndef NDEBUG
#include <cstdlib>
#include <type_traits>
#include "unroll.h"
#include "tensor_base.h"
#endif


namespace tsr {

  template <typename T, size_t... dims>
  class Tensor;


  template <typename Derived>
  class DenseBase: public TensorBase<Derived> {
    using Base = TensorBase<Derived>;
    using typename Base::Traits;
  
  public:
    using Scalar = typename Traits::Scalar;
    using Shape = typename Traits::Shape;

  protected:
    using Base::get_derived;
    using Base::decode_index;
    
  public:

    constexpr Scalar sequence(size_t n) {return sequence_ref(n);}
    constexpr const Scalar sequence(size_t n) const {return sequence_ref(n);}

    constexpr Scalar& sequence_ref(size_t n) {return get_derived()->sequence_ref(n);}
    constexpr const Scalar& sequence_ref(size_t n) const {return get_derived()->sequence_ref(n);}

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
      return sequence_ref(decode_index(index...));
    }
    template<typename... Index,
             std::enable_if_t<
               // Make sure the number of arguments is correct
               std::integral_constant<bool,
                                      !(sizeof...(Index)-Shape::get_dimension())
                                      >::value, int> =0
             >
    constexpr const auto& operator()(Index... index) const {
      return sequence_ref(decode_index(index...));
    }
    
  };

} // namespace tsr

#endif
