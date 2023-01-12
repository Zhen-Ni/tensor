#ifndef TENSORBASE_H
#define TENSORBASE_H

#ifndef NDEBUG
#include <cstdlib>
#include <type_traits>
#include "shape.h"
#endif


namespace tsr {

  template <typename Derived>
  struct BaseTraits;


  template <typename T, size_t... dims>
  class Tensor;
  

  namespace internal{
    template <typename T, typename ShapeType, size_t... dims>
    struct get_tensor {
      using type = typename get_tensor<T, typename ShapeType::ContainedType, dims..., ShapeType::size0>::type;
    };

    template <typename T, size_t... dims>
    struct get_tensor<T, internal::ShapeType<>, dims...> {
      using type = Tensor<T, dims...>;
    };
      
  } // end of namespace internal
  
  
  template <typename Derived>
  class TensorBase {
  public:
    using Traits = BaseTraits<Derived>;
    using Shape = typename Traits::Shape;
    using Scalar = typename Traits::Scalar;

  private:
    // TensorBase& operator=(const TensorBase&) = delete;
    // TensorBase& operator=(TensorBase&&) = delete;
    
  protected:
    constexpr Derived* get_derived() {return static_cast<Derived*>(this);}
    constexpr const Derived* get_derived() const {
      return static_cast<const Derived*>(this);
    }
        
  public:
    constexpr auto sequence(size_t n) {return get_derived()->sequence(n);}
    constexpr auto sequence(size_t n) const {return get_derived()->sequence(n);}
    
    template<typename... Index,
             // Make sure the number of arguments is correct
             std::enable_if_t<std::integral_constant<bool, !(sizeof...(Index)-Shape::get_dimension())>::value, int> =0>
    constexpr auto operator()(Index... index) {
      return sequence(internal::decode_index<Shape>(index...));
    }
    template<typename... Index,
               // Make sure the number of arguments is correct
             std::enable_if_t<std::integral_constant<bool, !(sizeof...(Index)-Shape::get_dimension())>::value, int> =0>
    constexpr auto operator()(Index... index) const {
      return sequence(internal::decode_index<Shape>(index...));
    }

    constexpr auto eval() const {
      typename internal::get_tensor<Scalar, Shape>::type res;
      res = *this;
      return res;
    }

    
  };
    
} // namespace tsr

#endif
