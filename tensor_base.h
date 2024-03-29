#ifndef TENSORBASE_H
#define TENSORBASE_H

#ifndef NDEBUG
#include <cstdlib>
#include <type_traits>
#include "shape.h"
#include "unroll.h"
#endif


namespace tsr {

  template <typename Derived>
  struct BaseTraits;

  template <typename T, size_t... dims>
  class Tensor;

  template <typename T>
  class TensorBase;
  

  namespace internal{

    template <typename T, typename ShapeType>
    struct get_tensor;

    template <typename T, size_t... dims>
    struct get_tensor<T, ShapeType<dims...>> {
      using type = Tensor<T, dims...>;
    };
    
    template <typename T, typename ShapeType>
    using get_tensor_t = typename get_tensor<T, ShapeType>::type;

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
    // constexpr auto sequence(size_t n) {return get_derived()->sequence(n);}
    constexpr auto sequence(size_t n) const {return get_derived()->sequence(n);}
    
    template<typename... Index>
    constexpr auto operator()(Index... index) {
      return sequence(Shape::decode_index(index...));
    }
    template<typename... Index>
    constexpr auto operator()(Index... index) const {
      return sequence(Shape::decode_index(index...));
    }

    constexpr auto eval() const {
      typename internal::get_tensor_t<Scalar, Shape> res;
      res = *this;
      return res;
    }
    
    
    // Additional class methods
    
    constexpr bool any() const {
      for (size_t i = 0; i != Shape::size; ++i) {
        if (sequence(i)) { return true; }
      }
      return false;
    }

    constexpr bool all() const {
      for (size_t i = 0; i != Shape::size; ++i) {
        if (!sequence(i)) { return false; }
      }
      return true;
    }
    
  };


} // namespace tsr

#endif
