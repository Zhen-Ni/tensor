#ifndef TENSORBASE_H
#define TENSORBASE_H

#ifndef NDEBUG
#include <cstdlib>
#include <type_traits>
#endif


namespace tsr {

  template <size_t... dims>
  struct ShapeType;

  
  template <typename Derived>
  struct TensorTraits;

  
  template <size_t dim0, size_t... dims>
  struct ShapeType<dim0, dims...> {
    using ContainedType = ShapeType<dims...>;
    static constexpr size_t dimension = sizeof...(dims) + 1;
    static constexpr size_t stride = ShapeType<dims...>::size;
    static constexpr size_t size = dim0 * stride;
    static constexpr size_t get_dimension() {return dimension;}
    static constexpr size_t get_stride() {return stride;}
    static constexpr size_t get_size() {return size;}
  };

  
  template <>
  struct ShapeType<> {
    using ContainedType = void;
    static constexpr size_t dimension = 0;
    static constexpr size_t stride = 0; // undefined: can be any value
    static constexpr size_t size = 1;
    static constexpr size_t get_dimension() {return dimension;}
    static constexpr size_t get_stride() {return stride;}
    static constexpr size_t get_size() {return size;}
  };


  template <typename Derived>
  class TensorBase {
  public:
    using Traits = TensorTraits<Derived>;
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
    
  protected:
    template<typename Shape=Shape, typename... Rest>
    constexpr size_t decode_index(size_t n, Rest... rest) const {
      return n * Shape::stride +
        decode_index<typename Shape::ContainedType>(rest...);
    }
    template<typename Shape>
    constexpr size_t decode_index() const {return 0;}
    
  public:
    constexpr auto sequence(size_t n) {return get_derived()->sequence(n);}
    constexpr auto sequence(size_t n) const {return get_derived()->sequence(n);}

    
  };
    
} // namespace tsr

#endif
