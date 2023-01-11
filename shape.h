#ifndef SHAPE_H
#define SHAPE_H

#ifndef NDEBUG
#include <cstdlib>
#endif

namespace tsr {

  namespace internal {
  
    template <size_t... dims>
    struct ShapeType;


    template <size_t dim0, size_t... dims>
    struct ShapeType<dim0, dims...> {
      using ContainedType = ShapeType<dims...>;
      static constexpr size_t size0 = dim0;
      static constexpr size_t dimension = sizeof...(dims) + 1;
      static constexpr size_t stride = ShapeType<dims...>::size;
      static constexpr size_t size = dim0 * stride;
      static constexpr size_t get_size0() {return size0;}
      static constexpr size_t get_dimension() {return dimension;}
      static constexpr size_t get_stride() {return stride;}
      static constexpr size_t get_size() {return size;}
    };
  
  
    template <>
    struct ShapeType<> {
      using ContainedType = void;
      static constexpr size_t size0 = 1;   // undefined
      static constexpr size_t dimension = 0;
      static constexpr size_t stride = 0; // undefined: can be any value
      static constexpr size_t size = 1;
      static constexpr size_t get_size0() {return size0;}
      static constexpr size_t get_dimension() {return dimension;}
      static constexpr size_t get_stride() {return stride;}
      static constexpr size_t get_size() {return size;}
    };

    template <typename Shape> constexpr size_t decode_index() {
      return 0;
    }
    template<typename Shape, typename... Rest>
    constexpr size_t decode_index(size_t n, Rest... rest) {
      return n * Shape::stride +
        decode_index<typename Shape::ContainedType>(rest...);
    }


  } // end of namespace internal




  
} // end of namespace tsr

#endif  // SHAPE_H
