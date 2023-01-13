#ifndef SHAPE_H
#define SHAPE_H

#ifndef NDEBUG
#include <cstdlib>
#include <type_traits>
#include <tuple>
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
      template<typename... Rest,
               std::enable_if_t<sizeof...(Rest) == sizeof...(dims), int> = 0>
      static constexpr size_t decode_index(size_t n, Rest... rest) {
        return n * stride + ContainedType::decode_index(rest...);
      }
    };
  
  
    template <>
    struct ShapeType<> {
      using ContainedType = void;
      static constexpr size_t size0 = 1;   // undefined
      static constexpr size_t dimension = 0;
      static constexpr size_t stride = 1; // undefined: can be any value
      static constexpr size_t size = 1;
      static constexpr size_t decode_index() { return 0; }
    };

    
    // Concatenate the dimensions of two ShapeTypes.
    template<typename, typename> struct shape_cat;

    template<typename T1, typename T2>
    using shape_cat_t = typename shape_cat<T1, T2>::type;
    
    template<size_t... dim1, size_t... dim2>
    struct shape_cat<ShapeType<dim1...>, ShapeType<dim2...>> {
      using type = ShapeType<dim1..., dim2...>;
    };


    // Reverse the dimensions of a ShapeType.
    template<typename T> struct shape_rev;
    
    template<typename T>
    using shape_rev_t = typename shape_rev<T>::type;
    
    template<size_t dim0, size_t... dims>
    struct shape_rev<ShapeType<dim0, dims...>> {
      using type = shape_cat_t<shape_rev_t<ShapeType<dims...>>,
                               ShapeType<dim0>>;
    };
    
    template<>
    struct shape_rev<ShapeType<>> {
      using type = ShapeType<>;
    };

    
    // template<size_t dimn, size_t... dims>
    // struct shape_rev<ShapeType<dims..., dimn>> {
      // using type = int;
    // };


  } // end of namespace internal
  
} // end of namespace tsr

#endif  // SHAPE_H
