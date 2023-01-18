#ifndef SHAPE_H
#define SHAPE_H

#ifndef NDEBUG
#include <cstdlib>
#include <type_traits>
#include <tuple>
#endif

namespace tsr {
  
    template <size_t... dims>
    struct ShapeType;


    template <size_t dim0, size_t... dims>
    struct ShapeType<dim0, dims...> {
      using ContainedType = ShapeType<dims...>;
      static constexpr size_t size0 = dim0;
      static constexpr size_t dimension = sizeof...(dims) + 1;
      static constexpr size_t stride = ShapeType<dims...>::size;
      static constexpr size_t size = dim0 * stride;
      static constexpr size_t decode_index(size_t n, decltype(dims)... rest) {
        return n * stride + ContainedType::decode_index(rest...);
      }
        static constexpr auto encode_index(size_t n) {
        return std::tuple_cat(std::make_tuple(n / ContainedType::size),
        ContainedType::encode_index(n %
        ContainedType::size));
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
      static constexpr std::tuple<> encode_index(size_t) { return {}; }
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


    // Shortcuts
    
    template<typename T, size_t n>
    struct shape_remove_first_n {
      using type = typename shape_remove_first_n<typename T::ContainedType, n - 1>::type;
    };
    template<typename T>
    struct shape_remove_first_n<T, 0> {
      static_assert(!std::is_same<T, void>::value,
                    "shape dimension less than n");
      using type = T;
    };
    
    template<typename T, size_t n>
    using shape_remove_fisrt_n_t = typename shape_remove_first_n<T, n>::type;

    template<typename T, size_t n>
    struct shape_get_last_n {
      static_assert(n <= T::dimension, "shape dimension less than n");
      using type = shape_remove_fisrt_n_t<T, T::dimension - n>;
    };

    template<typename T, size_t n>
    using shape_get_last_n_t = typename shape_get_last_n<T, n>::type;

    template<typename T, size_t n>
    struct shape_get_first_n {
      using type = shape_rev_t<shape_get_last_n_t<shape_rev_t<T>, n>>;
    };

    template<typename T, size_t n>
    using shape_get_first_n_t = typename shape_get_first_n<T, n>::type;

  
} // end of namespace tsr

#endif  // SHAPE_H
