#ifndef MAP_H
#define MAP_H

#ifndef NDEBUG
#include <cstdlib>
#include <type_traits>
#include "dense_base.h"
#endif

namespace tsr {

  template <typename T, size_t... dims>
  class Map;

  namespace internal{
    template <typename T, size_t... dims>
    struct map_slice {
      static_assert(sizeof...(dims), "can not get slice for tensor with dimension = 0");
      using type = void;
    };

    template <typename T, size_t dim0, size_t... rest_dim>
    struct map_slice<T, dim0, rest_dim...> {
      using type = Map<T, rest_dim...>;
    };

    
  } // end of namespace internal

  
  template <typename T, size_t... dims>
  class Map: public DenseBase<Map<T, dims...>> {
    
    using Parent = DenseBase<Map>;

  public:
    using typename Parent::Scalar;
    using typename Parent::Shape;

  private:
    Scalar* data;

  public:
    constexpr Scalar* get_data() {return data;}
    constexpr const Scalar* get_data() const {return data;}

    constexpr Scalar& sequence_ref(size_t n) {return data[n];}
    constexpr const Scalar& sequence_ref(size_t n) const {return data[n];}

  public:
    // Do not add const here, as cv-qualifiers
    // are duduced from the template.
    constexpr Map(Scalar* ptr): data(ptr) {}

    // Setting the elements in the tensor
    using Parent::operator=;

    constexpr auto operator[](size_t n)
    {return typename internal::map_slice<Scalar, dims...>::type
        (&data[Shape::stride * n]);}

    constexpr auto operator[](size_t n) const
    {return typename internal::map_slice<const Scalar, dims...>::type
        (&data[Shape::stride * n]);}
  };


  template <typename T, size_t... dims>
  struct BaseTraits<Map<T, dims...>> {
    using Shape = ShapeType<dims...>;
    using Scalar = T;
  };
  
}

#endif  // MAP_H
