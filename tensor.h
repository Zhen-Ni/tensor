#ifndef DENSE_H
#define DENSE_H


#ifndef NDEBUG
#include <cstdlib>
#include <type_traits>
#include "shape.h"
#include "dense_base.h"
#include "map.h"
#endif


namespace tsr {

 
  template <typename T, size_t... dims>
  class Tensor: public DenseBase<Tensor<T, dims...>> {
    friend class TensorBase<Tensor>;
    friend class DenseBase<Tensor>;

    using Parent = DenseBase<Tensor>;

  public:
    using typename Parent::Scalar;
    using typename Parent::Shape;
    
  private:
    Scalar data[Shape::size];
    
  public:
    constexpr Scalar* get_data() {return data;}
    constexpr const Scalar* get_data() const {return data;}

    constexpr Scalar& sequence_ref(size_t n) {return data[n];}
    constexpr const Scalar& sequence_ref(size_t n) const {return data[n];}

  public:
    constexpr Tensor() {};
    template <typename... E>
    constexpr Tensor(const E&... e): data{e...} {}

    using Parent::operator=;    
    
    constexpr auto operator[](size_t n)
    {return typename internal::map_slice<Scalar, dims...>::type
        (&data[Shape::get_stride() * n]);}

    constexpr auto operator[](size_t n) const
    {return typename internal::map_slice<const Scalar, dims...>::type
        (&data[Shape::get_stride() * n]);}
  };


    

  template <typename T, size_t... dims>
  struct BaseTraits<Tensor<T, dims...>> {
    using Shape = internal::ShapeType<dims...>;
    using Scalar = T;
  };  
  
} // namespace tsr

#endif
