#ifndef IO_H
#define IO_H

#ifndef NDEBUG
#include <sstream>
#include "unroll.h"
#include "tensor_base.h"
#include "dense_base.h"
#endif

namespace tsr {
  template<typename T>
  std::ostream& operator<<(std::ostream& os, const TensorBase<T>& tensor) {
    return os << tensor.eval(); }
  
  template<typename T,
           std::enable_if_t<T::Shape::dimension >= 2, int> = 0>
  std::ostream& operator<<(std::ostream& os, const DenseBase<T>& tensor)
  {
    os << "[";
    Unroll<0, T::Shape::get_size0()>::
      map([&](size_t index) {os << tensor[index] << ",\n ";});
    os << "]";
    return os;
  }

  template<typename T,
           std::enable_if_t<T::Shape::dimension == 1, int> = 0>
  std::ostream& operator<<(std::ostream& os, const DenseBase<T>& tensor)
  {
    os << "[";
    Unroll<0, T::Shape::get_size0()>::
      map([&](size_t index) {os << tensor[index] << ", ";});
    os << "]";
    return os;
  }

  template<typename T,
           std::enable_if_t<T::Shape::dimension == 0, int> = 0>
  std::ostream& operator<<(std::ostream& os, const DenseBase<T>& tensor)
  {
    return os << tensor();
  }

} // end of namespace tsr

#endif  // IO_H
