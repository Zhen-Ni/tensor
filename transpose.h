#ifndef TRANSPOSE_H
#define TRANSPOSE_H

#include <type_traits>
#include <utility>
#ifndef NDEBUG
#include "shape.h"
#include "tensor.h"
#endif


namespace tsr {

  namespace internal {

    template <class F, class Tuple, std::size_t... I>
    constexpr auto reversed_apply_impl(F&& f, Tuple&& t, std::index_sequence<I...>) {
      constexpr size_t N = std::tuple_size<std::remove_reference_t<Tuple>>::value;
      return std::forward<F>(f)(std::get<N-1-I>(std::forward<Tuple>(t))...);
    }
    
    template<typename F, typename Tuple>
    constexpr auto reversed_apply(F&& f, Tuple&& t) {
      return reversed_apply_impl
        (std::forward<F>(f), std::forward<Tuple>(t),
         std::make_index_sequence<std::tuple_size<std::remove_reference_t<Tuple>>::value>{});
    }
    
  }    // end of namespace internal

  
  template<typename T>
  class Transpose: public TensorBase<Transpose<T>> {

    using Base = TensorBase<Transpose<T>>;
    using typename Base::Traits;
    
  public:
    using Shape = typename Traits::Shape;
    using Scalar = typename Traits::Scalar;
    
  private:
    const TensorBase<T> & op;

  public:
    constexpr Transpose(const TensorBase<T>& op): op(op) {}
    
    constexpr auto sequence(size_t n) const {
      auto indexes = Shape::encode_index(n);
      n = internal::reversed_apply(TensorBase<T>::Shape::decode_index,
                                   indexes);
      return op.sequence(n);
    }
  };
  
  template<typename T>
  struct BaseTraits<Transpose<T>> {
    // Type check here.
    // It also give hints to an annoying problem, with code below:
    // const Tensor<int> a;
    // Transpose<decltype(a)> b(a);
    // where [T = const Tensor<int>], thus TensorBase<T> is a
    // undefined type. We can explicitly check the types here
    // to explicitly hint the problem.
    static_assert(std::is_base_of<TensorBase<T>, T>::value, "Transpose template should be subclass of TensorBase (use template function transpose instead)");
    using Shape = shape_rev_t<typename T::Shape>;
    using Scalar = typename T::Scalar;
  };


  template<typename T>
  constexpr auto transpose(const TensorBase<T>& op) {
    return Transpose<T>(op);
  }

  
}    // end of namespace tsr


#endif    // end of TRANSPOSE_H
