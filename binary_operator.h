#ifndef BINARY_OPERATOR_H
#define BINARY_OPERATOR_H

#ifndef NDEBUG
#include <type_traits>
#include "tensor_base.h"
#endif


namespace tsr{

  template <template<typename, typename, typename> class OperatorTemplate,
            typename Lhs,
            typename Rhs,
            typename Result>
  class BinaryOperator:
    public TensorBase<BinaryOperator<OperatorTemplate, Lhs, Rhs, Result>> {
    // Make sure the dimensions are correct.
    static_assert(std::is_same<typename Lhs::Shape,
                  typename Rhs::Shape>::value,
                  "dimensions of lhs and rhs should be the same");
    
    const Lhs& lhs;
    const Rhs& rhs;
    using Operator = OperatorTemplate<typename Lhs::Scalar,
                                      typename Rhs::Scalar,
                                      Result>;
    
  public:
    constexpr BinaryOperator(const TensorBase<Lhs>& lhs,
                             const TensorBase<Rhs>& rhs):
      lhs(*static_cast<const Lhs*>(&lhs)), rhs(*static_cast<const Rhs*>(&rhs)) {}

    constexpr auto sequence(size_t n) const {
      return Operator::call(lhs.sequence(n), rhs.sequence(n));
    }
    
  };

  
  template <template<typename, typename, typename> class OperatorTemplate,
            typename Lhs,
            typename Rhs,
            typename Result>
  struct BaseTraits<BinaryOperator<OperatorTemplate, Lhs, Rhs, Result>> {
    using Shape = typename Lhs::Shape;
    using Scalar = Result;
  };
  

  template <typename Lhs, typename Rhs, typename Res>
  struct AddOperator{
    static constexpr Res call(const Lhs& lhs, const Rhs& rhs) {return lhs + rhs;}
  };

  template <typename Lhs, typename Rhs>
  inline constexpr auto operator+(const TensorBase<Lhs>& lhs,
                                  const TensorBase<Rhs>& rhs) {
    return BinaryOperator<AddOperator, Lhs, Rhs,
                          decltype(typename Lhs::Scalar() +
                                   typename Rhs::Scalar())>(lhs, rhs);
  }

  template <typename Lhs, typename Rhs, typename Res>
  struct SubOperator{
    static constexpr Res call(const Lhs& lhs, const Rhs& rhs) {return lhs - rhs;}
  };

  template <typename Lhs, typename Rhs>
  inline constexpr auto operator-(const TensorBase<Lhs>& lhs,
                                  const TensorBase<Rhs>& rhs) {
    return BinaryOperator<SubOperator, Lhs, Rhs,
                          decltype(typename Lhs::Scalar() -
                                   typename Rhs::Scalar())>(lhs, rhs);
  }

  template <typename Lhs, typename Rhs, typename Res>
  struct MulOperator{
    static constexpr Res call(const Lhs& lhs, const Rhs& rhs) {return lhs * rhs;}
  };

  template <typename Lhs, typename Rhs>
  inline constexpr auto operator*(const TensorBase<Lhs>& lhs,
                                  const TensorBase<Rhs>& rhs) {
    return BinaryOperator<MulOperator, Lhs, Rhs,
                          decltype(typename Lhs::Scalar() *
                                   typename Rhs::Scalar())>(lhs, rhs);
  }
  
  template <typename Lhs, typename Rhs, typename Res>
  struct DivOperator{
    static constexpr Res call(const Lhs& lhs, const Rhs& rhs) {return lhs / rhs;}
  };

  template <typename Lhs, typename Rhs>
  inline constexpr auto operator/(const TensorBase<Lhs>& lhs,
                                  const TensorBase<Rhs>& rhs) {
    return BinaryOperator<DivOperator, Lhs, Rhs,
                          decltype(typename Lhs::Scalar() /
                                   typename Rhs::Scalar())>(lhs, rhs);
  }

  template <typename Lhs, typename Rhs, typename Res>
  struct DivModOperator{
    static constexpr Res call(const Lhs& lhs, const Rhs& rhs) {return lhs % rhs;}
  };

  template <typename Lhs, typename Rhs>
  inline constexpr auto operator%(const TensorBase<Lhs>& lhs,
                                  const TensorBase<Rhs>& rhs) {
    return BinaryOperator<DivModOperator, Lhs, Rhs,
                          decltype(typename Lhs::Scalar() %
                                   typename Rhs::Scalar())>(lhs, rhs);
  }

  template <typename Lhs, typename Rhs, typename Res>
  struct EqOperator{
    static constexpr Res call(const Lhs& lhs, const Rhs& rhs) {return lhs == rhs;}
  };

  template <typename Lhs, typename Rhs>
  inline constexpr auto operator==(const TensorBase<Lhs>& lhs,
                                   const TensorBase<Rhs>& rhs) {
    return BinaryOperator<EqOperator, Lhs, Rhs, bool>(lhs, rhs);
  }

  template <typename Lhs, typename Rhs, typename Res>
  struct NeOperator{
    static constexpr Res call(const Lhs& lhs, const Rhs& rhs) {return lhs != rhs;}
  };

  template <typename Lhs, typename Rhs>
  inline constexpr auto operator!=(const TensorBase<Lhs>& lhs,
                                   const TensorBase<Rhs>& rhs) {
    return BinaryOperator<NeOperator, Lhs, Rhs, bool>(lhs, rhs);
  }
  
  template <typename Lhs, typename Rhs, typename Res>
  struct LtOperator{
    static constexpr Res call(const Lhs& lhs, const Rhs& rhs) {return lhs < rhs;}
  };

  template <typename Lhs, typename Rhs>
  inline constexpr auto operator<(const TensorBase<Lhs>& lhs,
                                  const TensorBase<Rhs>& rhs) {
    return BinaryOperator<LtOperator, Lhs, Rhs, bool>(lhs, rhs);
  }

  template <typename Lhs, typename Rhs, typename Res>
  struct GtOperator{
    static constexpr Res call(const Lhs& lhs, const Rhs& rhs) {return lhs > rhs;}
  };

  template <typename Lhs, typename Rhs>
  inline constexpr auto operator>(const TensorBase<Lhs>& lhs,
                                  const TensorBase<Rhs>& rhs) {
    return BinaryOperator<GtOperator, Lhs, Rhs, bool>(lhs, rhs);
  }

  template <typename Lhs, typename Rhs, typename Res>
  struct LeOperator{
    static constexpr Res call(const Lhs& lhs, const Rhs& rhs) {return lhs <= rhs;}
  };

  template <typename Lhs, typename Rhs>
  inline constexpr auto operator<=(const TensorBase<Lhs>& lhs,
                                   const TensorBase<Rhs>& rhs) {
    return BinaryOperator<LeOperator, Lhs, Rhs, bool>(lhs, rhs);
  }

  template <typename Lhs, typename Rhs, typename Res>
  struct GeOperator{
    static constexpr Res call(const Lhs& lhs, const Rhs& rhs) {return lhs >= rhs;}
  };

  template <typename Lhs, typename Rhs>
  inline constexpr auto operator>=(const TensorBase<Lhs>& lhs,
                                   const TensorBase<Rhs>& rhs) {
    return BinaryOperator<GeOperator, Lhs, Rhs, bool>(lhs, rhs);
  }
  
} // namespace tsr

#endif  // #ifndef BINARY_OPERATOR_H
