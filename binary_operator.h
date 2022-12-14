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
            typename Result=decltype(typename Lhs::Scalar() +
                                     typename Rhs::Scalar()),
            std::enable_if_t<
              std::is_same<typename Lhs::Shape,
                           typename Rhs::Shape>::value, int> = 0
            >
  class BinaryOperator:
    public TensorBase<BinaryOperator<OperatorTemplate, Lhs, Rhs, Result>> {
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
    return BinaryOperator<AddOperator, Lhs, Rhs>(lhs, rhs);
  }

  template <typename Lhs, typename Rhs, typename Res>
  struct SubOperator{
    static constexpr Res call(const Lhs& lhs, const Rhs& rhs) {return lhs - rhs;}
  };

  template <typename Lhs, typename Rhs>
  inline constexpr auto operator-(const TensorBase<Lhs>& lhs,
                                  const TensorBase<Rhs>& rhs) {
    return BinaryOperator<SubOperator, Lhs, Rhs>(lhs, rhs);
  }

  template <typename Lhs, typename Rhs, typename Res>
  struct MulOperator{
    static constexpr Res call(const Lhs& lhs, const Rhs& rhs) {return lhs * rhs;}
  };

  template <typename Lhs, typename Rhs>
  inline constexpr auto operator*(const TensorBase<Lhs>& lhs,
                                  const TensorBase<Rhs>& rhs) {
    return BinaryOperator<MulOperator, Lhs, Rhs>(lhs, rhs);
  }
  
  template <typename Lhs, typename Rhs, typename Res>
  struct DivOperator{
    static constexpr Res call(const Lhs& lhs, const Rhs& rhs) {return lhs / rhs;}
  };

  template <typename Lhs, typename Rhs>
  inline constexpr auto operator/(const TensorBase<Lhs>& lhs,
                                  const TensorBase<Rhs>& rhs) {
    return BinaryOperator<DivOperator, Lhs, Rhs>(lhs, rhs);
  }

  template <typename Lhs, typename Rhs, typename Res>
  struct DivModOperator{
    static constexpr Res call(const Lhs& lhs, const Rhs& rhs) {return lhs % rhs;}
  };

  template <typename Lhs, typename Rhs>
  inline constexpr auto operator%(const TensorBase<Lhs>& lhs,
                                  const TensorBase<Rhs>& rhs) {
    return BinaryOperator<DivModOperator, Lhs, Rhs>(lhs, rhs);
  }
  
} // namespace tsr

#endif  // #ifndef BINARY_OPERATOR_H
