#ifndef UNARY_OPERATOR_H
#define UNARY_OPERATOR_H

#ifndef NDEBUG
#include <type_traits>
#include "tensor_base.h"
#endif


namespace tsr{

  template <template<typename, typename> class OperatorTemplate,
            typename Operand,
            typename Result = typename Operand::Scalar
            >
  class UnaryOperator:
    public TensorBase<UnaryOperator<OperatorTemplate, Operand, Result>> {
    const TensorBase<Operand>& operand;
    using Operator = OperatorTemplate<typename Operand::Scalar,
                                      Result>;
    
  public:
    constexpr UnaryOperator(const TensorBase<Operand>& operand):
      operand(operand) {}

    constexpr auto sequence(size_t n) const {
      return Operator::call(operand.sequence(n));
    }
  };

  
  template <template<typename, typename> class OperatorTemplate,
            typename Operand,
            typename Result>
  struct BaseTraits<UnaryOperator<OperatorTemplate, Operand, Result>> {
    using Shape = typename Operand::Shape;
    using Scalar = Result;
  };
  

  template <typename Operand, typename Res>
  struct NegOperator{
    static constexpr Res call(const Operand& operand) {return -operand;}
  };


  template <typename Operand>
  inline constexpr auto operator-(const TensorBase<Operand>& operand) {
    return UnaryOperator<NegOperator, Operand>(operand);
  }
  
} // namespace tsr

#endif  // #ifndef UNARY_OPERATOR_H
