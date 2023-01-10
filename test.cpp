#include <iostream>
using namespace std;

#include "tensor.h"

using namespace tsr;


int main() {
  constexpr Tensor<double, 2, 3> t1 = {1.1, 2.2, 3.3, 4.4, 5.5, 6.6};
  cout << t1(1,1) << endl;

  Tensor<int, 2, 3> t2;
  t2 = t1;
  
  cout << t2(1,1) << endl;

  Tensor<float, 2, 3> t3;
  t3 = t1 + t2 * t1 / t2;
  cout << t3(1,1) << endl;

  Constant<double, 2, 3> t4(1.2);
  t4 = 2.2;
  t3 = t4 + t3;
  cout << t3(1,1) << endl;
  
  return 0;
}
