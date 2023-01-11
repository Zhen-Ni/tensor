#include <iostream>
using namespace std;

#include "tensor"

using namespace tsr;


int test_tensor() {
  cout << "test tensor" << endl;
  
  constexpr Tensor<double, 2, 3> t1 = {1.1, 2.2, 3.3, 4.4, 5.5, 6.6};
  cout << t1 << endl;

  Tensor<int, 2, 3> t2;
  t2 = t1;
  auto t3 = t2;
  t3(0, 0) = 1000;
  cout << t2 << endl;
  cout << t3 << endl;

  t3 = t1;
  cout << t3 << endl;
  
  cout << "test tensor complete" << endl;

  return 0;
}


int test_map() {
  cout << "test map" << endl;
  
  constexpr Tensor<double, 2, 3, 2> t1 = {1.1, 2.2, 3.3, 4.4, 5.5, 6.6,
                                          7.7, 8.8, 9.9, 10., 11., 12.};
  cout << t1 << endl;
  
  auto m1 = t1[0];
  cout << m1 << endl;

  auto m2 = m1[1];
  cout << m2 << endl;

  Map<const double, 8> m3(t1.get_data());
  cout << m3 << endl;

  int a1[] = {1, 2, 3, 4};

  Map<int, 2, 2> m4(a1);
  // Map<int, 2, 2> m5;    // error, map do not have default constructor
  Map<int, 2, 2> m5(m4);
  m5(0, 2) = 1000;              // equal to m5(1, 1) = 1000;
  m5[0] = m4[1];
  cout << m4 << endl;
  cout << m5 << endl;
  
  cout << "test map complete" << endl;

  return 0;
}

int test_constant() {
  cout << "test constant" << endl;

  Constant<double, 2, 3> c1(1.2);
  cout << c1 << endl;

  auto c2 = c1;
  cout << c2 << endl;
  
  cout << "test constant complete" << endl;
  return 0;
  }


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

  test_tensor();
  test_map();
  test_constant();
  
  return 0;
}
