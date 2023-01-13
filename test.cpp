#include <iostream>
using namespace std;

#include "tensor"

using namespace tsr;


int test_tensor() {
   cout << "test tensor" << endl;
  
  Tensor<double, 2, 3> t10 = {1.1, 2.2, 3.3, 4.4, 5.5, 6.6};
  constexpr Tensor<double, 2, 3> t11 (1.1, 2.2, 3.3, 4.4, 5.5, 6.6);
  auto t1 = t10;
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
  
  Tensor<double, 2, 3, 2> t1 = {1.1, 2.2, 3.3, 4.4, 5.5, 6.6,
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


int test_binary_operator() {
  cout << "test binary operator" << endl;
  Tensor<double, 2, 3> a = {1., 2., 3., 4., 5., 6.};
  Tensor<double, 2, 3> b = {.1, .2, .3, .4, .5, .6};
  auto c = a + b;
  auto d = a - b;
  auto e = Constant<double, 2, 3>(1.5);
  // DO NOT USE auto here!
  // as (c + d) generates a temprary object, which will be
  // deconstructed immediately after the statement is executed.
  // If auto is used here, f will be an expression template holding
  // the temporary object (c + d), which leads to a dangling
  // reference when f is further used.
  Tensor<double, 2, 3> f = (c + d) * e;
  cout << f / Constant<double, 2, 3>(1.) << endl;
  cout << "test binary operator complete" << endl;
  return 0;
}


int test_unary_operator() {
  cout << "test unary operator" << endl;
  Tensor<float, 1> a{1};
  Tensor<float, 1> b = -a;
  std::cout << (a(0) == -b(0)) << std::endl;
  cout << "test unary operator complete" << endl;
  return 0;
}


int main() {

  test_tensor();
  test_map();
  test_constant();
  test_binary_operator();
  test_unary_operator();
  
  return 0;
}
