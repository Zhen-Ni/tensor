#include <assert.h>
#include <iostream>
using namespace std;

#include "tensor"

using namespace tsr;


int test_tensor() {
   cout << "test tensor" << endl;
  
  Tensor<double, 2, 3> t10 = {1.1, 2.2, 3.3, 4.4, 5.5, 6.6};
  constexpr Tensor<double, 2, 3> t11 (1.1, 2.2, 3.3, 4.4, 5.5, 6.6);
  assert((t10 == t11).all());
  auto t1 = t10;
  assert(t1(1, 1) == 5.5);

  Tensor<int, 2, 3> t2;
  t2 = t1;
  auto t3 = t2;
  t3(0, 0) = 1000;
  assert(t2(0, 0) == 1);
  assert(t3(0, 0) == 1000);

  t3 = t1;
  assert(t3(0, 0) == 1);
  
  cout << "test tensor complete" << endl;

  return 0;
}


int test_map() {
  cout << "test map" << endl;
  
  Tensor<double, 2, 3, 2> t1 = {1.1, 2.2, 3.3, 4.4, 5.5, 6.6,
    7.7, 8.8, 9.9, 10., 11., 12.};
  
  auto m1 = t1[0];
  auto m2 = m1[1];

  Map<const double, 8> m3(t1.get_data());
  assert((m2[0] == m3[2]).all());
  
  int a1[] = {1, 2, 3, 4};

  Map<int, 2, 2> m4(a1);
  // Map<int, 2, 2> m5;    // error, map do not have default constructor
  Map<int, 2, 2> m5(m4);
  m5(0, 2) = 1000;              // equal to m5(1, 1) = 1000;
  m5[0] = m4[1];
  assert((m4 == m5).all());
  assert(m4(0,0)==1000);
  
  cout << "test map complete" << endl;

  return 0;
}


int test_io() {
    cout << "test io" << endl;
    Tensor<double, 2, 3, 2> t1 = {1.1, 2.2, 3.3, 4.4, 5.5, 6.6,
      7.7, 8.8, 9.9, 10., 11., 12.};
    cout << t1 << endl;
    cout << t1[0] << endl;
    cout << t1[0][0] << endl;
    cout << t1[0][0][0] << endl;
    cout << "test io complete" << endl;
    return 0;
}


int test_constant() {
  cout << "test constant" << endl;

  constexpr Constant<double, 2, 3> c1(1.5);
  constexpr auto c2 = c1;
  static_assert(c2(1,2) == 1.5, "error");
  
  cout << "test constant complete" << endl;
  return 0;
}


int test_binary_operator() {
  cout << "test binary operator" << endl;
  constexpr Tensor<double, 2, 3> a(1., 2., 3., 4., 5., 6.);
  constexpr Tensor<double, 2, 3> b(.1, .2, .3, .4, .5, .6);
  auto c = a + b;
  auto d = a - b;
  constexpr auto e = Constant<double, 2, 3>(1.5);
  // DO NOT USE auto here!
  // as (c + d) generates a temprary object, which will be
  // deconstructed immediately after the statement is executed.
  // If auto is used here, f will be an expression template holding
  // the temporary object (c + d), which leads to a dangling
  // reference when f is further used.
  Tensor<double, 2, 3> f = (c + d) * e;
  // type of `res` is `Tensor`, as `eval` is used.
  auto res = (f / Constant<double, 2, 3>(1.)).eval();
  assert((res == Tensor<double, 2, 3>::linspace(3, 3)).all());
  cout << "test binary operator complete" << endl;
  return 0;
}


int test_unary_operator() {
  cout << "test unary operator" << endl;
  constexpr Tensor<float, 1> a(1.5f);
  constexpr Tensor<float, 1> b = -a;
  static_assert((a==-b).all(), "error");
  cout << "test unary complete" << endl;
  return 0;
}


int test_dot() {
  cout << "test dot" << endl;

  // matrix dot (gemm)
  constexpr Tensor<int, 2, 4> a(1, 2, 3, 4, 5, 6, 7, 8);
  constexpr Tensor<int, 4, 3> b(1, 1, 1, 2, 4, -1, 4, 5, 5, 1, 0, 7);
  constexpr Tensor<int, 2, 3> c = dot(a, b);
  static_assert((c==Tensor<int, 2, 3>(21, 24, 42, 53, 64, 90)).all(), "error");

  // gemv
  constexpr Tensor<int, 2> c2 = dot(a, Tensor<int, 4>(1, 1, 0, 0));
  static_assert((c2==Tensor<int, 2>(3, 11)).all(), "error");

  // gevv
  static_assert((dot(Tensor<int, 3>(1,2,3),
                      Tensor<int, 3>(3,0,-1)) 
                 ==Tensor<int>(0)).all(), "error");
  
  // tensor dot
  constexpr auto d = Tensor<int, 2, 3, 4>::linspace(0, 1);
  constexpr auto e = Tensor<int, 4, 3, 2>::linspace(0, 1);
  static_assert((dot(d, e)[0][0]==Tensor<int, 3, 2>(84, 90, 96, 102, 108, 114)).all(), "error");
  static_assert((dot(d, e)[1][2]==Tensor<int, 3, 2>(804, 890, 976, 1062, 1148, 1234)).all(), "error");
  
  
  cout << "test dot complete" << endl;
  return 0;
}


int main() {

  test_tensor();
  test_map();
  test_io();
  test_constant();
  test_binary_operator();
  test_unary_operator();
  test_dot();
  
  return 0;
}
