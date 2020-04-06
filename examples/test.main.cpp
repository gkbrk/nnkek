#include "NNKek.h"

int main(void) {
  NNKek::Linalg::Matrix<double, 3, 2> m1;

  m1(0, 0) = 1;
  m1(1, 0) = 2;
  m1(0, 1) = 3;
  m1(1, 1) = 4;
  m1(0, 2) = 5;
  m1(1, 2) = 6;

  m1.dump();

  NNKek::Linalg::Vector<double, 3> v;

  v[0] = 1;
  v[1] = 2;
  v[2] = 3;

  v.dump();

  auto result = v * m1;

  result.dump();
  return 0;
}
