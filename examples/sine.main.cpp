#include "NNKek.h"
#include <cmath>
#include <cstdint>
#include <stdio.h>

using namespace NNKek;
using namespace NNKek::Linalg;

template <typename L1, typename L2>
Vector<float, 1> getResult(L1 layer1, L2 layer2, float input) {
  Vector<float, 1> v;
  v[0] = input / 10.0;

  auto result = layer1.forward(v);
  auto result1 = Activation::tanh(result);
  auto result2 = layer2.forward(result1);
  auto result3 = Activation::tanh(result2);
  return result3;
}

template <typename L1, typename L2> double fitness(L1 layer1, L2 layer2) {
  double error = 0;
  size_t x = 0;

  for (float i = -5; i < 5; i += 0.05) {
    auto target = std::sin(i);
    auto result = getResult(layer1, layer2, i)[0];
    error += (target - result) * (target - result);
    x++;
  }

  return error / x;
}

int main(void) {
  Layer::Dense<float, 1, 20> layer1;
  Layer::Dense<float, 20, 1> layer2;

  double score = fitness(layer1, layer2);

  for (size_t i = 0; i < 50000; i++) {
    // printf("Iteration %ld, the error is %f\n", i, score);
    auto l1 = layer1;
    auto l2 = layer2;

    Mutation::testMutate(&l1.m_matrix, 0.05);
    Mutation::testMutate(&l2.m_matrix, 0.05);

    double newScore = fitness(l1, l2);

    if (newScore <= score) {
      score = newScore;
      layer1 = l1;
      layer2 = l2;
    }
  }

  for (float i = -5; i < 5; i += 0.001) {
    auto result = getResult(layer1, layer2, i);
    printf("%f\n", result[0]);
  }
  return 0;
}
