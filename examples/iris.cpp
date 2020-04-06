#include "NNKek.h"

using namespace NNKek;

struct Sample {
  Linalg::Vector<double, 4> input;
  Linalg::Vector<double, 3> output;
};

Container::Vector<Sample> samples;

template <typename L1, typename L2>
Linalg::Vector<double, 3> getResult(L1 &layer1, L2 &layer2,
                                    Linalg::Vector<double, 4> input) {
  auto result = layer1.forward(input);
  auto result1 = Activation::tanh(result);
  auto result2 = layer2.forward(result1);
  auto result3 = Activation::tanh(result2);
  return result3;
}

template <typename L1, typename L2> double fitness(L1 &layer1, L2 &layer2) {
  double error = 0;
  size_t x = 0;

  auto numTrain = samples.size() / 2;

  for (size_t i = 0; i < numTrain; i++) {
    auto target = samples[i].output;
    auto result = getResult(layer1, layer2, samples[i].input);
    error += (target - result).magSq();
    x++;
  }

  return error / x;
}

int main(void) {
  Fs::readLines("data/iris.data", [](Container::String line) {
    if (line.length() == 0)
      return;

    Sample s = {};
    char cat[64];
    sscanf(line.c_str(), "%lf,%lf,%lf,%lf,%s", &s.input[0], &s.input[1],
           &s.input[2], &s.input[3], cat);

    if (strcmp(cat, "Iris-setosa") == 0) {
      s.output[0] = 1;
    } else if (strcmp(cat, "Iris-versicolor") == 0) {
      s.output[1] = 1;
    } else {
      s.output[2] = 1;
    }

    samples.push(s);
  });

  Util::shuffle(samples, samples.size());

  Layer::Dense<double, 4, 50> layer1;
  Layer::Dense<double, 50, 3> layer2;

  double score = fitness(layer1, layer2);

  for (size_t i = 0; i < 100000; i++) {
    printf("Iteration %ld, the error is %f                      \r", i, score);
    auto l1 = layer1;
    auto l2 = layer2;

    Mutation::normalMutate(&l1.m_matrix, 0.0001);
    Mutation::normalMutate(&l2.m_matrix, 0.0001);

    double newScore = fitness(l1, l2);

    if (newScore <= score) {
      score = newScore;
      layer1 = l1;
      layer2 = l2;
    }
  }

  auto numTrain = samples.size() / 2;

  size_t correct = 0;
  size_t incorrect = 0;

  for (size_t i = numTrain; i < samples.size(); i++) {
    auto target = samples[i].output;
    auto result = getResult(layer1, layer2, samples[i].input);

    if (target.argmax() == result.argmax()) {
      correct++;
    } else {
      incorrect++;
    }
  }

  double total = correct + incorrect;
  printf("Correct %f, incorrect %f total %f\n", correct / total,
         incorrect / total, total);

  return 0;
}
