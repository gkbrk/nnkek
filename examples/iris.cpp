#include "NNKek.h"

using namespace NNKek;

struct Sample {
  Linalg::Vector<double, 4> input;
  Linalg::Vector<double, 3> output;
};

Container::Vector<Sample> samples;
size_t numTrain;

template <typename L1, typename L2>
Linalg::Vector<double, 3> getResult(L1 &layer1, L2 &layer2,
                                    Linalg::Vector<double, 4> input) {
  auto result = layer1.forward(input);
  auto result1 = Activation::relu(result);
  auto result2 = layer2.forward(result1);
  auto result3 = Activation::softmax(result2);
  return result3;
}

template <typename L1, typename L2> double fitness(L1 &layer1, L2 &layer2) {
  double error = 0;
  size_t x = 0;

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

  Layer::Dense<double, 4, 2> layer1;
  Layer::Dense<double, 2, 3> layer2;

  numTrain = samples.size() * 0.8;
  double score = fitness(layer1, layer2);

  for (size_t i = 0; score > 0.1; i++) {
    if (i % 10 == 0) {
      printf("Iteration %ld, the error is %f                      \r", i,
             score);
      fflush(stdout);
    }

    auto cost = [&layer1, &layer2]() { return fitness(layer1, layer2); };

    Mutation::costMutate(&layer1.m_matrix, cost, 0.001);
    Mutation::costMutate(&layer2.m_matrix, cost, 0.001);
    score = fitness(layer1, layer2);
  }

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
