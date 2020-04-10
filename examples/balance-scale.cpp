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
  Fs::readLines("data/balance-scale.data", [](Container::String line) {
    if (line.length() == 0)
      return;

    Sample s = {};

    {
      auto parts = line.split_at(',');

      if (strcmp(parts.first().c_str(), "B") == 0) {
        s.output[0] = 1;
      } else if (strcmp(parts.first().c_str(), "L") == 0) {
        s.output[1] = 1;
      } else {
        s.output[2] = 1;
      }

      line = parts.second();
    }

    for (size_t i = 0; i < 4; i++) {
      auto parts = line.split_at(',');

      s.input[i] = atof(parts.first().c_str()) / 10.0;
      line = parts.second();
    }

    samples.push(s);
  });

  Util::shuffle(samples, samples.size());

  numTrain = samples.size() * 0.5;

  Layer::Dense<double, 4, 5> layer1;
  Layer::Dense<double, 5, 3> layer2;

  double score = fitness(layer1, layer2);

  for (size_t i = 0; i < 100; i++) {
    if (i % 1000 == 0) {
      printf("Iteration %ld, the error is %f                      \r", i,
             score);
      fflush(stdout);
    }
    auto l1 = layer1;
    auto l2 = layer2;

    Mutation::normalMutate(&l1.m_matrix, 0.000001);
    Mutation::normalMutate(&l2.m_matrix, 0.000001);

    double newScore = fitness(l1, l2);

    if (newScore <= score) {
      score = newScore;
      layer1 = l1;
      layer2 = l2;
    }
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
