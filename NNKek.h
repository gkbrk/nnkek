// NNKek: Neural network library based on evolutionary algorithms
// Copyright (C) 2020 Gokberk Yaltirakli
//
// This program is free software: you can redistribute it and/or modify
// it under the terms of the GNU Affero General Public License as published by
// the Free Software Foundation, either version 3 of the License, or
// (at your option) any later version.
//
// This program is distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
// GNU Affero General Public License for more details.
//
// You should have received a copy of the GNU Affero General Public License
// along with this program.  If not, see <https://www.gnu.org/licenses/>.

#ifndef NNKEK_H_
#define NNKEK_H_

#include <cmath>
#include <cstddef>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>

#define ASSERT(x)                                                              \
  do {                                                                         \
    if (!(x)) {                                                                \
      printf("ASSERTION FAILED\n");                                            \
      printf("%s:%s:%d\n", __func__, __FILE__, __LINE__);                      \
      exit(1);                                                                 \
    }                                                                          \
  } while (0)

namespace NNKek {

std::random_device rd;
std::mt19937 gen(rd());

namespace Util {

template <typename T> T map(T value, T f1, T t1, T f2, T t2) {
  return f2 + ((t2 - f2) * (value - f1)) / (t1 - f1);
}

template <typename T> T min(T v1, T v2) {
  if (v1 <= v2) {
    return v1;
  }

  return v2;
}

template <typename T> T max(T v1, T v2) {
  if (v1 >= v2) {
    return v1;
  }

  return v2;
}

template <typename T> void swap(T &v1, T &v2) {
  T tmp = v2;
  v2 = v1;
  v1 = tmp;
}

template <typename T> void shuffle(T &container, size_t length) {
  for (size_t i = length - 1; i > 0; i--) {
    size_t j = rand() % (i + 1);
    swap(container[i], container[j]);
  }
}

template <typename T> T random_range(T from, T to) {
  return map((T)gen(), (T)gen.min(), (T)gen.max(), from, to);
}

} // namespace Util

namespace Container {

template <typename T> class Option {
public:
  Option(T val) {
    m_has_value = true;
    m_val = val;
  }

  Option() { m_has_value = false; }

  T or_default(T val) {
    if (m_has_value) {
      return m_val;
    } else {
      return val;
    }
  }

  T value() {
    ASSERT(m_has_value);
    return m_val;
  }

  bool is_some() { return m_has_value; }

private:
  bool m_has_value;
  T m_val;
};

template <typename T, typename L> class Pair {
public:
  Pair(T left, L right) : m_left{left}, m_right{right} {}

  T first() { return m_left; }

  L second() { return m_right; }

private:
  T m_left;
  L m_right;
};

class String {
public:
  explicit String(const char *str, size_t size) {
    m_value = static_cast<char *>(malloc(size + 1));
    memcpy(m_value, str, size);
    m_value[size] = 0;
    m_size = size;
  }

  ~String() { free(m_value); }

  String(const String &other) : String(other.m_value, other.m_size) {}

  String &operator=(const String &other) {
    if (this != &other) {
      free(m_value);
      m_size = other.m_size;
      m_value = static_cast<char *>(malloc(m_size + 1));
      memcpy(m_value, other.m_value, other.m_size);
      m_value[m_size] = 0;
    }

    return *this;
  }

  char *c_str() const { return m_value; }

  char operator[](size_t index) const { return m_value[index]; }

  size_t length() const { return m_size; }

  Pair<String, String> split_at(char);

private:
  char *m_value;
  size_t m_size;
};

class StringBuilder {
public:
  explicit StringBuilder() {
    m_size = 0;
    m_buffer = NULL;
  }

  StringBuilder(const StringBuilder &other) {
    m_buffer = static_cast<char *>(malloc(other.m_size));
    memcpy(m_buffer, other.m_buffer, other.m_size);
    m_size = other.m_size;
  }

  ~StringBuilder() { free(m_buffer); }

  StringBuilder &operator=(const StringBuilder &other) {
    if (this != &other) {
      free(m_buffer);
      m_size = other.m_size;

      m_buffer = static_cast<char *>(malloc(m_size));
      memcpy(m_buffer, other.m_buffer, m_size);
    }

    return *this;
  }

  void append(char c) {
    m_buffer = static_cast<char *>(realloc(m_buffer, m_size + 1));
    m_buffer[m_size] = c;
    m_size++;
  }

  void append(const char *str) {
    for (size_t i = 0; i < strlen(str); i++) {
      append(str[i]);
    }
  }

  void append(String str) {
    for (size_t i = 0; i < str.length(); i++) {
      append(str[i]);
    }
  }

  void clear() { m_size = 0; }

  String to_string() const { return String(m_buffer, m_size); }

  size_t length() const { return m_size; }

private:
  char *m_buffer;
  size_t m_size;
};

Pair<String, String> String::split_at(char sep) {
  StringBuilder s1;
  StringBuilder s2;

  bool t1 = true;
  for (size_t i = 0; i < m_size; i++) {
    if (m_value[i] == sep && t1) {
      t1 = false;
      continue;
    }
    if (t1) {
      s1.append(m_value[i]);
    } else {
      s2.append(m_value[i]);
    }
  }

  return Pair<String, String>(s1.to_string(), s2.to_string());
}

template <typename T> class Vector {
public:
  Vector() {
    m_values = NULL;
    m_size = 0;
    m_capacity = 0;
  }

  Vector(const Vector &other) {
    m_size = 0;
    m_values = NULL;
    setCapacity(other.m_size);

    for (size_t i = 0; i < other.m_size; i++) {
      push(other.m_values[i]);
    }
  }

  ~Vector() {
    for (size_t i = 0; i < m_size; i++) {
      m_values[i].~T();
    }
    free(m_values);
  }

  Vector &operator=(const Vector &other) {
    if (this != &other) {
      for (size_t i = 0; i < m_size; i++) {
        m_values[i].~T();
      }
      free(m_values);

      m_size = other.m_size;
      setCapacity(m_size);

      for (size_t i = 0; i < m_size; i++) {
        push(other.m_values[i]);
      }
    }

    return *this;
  }

  T &operator[](size_t i) {
    ASSERT(i < m_size);
    return m_values[i];
  }

  const T &operator[](size_t i) const {
    ASSERT(i < m_size);
    return m_values[i];
  }

  void setCapacity(size_t cap) {
    m_values = static_cast<T *>(
        realloc(static_cast<void *>(m_values), cap * sizeof(T)));
    m_capacity = cap;
  }

  void push(T val) {
    if (m_size + 1 > m_capacity) {
      size_t capacity = m_capacity == 0 ? 8 : m_capacity * 2;
      setCapacity(capacity);
    }

    new (&m_values[m_size]) T(val);
    m_size++;
  }

  T pop() {
    ASSERT(m_size > 0);
    m_size--;
    T val = m_values[m_size];
    m_values[m_size].~T();
    return val;
  }

  size_t size() const { return m_size; }

private:
  T *m_values;
  size_t m_size;
  size_t m_capacity;
};

} // namespace Container

namespace Fs {

template <typename F> void readLines(const char *path, F callback) {
  FILE *f = fopen(path, "r");

  Container::StringBuilder sb;

  while (true) {
    char ch[1];
    auto read = fread(ch, sizeof(char), 1, f);

    if (read < 1) {
      fclose(f);
      if (sb.length() > 0) {
        callback(sb.to_string());
      }
      break;
    }

    char c = ch[0];

    if (c == '\n') {
      callback(sb.to_string());
      sb.clear();
    } else {
      sb.append(c);
    }
  }
}

} // namespace Fs

namespace Linalg {

template <typename T, size_t ROWS, size_t COLS> class Matrix {
public:
  Matrix() {
    m_values = static_cast<T *>(malloc(ROWS * COLS * sizeof(T)));
    for (size_t y = 0; y < ROWS; y++)
      for (size_t x = 0; x < COLS; x++)
        operator()(x, y) = 0;
  }

  Matrix(const Matrix &m) : Matrix() {
    for (size_t y = 0; y < ROWS; y++)
      for (size_t x = 0; x < COLS; x++)
        operator()(x, y) = m(x, y);
  }

  ~Matrix() { free(m_values); }

  T &operator()(size_t x, size_t y) {
    ASSERT(x < COLS);
    ASSERT(y < ROWS);
    const size_t index = y * COLS + x;
    return m_values[index];
  }

  const T &operator()(size_t x, size_t y) const {
    ASSERT(x < COLS);
    ASSERT(y < ROWS);
    const size_t index = y * COLS + x;
    return m_values[index];
  }

  template <size_t OTHER_ROWS, size_t OTHER_COLS>
  Matrix<T, ROWS, OTHER_COLS>
  operator*(const Matrix<T, OTHER_ROWS, OTHER_COLS> &other) const {
    static_assert(COLS == OTHER_ROWS);

    Matrix<T, ROWS, OTHER_COLS> m;

    for (size_t y = 0; y < ROWS; y++)
      for (size_t x = 0; x < OTHER_COLS; x++)
        m(x, y) = 0;

    for (size_t i = 0; i < ROWS; i++)
      for (size_t j = 0; j < OTHER_COLS; j++)
        for (size_t k = 0; k < COLS; k++) {
          m(j, i) = m(j, i) + this->operator()(k, i) * other(j, k);
        }

    return m;
  }

  template <typename L> Matrix<T, ROWS, COLS> operator*(L val) {
    Matrix<T, ROWS, COLS> result;

    for (size_t y = 0; y < ROWS; y++)
      for (size_t x = 0; x < COLS; x++)
        result(x, y) *= val;

    return result;
  }

  Matrix operator=(const Matrix &other) {
    for (size_t y = 0; y < ROWS; y++)
      for (size_t x = 0; x < COLS; x++)
        operator()(x, y) = other(x, y);
    return *this;
  }

  void dump() const {
    for (size_t y = 0; y < ROWS; y++) {
      printf("[ ");
      for (size_t x = 0; x < COLS; x++) {
        printf("%f ", this->operator()(x, y));
      }
      printf(" ]\n");
    }
  }

private:
  T *m_values;
};

template <typename T, size_t SIZE> class Vector {
public:
  Vector() { m_values = static_cast<T *>(malloc(SIZE * sizeof(T))); }

  Vector(const Vector &v) {
    m_values = static_cast<T *>(malloc(SIZE * sizeof(T)));

    memcpy(m_values, v.m_values, SIZE * sizeof(T));
  }

  ~Vector() { free(m_values); }

  T &operator[](size_t i) {
    ASSERT(i < SIZE);
    return m_values[i];
  }

  const T &operator[](size_t i) const {
    ASSERT(i < SIZE);
    return m_values[i];
  }

  Vector<T, SIZE> &operator=(const Vector<T, SIZE> &other) {
    if (this != &other) {
      for (size_t i = 0; i < SIZE; i++)
        operator[](i) = other[i];
    }
    return *this;
  }

  template <size_t COLS>
  Vector<T, COLS> operator*(const Matrix<T, SIZE, COLS> &m) const {
    Vector<T, COLS> result;

    for (size_t x = 0; x < COLS; x++) {
      T sum = 0;
      for (size_t y = 0; y < SIZE; y++) {
        sum += m(x, y) * this->operator[](y);
      }
      result[x] = sum;
    }

    return result;
  }

  Vector<T, SIZE> operator-(const Vector<T, SIZE> &other) {
    Vector<T, SIZE> v = *this;
    for (size_t i = 0; i < SIZE; i++)
      v[i] -= other[i];
    return v;
  }

  T magSq() const {
    T sum = 0;
    for (size_t i = 0; i < SIZE; i++)
      sum += operator[](i) * operator[](i);
    return sum;
  }

  T mag() const { return std::sqrt(magSq()); }

  size_t argmax() const {
    size_t maxIndex = 0;
    T max = 0;

    for (size_t i = 0; i < SIZE; i++) {
      if (m_values[i] >= max) {
        maxIndex = i;
        max = m_values[i];
      }
    }

    return maxIndex;
  }

  void dump() const {
    printf("[ ");
    for (size_t i = 0; i < SIZE; i++) {
      printf("%f ", this->operator[](i));
    }
    printf(" ]\n");
  }

private:
  T *m_values;
};

} // namespace Linalg

namespace Layer {

template <typename T, size_t INP, size_t OUT> class Dense {
public:
  Dense() : m_matrix() {}

  Dense(const Dense &other) {
    m_matrix = Linalg::Matrix<T, INP + 1, OUT>(other.m_matrix);
  }

  Linalg::Vector<T, OUT> forward(const Linalg::Vector<T, INP> &input) const {
    Linalg::Vector<T, INP + 1> v;
    for (size_t i = 0; i < INP; i++)
      v[i] = input[i];

    v[INP] = 1;
    return v * m_matrix;
  }

  Dense operator=(const Dense other) {
    m_matrix = other.m_matrix;
    return *this;
  }

  Linalg::Matrix<T, INP + 1, OUT> m_matrix;
};

} // namespace Layer

namespace Activation {

template <typename T, size_t IN>
Linalg::Vector<T, IN> tanh(const Linalg::Vector<T, IN> &vec) {
  Linalg::Vector<T, IN> result;

  for (size_t i = 0; i < IN; i++) {
    result[i] = std::tanh(vec[i]);
  }

  return result;
}

template <typename T, size_t IN>
Linalg::Vector<T, IN> relu(const Linalg::Vector<T, IN> &vec) {
  Linalg::Vector<T, IN> result;

  for (size_t i = 0; i < IN; i++) {
    result[i] = vec[i] > 0 ? vec[i] : 0;
  }

  return result;
}

template <typename T, size_t IN>
Linalg::Vector<T, IN> fastSigmoid(const Linalg::Vector<T, IN> &vec) {
  Linalg::Vector<T, IN> result;

  for (size_t i = 0; i < IN; i++) {
    result[i] = vec[i] / (1 + abs(vec[i]));
  }

  return result;
}

template <typename T, size_t IN>
Linalg::Vector<T, IN> softmax(const Linalg::Vector<T, IN> &vec) {
  Linalg::Vector<T, IN> result;

  T sum = 0;

  for (size_t i = 0; i < IN; i++) {
    sum += std::exp(vec[i]);
  }

  for (size_t i = 0; i < IN; i++) {
    result[i] = std::exp(vec[i]) / sum;
  }

  return result;
}

} // namespace Activation

namespace Mutation {

template <typename T, size_t ROWS, size_t COLS>
void testMutate(Linalg::Matrix<T, ROWS, COLS> *matrix, float rate = 0.5) {
  std::random_device rd;
  std::mt19937 gen(rd());
  std::uniform_real_distribution<> dis(-10, 10);
  std::uniform_real_distribution<> prob(0, 1);

  for (size_t y = 0; y < ROWS; y++) {
    for (size_t x = 0; x < COLS; x++) {
      if (prob(gen) < rate) {
        matrix->operator()(x, y) = dis(gen);
      }
    }
  }
}

template <typename T, size_t ROWS, size_t COLS>
void normalMutate(Linalg::Matrix<T, ROWS, COLS> *matrix, float stddev) {
  std::random_device rd;
  std::mt19937 gen(rd());
  std::normal_distribution<> normal(0, stddev);

  for (size_t y = 0; y < ROWS; y++) {
    for (size_t x = 0; x < COLS; x++) {
      matrix->operator()(x, y) += normal(gen);
    }
  }
}

template <typename T, size_t ROWS, size_t COLS>
void normalMutate(Linalg::Matrix<T, ROWS, COLS> *matrix, float rate,
                  float stddev) {
  std::random_device rd;
  std::mt19937 gen(rd());
  std::uniform_real_distribution<> prob(0, 1);
  std::normal_distribution<> normal(0, stddev);

  for (size_t y = 0; y < ROWS; y++) {
    for (size_t x = 0; x < COLS; x++) {
      if (prob(gen) < rate) {
        matrix->operator()(x, y) += normal(gen);
      }
    }
  }
}

template <typename T, typename F, size_t ROWS, size_t COLS>
void costMutate(Linalg::Matrix<T, ROWS, COLS> *matrix, F f, T stddev) {
  size_t y = Util::random_range((size_t)0, ROWS);
  size_t x = Util::random_range((size_t)0, COLS);
  std::normal_distribution<> normalDist{0, stddev};
  T diff = normalDist(gen);

  auto cost = f();
  matrix->operator()(x, y) += diff;
  auto costPos = f();

  if (costPos > cost) {
    matrix->operator()(x, y) -= diff;
  }
}

} // namespace Mutation

} // namespace NNKek

#undef ASSERT

#endif
