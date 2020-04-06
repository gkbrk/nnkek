CXX ?= c++

CXXFLAGS = -Wall -Wextra -Werror -pedantic -std=c++17
CXXFLAGS += -Ofast -flto -s
#CXXFLAGS += -O0 -ggdb
CXXFLAGS += -I.

EXAMPLES := $(wildcard examples/*.cpp)
OUTPUTS  := $(patsubst %.cpp, %.out, $(EXAMPLES))

all: $(OUTPUTS)
.PHONY: all

%.out: %.cpp NNKek.h
	$(CXX) $(CXXFLAGS) -o "$@" $<

clean:
	rm -f examples/*.out
.PHONY: clean

format:
	clang-format -i NNKek.h examples/*.cpp
.PHONY: format

reload:
	find . -type f -name '*.cpp' -o -name '*.h' > QtCreator.files
.PHONY: reload
