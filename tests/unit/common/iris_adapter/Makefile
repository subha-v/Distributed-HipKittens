CXX ?= c++
CXXFLAGS ?= -std=c++20 -O2 -Wall -Wextra -Werror -pedantic

BUILD_DIR ?= build
TEST_SOURCE := iris_adapter.cpp
ADAPTER_HEADER := ../../../../distributed-kernels/common/iris_adapter.hpp
INCLUDE_ROOT := ../../../../include
CDNA3_HEADER := $(INCLUDE_ROOT)/cdna3/types/global/pgl.cuh
CDNA4_HEADER := $(INCLUDE_ROOT)/cdna4/types/global/pgl.cuh
FIXTURES := fixtures/kittens.cuh fixtures/iris/iris.hpp
CDNA3_TEST := $(BUILD_DIR)/iris_adapter_cdna3
CDNA4_TEST := $(BUILD_DIR)/iris_adapter_cdna4

.PHONY: all run clean

all: $(CDNA3_TEST) $(CDNA4_TEST)

$(CDNA3_TEST): $(TEST_SOURCE) $(ADAPTER_HEADER) $(CDNA3_HEADER) $(FIXTURES)
	mkdir -p $(@D)
	$(CXX) $(CXXFLAGS) -Ifixtures -I$(INCLUDE_ROOT) \
		-DPGL_HEADER='"cdna3/types/global/pgl.cuh"' $< -o $@

$(CDNA4_TEST): $(TEST_SOURCE) $(ADAPTER_HEADER) $(CDNA4_HEADER) $(FIXTURES)
	mkdir -p $(@D)
	$(CXX) $(CXXFLAGS) -Ifixtures -I$(INCLUDE_ROOT) \
		-DPGL_HEADER='"cdna4/types/global/pgl.cuh"' $< -o $@

run: all
	$(CDNA3_TEST)
	$(CDNA4_TEST)

clean:
	rm -rf $(BUILD_DIR)
