CXX ?= c++
CXXFLAGS ?= -std=c++20 -O2 -Wall -Wextra -Werror -pedantic

BUILD_DIR ?= build
TEST_SOURCE := kernel_host_abi_mi300x.cpp
KERNEL_HEADERS := \
	../../../../distributed-kernels/common/iris_adapter.hpp \
	../../../../distributed-kernels/gemm_rs/gemm_rs_hk_adapter.cuh \
	../../../../distributed-kernels/gemm_rs/gemm_rs_host_abi.hpp \
	../../../../distributed-kernels/gemm_rs/gemm_rs_mi300x_constants.cuh \
	../../../../distributed-kernels/gemm_rs/gemm_rs_mi300x_host_abi.hpp
INCLUDE_ROOT := ../../../../include
CDNA3_HEADER := $(INCLUDE_ROOT)/cdna3/types/global/pgl.cuh
CDNA4_HEADER := $(INCLUDE_ROOT)/cdna4/types/global/pgl.cuh
# The mi300x host ABI shares the gfx950 test's IRIS/kittens stub fixtures.
FIXTURES := ../kernel_host_abi/fixtures/kittens.cuh \
	../kernel_host_abi/fixtures/iris/iris.hpp
CDNA3_TEST := $(BUILD_DIR)/kernel_host_abi_mi300x_cdna3
CDNA4_TEST := $(BUILD_DIR)/kernel_host_abi_mi300x_cdna4

.PHONY: all run clean

all: $(CDNA3_TEST) $(CDNA4_TEST)

$(CDNA3_TEST): $(TEST_SOURCE) $(KERNEL_HEADERS) $(CDNA3_HEADER) $(FIXTURES)
	mkdir -p $(@D)
	$(CXX) $(CXXFLAGS) -I../kernel_host_abi/fixtures -I$(INCLUDE_ROOT) \
		-DPGL_HEADER='"cdna3/types/global/pgl.cuh"' $< -o $@

$(CDNA4_TEST): $(TEST_SOURCE) $(KERNEL_HEADERS) $(CDNA4_HEADER) $(FIXTURES)
	mkdir -p $(@D)
	$(CXX) $(CXXFLAGS) -I../kernel_host_abi/fixtures -I$(INCLUDE_ROOT) \
		-DPGL_HEADER='"cdna4/types/global/pgl.cuh"' $< -o $@

run: all
	$(CDNA3_TEST)
	$(CDNA4_TEST)

clean:
	rm -rf $(BUILD_DIR)
