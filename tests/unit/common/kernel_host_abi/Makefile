CXX ?= c++
CXXFLAGS ?= -std=c++20 -O2 -Wall -Wextra -Werror -pedantic

BUILD_DIR ?= build
TEST_SOURCE := kernel_host_abi.cpp
KERNEL_HEADERS := \
	../../../../distributed-kernels/common/iris_adapter.hpp \
	../../../../distributed-kernels/gemm_rs/gemm_rs_hk_adapter.cuh \
	../../../../distributed-kernels/gemm_rs/gemm_rs_host_abi.hpp \
	../../../../distributed-kernels/fused_moe/moe_hk_adapter.cuh \
	../../../../distributed-kernels/fused_moe/moe_mps_adapter.cuh \
	../../../../distributed-kernels/fused_moe/moe_host_abi.hpp
INCLUDE_ROOT := ../../../../include
CDNA3_HEADER := $(INCLUDE_ROOT)/cdna3/types/global/pgl.cuh
CDNA4_HEADER := $(INCLUDE_ROOT)/cdna4/types/global/pgl.cuh
FIXTURES := fixtures/kittens.cuh fixtures/iris/iris.hpp
CDNA3_TEST := $(BUILD_DIR)/kernel_host_abi_cdna3
CDNA4_TEST := $(BUILD_DIR)/kernel_host_abi_cdna4

.PHONY: all run clean

all: $(CDNA3_TEST) $(CDNA4_TEST)

$(CDNA3_TEST): $(TEST_SOURCE) $(KERNEL_HEADERS) $(CDNA3_HEADER) $(FIXTURES)
	mkdir -p $(@D)
	$(CXX) $(CXXFLAGS) -Ifixtures -I$(INCLUDE_ROOT) \
		-DPGL_HEADER='"cdna3/types/global/pgl.cuh"' $< -o $@

$(CDNA4_TEST): $(TEST_SOURCE) $(KERNEL_HEADERS) $(CDNA4_HEADER) $(FIXTURES)
	mkdir -p $(@D)
	$(CXX) $(CXXFLAGS) -Ifixtures -I$(INCLUDE_ROOT) \
		-DPGL_HEADER='"cdna4/types/global/pgl.cuh"' $< -o $@

run: all
	$(CDNA3_TEST)
	$(CDNA4_TEST)

clean:
	rm -rf $(BUILD_DIR)
