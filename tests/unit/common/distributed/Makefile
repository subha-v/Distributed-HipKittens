CXX ?= c++
CXXFLAGS ?= -std=c++20 -O2 -Wall -Wextra -Werror -pedantic
CXXFLAGS += -Wno-unknown-pragmas

BUILD_DIR ?= build
TEST_SOURCE := host_semantics.cpp
CDNA3_HEADER := ../../../../include/cdna3/ops/group/distributed/distributed.cuh
CDNA4_HEADER := ../../../../include/cdna4/ops/group/distributed/distributed.cuh
CDNA3_HEADERS := $(wildcard ../../../../include/cdna3/ops/group/distributed/*.cuh) \
	$(wildcard ../../../../include/cdna3/ops/group/distributed/detail/*.cuh) \
	../../../../include/cdna3/types/global/pgl.cuh
CDNA4_HEADERS := $(wildcard ../../../../include/cdna4/ops/group/distributed/*.cuh) \
	$(wildcard ../../../../include/cdna4/ops/group/distributed/detail/*.cuh) \
	../../../../include/cdna4/types/global/pgl.cuh
CDNA3_TEST := $(BUILD_DIR)/distributed_cdna3
CDNA4_TEST := $(BUILD_DIR)/distributed_cdna4

.PHONY: all run clean

all: $(CDNA3_TEST) $(CDNA4_TEST)

$(CDNA3_TEST): $(TEST_SOURCE) $(CDNA3_HEADERS)
	mkdir -p $(@D)
	$(CXX) $(CXXFLAGS) -DDISTRIBUTED_HEADER='"$(CDNA3_HEADER)"' $< -o $@

$(CDNA4_TEST): $(TEST_SOURCE) $(CDNA4_HEADERS)
	mkdir -p $(@D)
	$(CXX) $(CXXFLAGS) -DDISTRIBUTED_HEADER='"$(CDNA4_HEADER)"' $< -o $@

run: all
	$(CDNA3_TEST)
	$(CDNA4_TEST)

clean:
	rm -rf $(BUILD_DIR)
