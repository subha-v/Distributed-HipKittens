CXX ?= c++
CXXFLAGS ?= -std=c++20 -O2 -Wall -Wextra -Werror -pedantic

BUILD_DIR ?= build
TEST_SOURCE := pgl.cpp
CDNA3_HEADER := ../../../../../include/cdna3/types/global/pgl.cuh
CDNA4_HEADER := ../../../../../include/cdna4/types/global/pgl.cuh
CDNA3_TEST := $(BUILD_DIR)/pgl_cdna3
CDNA4_TEST := $(BUILD_DIR)/pgl_cdna4

.PHONY: all run clean

all: $(CDNA3_TEST) $(CDNA4_TEST)

$(CDNA3_TEST): $(TEST_SOURCE) $(CDNA3_HEADER)
	mkdir -p $(@D)
	$(CXX) $(CXXFLAGS) -DPGL_HEADER='"$(CDNA3_HEADER)"' $< -o $@

$(CDNA4_TEST): $(TEST_SOURCE) $(CDNA4_HEADER)
	mkdir -p $(@D)
	$(CXX) $(CXXFLAGS) -DPGL_HEADER='"$(CDNA4_HEADER)"' $< -o $@

run: all
	$(CDNA3_TEST)
	$(CDNA4_TEST)

clean:
	rm -rf $(BUILD_DIR)
