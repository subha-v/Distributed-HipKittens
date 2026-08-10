// Explicitly separate build entry point for the unmeasured one-launch D-prime
// composition. Never link this translation unit with gemm_rs_device_tile.cpp.
#define HK_GEMM_RS_DPRIME_EXPERIMENTAL 1
#include "gemm_rs_device_tile.cpp"
