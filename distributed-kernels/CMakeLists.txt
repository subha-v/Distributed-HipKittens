cmake_minimum_required(VERSION 3.21)

# Select hipcc before project() performs compiler detection. A caller-provided
# toolchain, CMAKE_CXX_COMPILER, or CXX environment variable remains authoritative.
set(ROCM_PATH "$ENV{ROCM_PATH}" CACHE PATH "ROCm install prefix")
if(NOT ROCM_PATH)
  set(ROCM_PATH "/opt/rocm" CACHE PATH "ROCm install prefix" FORCE)
endif()
if((NOT DEFINED CMAKE_TOOLCHAIN_FILE OR CMAKE_TOOLCHAIN_FILE STREQUAL "") AND
   (NOT DEFINED CMAKE_CXX_COMPILER OR CMAKE_CXX_COMPILER STREQUAL ""))
  if(NOT DEFINED ENV{CXX} OR "$ENV{CXX}" STREQUAL "")
    set(CMAKE_CXX_COMPILER "${ROCM_PATH}/bin/hipcc" CACHE FILEPATH
        "C++ compiler used for HIP kernels")
  endif()
endif()

project(distributed_kernels LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)
if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
  set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
  set_property(CACHE CMAKE_BUILD_TYPE PROPERTY STRINGS
               Debug Release RelWithDebInfo MinSizeRel)
endif()

get_filename_component(HIPKITTENS_ROOT "${CMAKE_CURRENT_LIST_DIR}/.." ABSOLUTE)
if(NOT EXISTS "${HIPKITTENS_ROOT}/include/kittens.cuh")
  message(FATAL_ERROR
    "Bad HIPKITTENS_ROOT (${HIPKITTENS_ROOT}): missing include/kittens.cuh")
endif()

message(STATUS "HipKittens root: ${HIPKITTENS_ROOT}")
message(STATUS "ROCm prefix: ${ROCM_PATH}")
message(STATUS "C++ compiler: ${CMAKE_CXX_COMPILER}")

# -----------------------------------------------------------------------------
# GPU target
# -----------------------------------------------------------------------------
set(GPU_TARGET "CDNA4" CACHE STRING "GPU target architecture: CDNA3|CDNA4")
set_property(CACHE GPU_TARGET PROPERTY STRINGS CDNA3 CDNA4)

if(GPU_TARGET STREQUAL "CDNA3")
  set(DK_GPU_ARCH gfx942)
  set(DK_GPU_DEFINITION KITTENS_CDNA3)
  set(DK_GPU_COMPILE_OPTIONS "")
elseif(GPU_TARGET STREQUAL "CDNA4")
  set(DK_GPU_ARCH gfx950)
  set(DK_GPU_DEFINITION KITTENS_CDNA4)
  set(DK_GPU_COMPILE_OPTIONS -ffast-math)
else()
  message(FATAL_ERROR "Unknown GPU_TARGET: ${GPU_TARGET}")
endif()

message(STATUS "GPU target: ${GPU_TARGET} (${DK_GPU_ARCH})")

# -----------------------------------------------------------------------------
# IRIS C++ source selection
# -----------------------------------------------------------------------------
# IRIS_SOURCE_DIR may name the C++ development directory itself, the irisx
# directory, or an IRIS repository checkout. The first matching include tree is
# used. No absolute workstation path is stored in this source file.
set(IRIS_SOURCE_DIR "" CACHE PATH
    "IRIS C++ source directory (normally irisx/development)")
option(DK_FETCH_IRIS
       "Fetch the historical IRIS C++ branch when no local source is available"
       ON)
set(IRIS_FETCH_REPOSITORY "https://github.com/ROCm/iris.git" CACHE STRING
    "Fallback IRIS repository")
set(IRIS_FETCH_TAG "muhaawd/irisx" CACHE STRING
    "Fallback IRIS revision; use an immutable commit for reproducible builds")

function(dk_resolve_iris_root candidate output_variable)
  set(_iris_candidates
    "${candidate}"
    "${candidate}/development"
    "${candidate}/irisx/development"
    "${candidate}/irisx"
  )
  foreach(_iris_candidate IN LISTS _iris_candidates)
    if(EXISTS "${_iris_candidate}/include/iris/iris.hpp" AND
       EXISTS "${_iris_candidate}/include/iris/logger.hpp")
      get_filename_component(_iris_root "${_iris_candidate}" REALPATH)
      set(${output_variable} "${_iris_root}" PARENT_SCOPE)
      return()
    endif()
  endforeach()
  set(${output_variable} "" PARENT_SCOPE)
endfunction()

set(_dk_iris_origin "")
if(IRIS_SOURCE_DIR)
  dk_resolve_iris_root("${IRIS_SOURCE_DIR}" DK_IRIS_ROOT)
  if(NOT DK_IRIS_ROOT)
    message(FATAL_ERROR
      "IRIS_SOURCE_DIR='${IRIS_SOURCE_DIR}' does not contain the IRIS C++ "
      "headers (expected include/iris/iris.hpp, optionally below "
      "irisx/development).")
  endif()
  set(_dk_iris_origin "explicit IRIS_SOURCE_DIR")
else()
  # Portable sibling convention:
  #   <workspace>/Distributed-HipKittens
  #   <workspace>/iris/irisx/development
  get_filename_component(_dk_iris_sibling
    "${HIPKITTENS_ROOT}/../iris/irisx/development" ABSOLUTE)
  dk_resolve_iris_root("${_dk_iris_sibling}" DK_IRIS_ROOT)
  if(DK_IRIS_ROOT)
    set(_dk_iris_origin "sibling checkout")
  endif()
endif()

if(NOT DK_IRIS_ROOT)
  if(NOT DK_FETCH_IRIS)
    message(FATAL_ERROR
      "No local IRIS C++ source was found. Set IRIS_SOURCE_DIR or enable "
      "DK_FETCH_IRIS.")
  endif()

  include(FetchContent)
  FetchContent_Declare(dk_iris_source
    GIT_REPOSITORY "${IRIS_FETCH_REPOSITORY}"
    GIT_TAG "${IRIS_FETCH_TAG}"
  )
  FetchContent_GetProperties(dk_iris_source)
  if(NOT dk_iris_source_POPULATED)
    message(STATUS
      "Fetching IRIS C++ source from ${IRIS_FETCH_REPOSITORY}@${IRIS_FETCH_TAG}")
    FetchContent_Populate(dk_iris_source)
  endif()
  dk_resolve_iris_root("${dk_iris_source_SOURCE_DIR}" DK_IRIS_ROOT)
  if(NOT DK_IRIS_ROOT)
    message(FATAL_ERROR
      "Fetched IRIS source does not contain include/iris/iris.hpp in a "
      "recognized irisx layout.")
  endif()
  set(_dk_iris_origin "fetched fallback")
endif()

message(STATUS "IRIS C++ root (${_dk_iris_origin}): ${DK_IRIS_ROOT}")

# IRIS's C++ API is header-only. The development snapshot's own CMake project
# currently enables its examples, benchmarks, and tests unconditionally, so do
# not add it as a subdirectory. This target mirrors the dependency contract of
# that project's iris::iris interface target without configuring unrelated work.
find_package(MPI REQUIRED COMPONENTS CXX)
find_package(spdlog CONFIG QUIET)
if(NOT TARGET spdlog::spdlog)
  include(FetchContent)
  set(SPDLOG_BUILD_EXAMPLE OFF CACHE BOOL "" FORCE)
  set(SPDLOG_BUILD_TESTS OFF CACHE BOOL "" FORCE)
  set(SPDLOG_BUILD_BENCH OFF CACHE BOOL "" FORCE)
  FetchContent_Declare(dk_spdlog
    GIT_REPOSITORY https://github.com/gabime/spdlog.git
    GIT_TAG v1.16.0
    GIT_SHALLOW TRUE
  )
  FetchContent_MakeAvailable(dk_spdlog)
endif()

add_library(dk_iris INTERFACE)
add_library(iris::iris ALIAS dk_iris)
target_include_directories(dk_iris SYSTEM INTERFACE
  "${DK_IRIS_ROOT}/include"
)
target_link_libraries(dk_iris INTERFACE
  MPI::MPI_CXX
  spdlog::spdlog
)

# -----------------------------------------------------------------------------
# Python extension dependencies and shared kernel configuration
# -----------------------------------------------------------------------------
find_package(Python3 REQUIRED COMPONENTS Interpreter Development)
execute_process(
  COMMAND "${Python3_EXECUTABLE}" -m pybind11 --cmakedir
  RESULT_VARIABLE _dk_pybind11_result
  OUTPUT_VARIABLE _dk_pybind11_cmake_dir
  OUTPUT_STRIP_TRAILING_WHITESPACE
  ERROR_QUIET
)
if(_dk_pybind11_result STREQUAL "0" AND
   NOT _dk_pybind11_cmake_dir STREQUAL "")
  set(pybind11_DIR "${_dk_pybind11_cmake_dir}")
endif()
find_package(pybind11 CONFIG REQUIRED)

add_library(dk_common INTERFACE)
target_compile_definitions(dk_common INTERFACE
  "${DK_GPU_DEFINITION}"
  $<$<STREQUAL:${GPU_TARGET},CDNA4>:HIP_ENABLE_WARP_SYNC_BUILTINS>
)
target_compile_options(dk_common INTERFACE
  "--offload-arch=${DK_GPU_ARCH}"
  ${DK_GPU_COMPILE_OPTIONS}
  -w
  -Rpass-analysis=kernel-resource-usage
)
target_include_directories(dk_common INTERFACE
  "${HIPKITTENS_ROOT}/include"
  "${ROCM_PATH}/include"
)
target_link_libraries(dk_common INTERFACE iris::iris)

# -----------------------------------------------------------------------------
# Validated example discovery
# -----------------------------------------------------------------------------
set(DK_BUILD "ALL" CACHE STRING
    "Kernel module to build: ALL or one validated kernel directory")

# These are source-port candidates, not buildable/promoted kernels. Keep the
# exclusion explicit so adding a kernel.cpp during porting cannot silently put
# an unvalidated source into DK_BUILD=ALL.
set(DK_UNVALIDATED_PORT_DIRS gemm_rs fused_moe)

file(GLOB _dk_children
  CONFIGURE_DEPENDS
  LIST_DIRECTORIES true
  RELATIVE "${CMAKE_CURRENT_SOURCE_DIR}"
  "${CMAKE_CURRENT_SOURCE_DIR}/*"
)
set(DK_KERNEL_DIRS "")
foreach(_dk_child IN LISTS _dk_children)
  if(NOT IS_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}/${_dk_child}" OR
     NOT EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/${_dk_child}/kernel.cpp")
    continue()
  endif()

  list(FIND DK_UNVALIDATED_PORT_DIRS "${_dk_child}" _dk_unvalidated_index)
  if(NOT _dk_unvalidated_index EQUAL -1)
    message(STATUS "Excluding unvalidated port from DK_BUILD: ${_dk_child}")
    continue()
  endif()
  list(APPEND DK_KERNEL_DIRS "${_dk_child}")
endforeach()
list(SORT DK_KERNEL_DIRS)

if(NOT DK_KERNEL_DIRS)
  message(FATAL_ERROR
    "No validated kernel directories found (expected */kernel.cpp under "
    "${CMAKE_CURRENT_SOURCE_DIR})")
endif()
set_property(CACHE DK_BUILD PROPERTY STRINGS ALL ${DK_KERNEL_DIRS})
message(STATUS "Validated kernels: ${DK_KERNEL_DIRS}")

# Each directory gets a unique CMake target but preserves the existing Python
# module name, tk_kernel, because the checked-in examples import that API.
function(add_dk_kernel directory)
  set(_dk_source "${CMAKE_CURRENT_SOURCE_DIR}/${directory}/kernel.cpp")
  set(_dk_target "${directory}")

  pybind11_add_module(${_dk_target} "${_dk_source}")
  target_link_libraries(${_dk_target} PRIVATE dk_common)
  set_target_properties(${_dk_target} PROPERTIES
    PREFIX ""
    OUTPUT_NAME "tk_kernel"
    SUFFIX ".${Python3_SOABI}.so"
    LIBRARY_OUTPUT_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}/${directory}"
  )
  add_custom_command(TARGET ${_dk_target} POST_BUILD
    COMMAND "${CMAKE_COMMAND}" -E echo
            "Built: ${directory}/tk_kernel.${Python3_SOABI}.so"
  )
endfunction()

if(DK_BUILD STREQUAL "ALL")
  foreach(_dk_directory IN LISTS DK_KERNEL_DIRS)
    add_dk_kernel("${_dk_directory}")
  endforeach()
else()
  list(FIND DK_KERNEL_DIRS "${DK_BUILD}" _dk_kernel_index)
  if(_dk_kernel_index EQUAL -1)
    message(FATAL_ERROR
      "DK_BUILD='${DK_BUILD}' is not validated. Available: ${DK_KERNEL_DIRS}")
  endif()
  add_dk_kernel("${DK_BUILD}")
endif()

# Shared IRIS allocation/IPC bindings, emitted next to the Python examples.
pybind11_add_module(iris_py "${CMAKE_CURRENT_SOURCE_DIR}/iris_py.cpp")
target_link_libraries(iris_py PRIVATE dk_common)
set_target_properties(iris_py PROPERTIES
  PREFIX ""
  OUTPUT_NAME "iris_py"
  SUFFIX ".${Python3_SOABI}.so"
  LIBRARY_OUTPUT_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}"
)
add_custom_command(TARGET iris_py POST_BUILD
  COMMAND "${CMAKE_COMMAND}" -E echo "Built: iris_py.${Python3_SOABI}.so"
)
