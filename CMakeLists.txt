cmake_minimum_required(VERSION 3.20)

project(TensorEECs VERSION 0.1.0 LANGUAGES CXX)

option(BUILD_SHARED_LIB "Build shared C++ library" ON)
option(BUILD_PYTHON_BINDINGS "Build pybind11 module" OFF)
option(BUILD_TESTS "Build C++ tests" ON)
option(BUILD_BENCHMARKS "Build C++ benchmark executable" ON)

set(TENSOREECS_DEFAULT_NUMERIC_TYPE "float64" CACHE STRING "Default runtime numeric type: float32|float64")
set(TENSOREECS_DEFAULT_STORAGE_MODE "sparse" CACHE STRING "Default runtime storage mode: sparse|dense")
set_property(CACHE TENSOREECS_DEFAULT_NUMERIC_TYPE PROPERTY STRINGS float32 float64)
set_property(CACHE TENSOREECS_DEFAULT_STORAGE_MODE PROPERTY STRINGS sparse dense)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

include(CheckIPOSupported)
check_ipo_supported(RESULT ipo_supported OUTPUT ipo_error)
if(ipo_supported)
  set(CMAKE_INTERPROCEDURAL_OPTIMIZATION_RELEASE ON)
  set(CMAKE_INTERPROCEDURAL_OPTIMIZATION_RELWITHDEBINFO ON)
endif()

list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake")
include(CompilerWarnings)

find_package(Eigen3 REQUIRED)

add_library(tensoreecs_obj OBJECT
  src/eec_runtime.cpp
)

target_include_directories(tensoreecs_obj PUBLIC
  ${CMAKE_CURRENT_SOURCE_DIR}/include
)

target_link_libraries(tensoreecs_obj PUBLIC Eigen3::Eigen)
target_compile_definitions(tensoreecs_obj PUBLIC
  TENSOREECS_DEFAULT_NUMERIC_TYPE_STR="${TENSOREECS_DEFAULT_NUMERIC_TYPE}"
  TENSOREECS_DEFAULT_STORAGE_MODE_STR="${TENSOREECS_DEFAULT_STORAGE_MODE}"
)
target_enable_strict_warnings(tensoreecs_obj)

if(BUILD_SHARED_LIB)
  add_library(tensoreecs SHARED $<TARGET_OBJECTS:tensoreecs_obj>)
else()
  add_library(tensoreecs STATIC $<TARGET_OBJECTS:tensoreecs_obj>)
endif()

add_library(TensorEECs::tensoreecs ALIAS tensoreecs)

target_include_directories(tensoreecs PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
)

target_link_libraries(tensoreecs PUBLIC Eigen3::Eigen)
target_enable_strict_warnings(tensoreecs)

if(BUILD_PYTHON_BINDINGS)
  add_subdirectory(python)
endif()

if(BUILD_TESTS)
  enable_testing()
  add_subdirectory(tests)
endif()

if(BUILD_BENCHMARKS)
  add_subdirectory(benchmarks)
endif()
