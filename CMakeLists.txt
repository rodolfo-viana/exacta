cmake_minimum_required(VERSION 3.18...3.31)
project(exacta LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(Python 3.10 COMPONENTS Interpreter Development.Module REQUIRED)
find_package(nanobind CONFIG REQUIRED)
find_package(OpenMP)

# Kernels behind a plain C interface (csrc/include/exacta/kernels.h). They take raw pointers
# and never link against PyTorch, so the wheel does not depend on a specific torch version.
add_library(exacta_kernels STATIC csrc/cpu/edt.cpp)
target_include_directories(exacta_kernels PUBLIC csrc/include)
set_target_properties(exacta_kernels PROPERTIES POSITION_INDEPENDENT_CODE ON)
if(OpenMP_CXX_FOUND)
  target_link_libraries(exacta_kernels PUBLIC OpenMP::OpenMP_CXX)
endif()

nanobind_add_module(_C NB_STATIC STABLE_ABI csrc/bindings.cpp)
target_link_libraries(_C PRIVATE exacta_kernels)
install(TARGETS _C LIBRARY DESTINATION exacta)
