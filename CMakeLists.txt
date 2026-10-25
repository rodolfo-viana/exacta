cmake_minimum_required(VERSION 3.18...3.31)
project(exacta LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# scikit-build-core sets SKBUILD_SABI_COMPONENT to Development.SABIModule when the wheel targets the
# stable ABI (wheel.py-api in pyproject.toml); without it nanobind quietly builds for the running
# Python only, and the abi3-tagged wheel fails to import on other versions.
find_package(Python 3.10 COMPONENTS Interpreter Development.Module ${SKBUILD_SABI_COMPONENT} REQUIRED)
find_package(nanobind CONFIG REQUIRED)
find_package(OpenMP)

# Kernels behind a plain C interface (csrc/include/exacta/kernels.h). They take raw pointers
# and never link against PyTorch, so the wheel does not depend on a specific torch version.
add_library(exacta_kernels STATIC csrc/cpu/counts.cpp csrc/cpu/edt.cpp csrc/cpu/extents.cpp
            csrc/cpu/label.cpp csrc/cpu/mesh.cpp csrc/cpu/meshing.cpp csrc/cpu/morphology.cpp
            csrc/cpu/sort.cpp csrc/cpu/skeleton.cpp csrc/cpu/pooling.cpp)
target_include_directories(exacta_kernels PUBLIC csrc/include)
set_target_properties(exacta_kernels PROPERTIES POSITION_INDEPENDENT_CODE ON)
# The kernels reproduce reference expressions to the last bit, so the compiler must round every
# product and sum as written: no fused multiply-adds (GCC and Clang may form them otherwise
# whenever the target has FMA, as with -march=native).
if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
  target_compile_options(exacta_kernels PRIVATE -ffp-contract=off)
endif()
if(OpenMP_CXX_FOUND)
  target_link_libraries(exacta_kernels PUBLIC OpenMP::OpenMP_CXX)
endif()

nanobind_add_module(_C NB_STATIC STABLE_ABI csrc/bindings.cpp)
target_link_libraries(_C PRIVATE exacta_kernels)
install(TARGETS _C LIBRARY DESTINATION exacta)
