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
            csrc/cpu/sort.cpp csrc/cpu/skeleton.cpp csrc/cpu/pooling.cpp csrc/cpu/probability.cpp
            csrc/cpu/regression.cpp csrc/cpu/sum.cpp)
target_include_directories(exacta_kernels PUBLIC csrc/include)
set_target_properties(exacta_kernels PROPERTIES POSITION_INDEPENDENT_CODE ON)
# The kernels reproduce reference expressions to the last bit, and the exactness proof of the
# distance transform assumes every product and sum is rounded once, as written: no fused
# multiply-adds, no extended precision. Compilers form fused multiply-adds by default whenever
# the target has them (GCC and Clang with -march=native or on aarch64; Intel's icx by default),
# so every supported compiler gets the flags that forbid it, and the build stops for any other.
set(EXACTA_FP_FLAGS "")
if(CMAKE_CXX_COMPILER_ID STREQUAL "MSVC")
  # Since Visual Studio 2022 (cl 19.30) /fp:precise does not contract unless /fp:contract is
  # given; earlier versions may contract under /fp:precise (Microsoft's /fp documentation).
  if(CMAKE_CXX_COMPILER_VERSION VERSION_LESS 19.30)
    message(FATAL_ERROR
      "Exacta needs MSVC 19.30 (Visual Studio 2022) or later: older versions contract "
      "a * b + c into a fused multiply-add under /fp:precise, which the kernels forbid. "
      "Found MSVC ${CMAKE_CXX_COMPILER_VERSION}.")
  endif()
  set(EXACTA_FP_FLAGS /fp:precise)
elseif(CMAKE_CXX_COMPILER_ID MATCHES "^(GNU|Clang|AppleClang|IntelLLVM)$")
  if(CMAKE_CXX_COMPILER_FRONTEND_VARIANT STREQUAL "MSVC")
    # clang-cl and icx on Windows: /fp:precise only sets contraction to "on" (within one
    # expression), so the clang option is passed through, after it.
    set(EXACTA_FP_FLAGS /fp:precise /clang:-ffp-contract=off)
  elseif(CMAKE_CXX_COMPILER_ID STREQUAL "IntelLLVM")
    # icx defaults to -fp-model=fast, which implies -ffp-contract=fast. -fp-model=precise only
    # lowers it to "on", so the contraction option comes after it, which fixes it at "off".
    set(EXACTA_FP_FLAGS -fp-model=precise -ffp-contract=off)
  else()
    # GCC contracts across statements by default (-ffp-contract=fast) and Clang within one
    # expression (=on); Apple clang follows Clang.
    set(EXACTA_FP_FLAGS -ffp-contract=off)
  endif()
else()
  message(FATAL_ERROR
    "Exacta's kernels must be compiled without floating-point contraction (fused multiply-add), "
    "and no flags for it are known for the compiler ${CMAKE_CXX_COMPILER_ID} "
    "(${CMAKE_CXX_COMPILER}). Supported: GNU, Clang, AppleClang, IntelLLVM and MSVC 19.30 or "
    "later. Add the compiler's flags in CMakeLists.txt to support another.")
endif()

# A flag that the compiler ignores or rejects would silently leave contraction on (MSVC only warns
# about unknown options), so each one must be accepted.
include(CheckCXXCompilerFlag)
foreach(flag IN LISTS EXACTA_FP_FLAGS)
  string(MAKE_C_IDENTIFIER "EXACTA_ACCEPTS_${flag}" accepted_var)
  check_cxx_compiler_flag("${flag}" ${accepted_var})
  if(NOT ${accepted_var})
    message(FATAL_ERROR
      "The ${CMAKE_CXX_COMPILER_ID} compiler ${CMAKE_CXX_COMPILER} does not accept ${flag}, "
      "which Exacta needs to forbid floating-point contraction in its kernels.")
  endif()
endforeach()
target_compile_options(exacta_kernels PRIVATE ${EXACTA_FP_FLAGS})

# The flags are what the compilers document; this checks what they do. The probe is built with
# the same flags and optimized, and must round a product and a sum separately. It is skipped
# only where it cannot run (cross-compiling, a universal macOS build).
list(LENGTH CMAKE_OSX_ARCHITECTURES exacta_osx_archs)
if(CMAKE_CROSSCOMPILING AND NOT CMAKE_CROSSCOMPILING_EMULATOR)
  message(STATUS "Exacta: cross-compiling, floating-point contraction probe not run")
elseif(exacta_osx_archs GREATER 1)
  message(STATUS "Exacta: universal macOS build, floating-point contraction probe not run")
else()
  if(CMAKE_CXX_COMPILER_FRONTEND_VARIANT STREQUAL "MSVC" OR CMAKE_CXX_COMPILER_ID STREQUAL "MSVC")
    set(exacta_probe_opt /O2)
  else()
    set(exacta_probe_opt -O2)
  endif()
  unset(EXACTA_PROBE_RUN CACHE)  # try_run caches its result; the flags may have changed
  unset(EXACTA_PROBE_COMPILED CACHE)
  try_run(EXACTA_PROBE_RUN EXACTA_PROBE_COMPILED
          "${CMAKE_BINARY_DIR}/exacta_fp_contract_probe"
          "${CMAKE_CURRENT_SOURCE_DIR}/csrc/fp_contract_probe.cpp"
          COMPILE_DEFINITIONS ${exacta_probe_opt} ${EXACTA_FP_FLAGS}
          RUN_OUTPUT_VARIABLE exacta_probe_output)
  if(NOT EXACTA_PROBE_COMPILED)
    message(FATAL_ERROR "Exacta: the floating-point contraction probe did not compile with "
            "${exacta_probe_opt} ${EXACTA_FP_FLAGS}; see CMakeFiles/CMakeConfigureLog.yaml.")
  elseif(NOT "${EXACTA_PROBE_RUN}" STREQUAL "0")
    string(REPLACE ";" " " exacta_fp_flags_text "${EXACTA_FP_FLAGS}")
    string(STRIP "${exacta_probe_output}" exacta_probe_output)
    message(FATAL_ERROR
      "The ${CMAKE_CXX_COMPILER_ID} compiler ${CMAKE_CXX_COMPILER} fuses multiplications and "
      "additions (or keeps extended precision) even with ${exacta_fp_flags_text}, which "
      "Exacta's kernels forbid (${exacta_probe_output}). Check CMAKE_CXX_FLAGS for -ffast-math, "
      "/fp:fast, -mfma or similar, and the compiler's floating-point model.")
  endif()
endif()
if(OpenMP_CXX_FOUND)
  target_link_libraries(exacta_kernels PUBLIC OpenMP::OpenMP_CXX)
endif()

nanobind_add_module(_C NB_STATIC STABLE_ABI csrc/bindings.cpp)
target_link_libraries(_C PRIVATE exacta_kernels)
install(TARGETS _C LIBRARY DESTINATION exacta)
