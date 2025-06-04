cmake_minimum_required(VERSION 3.16)
project(FlopsBenchmark)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Optimization flags for Intel CPUs
set(CMAKE_CXX_FLAGS_RELEASE "-O3 -march=native -mtune=native -ffast-math -funroll-loops")
set(CMAKE_BUILD_TYPE Release)

# Find Intel MKL (optional, for potential optimizations)
find_package(PkgConfig)
if(PkgConfig_FOUND)
    pkg_check_modules(MKL mkl-dynamic-lp64-seq)
    if(MKL_FOUND)
        add_definitions(-DUSE_MKL)
        link_directories(${MKL_LIBRARY_DIRS})
    endif()
endif()

# Create executable
add_executable(flops_benchmark benchmark.cpp)

# Link libraries
target_link_libraries(flops_benchmark pthread)
if(MKL_FOUND)
    target_link_libraries(flops_benchmark ${MKL_LIBRARIES})
endif()

# Compiler-specific optimizations
if(CMAKE_CXX_COMPILER_ID STREQUAL "GNU")
    target_compile_options(flops_benchmark PRIVATE -fopenmp)
    target_link_options(flops_benchmark PRIVATE -fopenmp)
elseif(CMAKE_CXX_COMPILER_ID STREQUAL "Intel")
    target_compile_options(flops_benchmark PRIVATE -qopenmp)
    target_link_options(flops_benchmark PRIVATE -qopenmp)
endif()