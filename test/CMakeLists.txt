cmake_minimum_required(VERSION 3.16.3)
project(CUDA_ACTORS)

# 1) Enforce C++17 (needs modern C++ features)
set(CMAKE_CXX_STANDARD        17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS      OFF)

# 2) Set CAF source and build directories
set(CAF_SRC "/student/nqr159/cuda-actors/actor-framework")
set(CAF_BUILD "${CAF_SRC}/build")

# 3) Include directories (CAF + ViennaCL)
include_directories(
  "${CAF_SRC}/include"
  "${CAF_SRC}/libcaf_io"
  "${CAF_SRC}/libcaf_core"
  "${CAF_SRC}/libcaf_opencl"
  "${CAF_BUILD}/libcaf_core"  # for generated build_config.hpp  
  "${CAF_SRC}/libcaf_cuda"
)


# 5) Declare your executable
add_executable(test main.test.cpp)

# 6) Link against CAF shared libraries and OpenCL
target_link_libraries(test
  PRIVATE
    "${CAF_BUILD}/libcaf_core/libcaf_core.so"
    "${CAF_BUILD}/libcaf_io/libcaf_io.so"
    "${CAF_BUILD}/libcaf_cuda/libcaf_cuda.so"
)

# — Optional: extra include dirs (e.g., ViennaCL) —
# target_include_directories(laSolv PRIVATE
#   "/student/nqr159/opencl-actors/OpenCL/code/viennacl-dev"
# )
