#project(caf_opencl C CXX)
include(CMakePrintHelpers)

# get header files; only needed by CMake generators,
# e.g., for creating proper Xcode projects
file(GLOB_RECURSE CAF_CUDA_HEADERS "caf/*.hpp")

#cmake_print_variables(CAF_CUDA_HEADERS)



# list cpp files excluding platform-dependent files
set(LIBCAF_CUDA_SRCS
  src/device.cpp
  src/global.cpp
  src/manager.cpp
  src/opencl_err.cpp
  src/platform.cpp
  src/program.cpp
)

# -- add targets ---------------------------------------------------------------

caf_add_component(
  cuda
  DEPENDENCIES
    PUBLIC
    CAF::core
      $<$<CXX_COMPILER_ID:MSVC>:ws2_32>
    PRIVATE
      CAF::internal
      #CAF::CORE
    HEADERS
    ${CAF_CUDA_HEADERS}
  SOURCES
    ${LIBCAF_CUDA_SRCS}
     )
