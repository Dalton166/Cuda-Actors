#pragma once

#include <string>
#include <iostream>
#include <stdexcept>

#include <caf/logger.hpp>
#include <cuda.h>
#include "caf/cuda/types.hpp" //be sure to include any caf-cuda headers after this one
#include "caf/cuda/nd_range.hpp"
#include "caf/cuda/cuda-actors.hpp"
#include "caf/cuda/utility.hpp"
#include <nvrtc.h>

//a strange fix required in order to get the .so files to become viewable for binaries
//linking against them, if this is not defined with classes you want viewable then
//the linker will complain
#if defined(_MSC_VER)
  #define CAF_CUDA_EXPORT __declspec(dllexport)
#else
  #define CAF_CUDA_EXPORT __attribute__((visibility("default")))
#endif



//memory access flags, required for identifying which
//gpu buffers are input and output buffers
#define IN 0 
#define IN_OUT 1
#define OUT 2
#define NOT_IN_USE -1


//argument tags
struct input_tag {};
struct output_tag {};
struct input_output_tag {};


//struct wrappers to hold store buffers to declare them as in or out 
template <typename T>
struct in {
    using value_type = T;
    std::vector<T> buffer;
};

template <typename T>
struct out {
    using value_type = T;
    std::vector<T> buffer;
};

template <typename T>
struct in_out {
    using value_type = T;
    std::vector<T> buffer;
};


//helper function to check errors
void inline check(CUresult result, const char* msg) {
    if (result != CUDA_SUCCESS) {
        const char* err_str;
        cuGetErrorString(result, &err_str);
        std::cerr << "CUDA Driver API Error (" << msg << "): " << err_str << "\n";
        exit(1);
    }
}







// Check CUDA errors macro
#define CHECK_CUDA(call) \
    do { CUresult err = call; if (err != CUDA_SUCCESS) { \
        const char* errStr; cuGetErrorString(err, &errStr); \
        std::cerr << "CUDA Error: " << errStr << std::endl; exit(1); }} while(0)

// Check NVRTC errors macro
#define CHECK_NVRTC(call) \
    do { nvrtcResult res = call; if (res != NVRTC_SUCCESS) { \
        std::cerr << "NVRTC Error: " << nvrtcGetErrorString(res) << std::endl; exit(1); }} while(0)




namespace caf::cuda {

inline std::string opencl_error(int /*err*/) {
  return "CUDA support disabled";
}

inline std::string event_status(void* /*event*/) {
  return "CUDA support disabled";
}

//For right now this gets commented out to fix a compiler error but may be useful 
//later on
//inline std::ostream& operator<<(std::ostream& os, int /*device_type*/) {
  //os << "CUDA disabled";
 // return os;
//}

} // namespace caf::cuda
