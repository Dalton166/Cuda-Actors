#pragma once

#include <string>
#include <iostream>
#include <stdexcept>

#include <caf/logger.hpp>


#include "caf/opencl/cuda-actors.hpp"




//a strange fix required in order to get the .so files to become viewable for binaries
//linking against them, if this is not defined with classes you want viewable then
//the linker will complain
#if defined(_MSC_VER)
  #define CAF_OPENCL_EXPORT __declspec(dllexport)
#else
  #define CAF_OPENCL_EXPORT __attribute__((visibility("default")))
#endif


namespace caf::opencl {

inline std::string opencl_error(int /*err*/) {
  return "OpenCL support disabled";
}

inline std::string event_status(void* /*event*/) {
  return "OpenCL support disabled";
}

//For right now this gets commented out to fix a compiler error but may be useful 
//later on
//inline std::ostream& operator<<(std::ostream& os, int /*device_type*/) {
  //os << "OpenCL disabled";
 // return os;
//}

} // namespace caf::opencl
