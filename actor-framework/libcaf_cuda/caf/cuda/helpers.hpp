/*
 * A file full of helper functions that can be called by any class at anytime 
 * if desired 
 */
#pragma once
#include "caf/cuda/global.hpp"
#include <random>
#include <climits>

namespace caf::cuda {

	CAF_CUDA_EXPORT int random_number();
	CAF_CUDA_EXPORT std::vector<char> load_cubin(const std::string&);
	CAF_CUDA_EXPORT bool compile_nvrtc_program(const char* source, CUdevice device, std::vector<char>& ptx_out);

}
