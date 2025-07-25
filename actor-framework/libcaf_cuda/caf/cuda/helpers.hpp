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
}
