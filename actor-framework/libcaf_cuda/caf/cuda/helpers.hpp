/*
 * A file full of helper functions that can be called by any class at anytime 
 * if desired 
 */
#pragma once
#include "caf/cuda/global.hpp"
#include <random>
#include <climits>
#include <fstream>

namespace caf::cuda {

	CAF_CUDA_EXPORT int random_number();
	CAF_CUDA_EXPORT std::vector<char> load_cubin(const std::string&);
	CAF_CUDA_EXPORT bool compile_nvrtc_program(const char* source, CUdevice device, std::vector<char>& ptx_out);

// Extract first matching vector<T> from outputs
template <typename T>
std::optional<std::vector<T>> extract_vector_or_empty(const std::vector<output_buffer>& outputs) {
    for (const auto& out : outputs) {
        if (auto ptr = std::get_if<std::vector<T>>(&out.data)) {
            return *ptr; // copy found vector<T>
        }
    }
    return std::nullopt; // none found
}

// Extract first matching vector<T> from outputs
template <typename T>
std::vector<T> extract_vector(const std::vector<output_buffer>& outputs) {
    for (const auto& out : outputs) {
        if (auto ptr = std::get_if<std::vector<T>>(&out.data)) {
            return *ptr; // copy and return matching vector<T>
        }
    }
    return {}; // no matching vector found, return empty vector
}




}
