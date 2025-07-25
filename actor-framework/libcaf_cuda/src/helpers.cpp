#include "caf/cuda/helpers.hpp"

namespace caf::cuda {
//gets a random number
CAF_CUDA_EXPORT int random_number() {
    static std::random_device rd;
    static std::mt19937 gen(rd());
    static std::uniform_int_distribution<int> distrib(INT_MIN, INT_MAX);
    return distrib(gen);  
}

}
