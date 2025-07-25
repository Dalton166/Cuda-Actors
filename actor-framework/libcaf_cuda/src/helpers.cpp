#include "caf/cuda/helpers.hpp"

namespace caf::cuda {
//gets a random number
CAF_CUDA_EXPORT int random_number() {
    static std::random_device rd;
    static std::mt19937 gen(rd());
    static std::uniform_int_distribution<int> distrib(INT_MIN, INT_MAX);
    return distrib(gen);  
}


//helper function designed to read in a cubin from file
std::vector<char> load_cubin(const std::string& filename) {
  std::ifstream in(filename, std::ios::binary);
  if (!in)
    throw std::runtime_error("Failed to open CUBIN file: " + filename);

  return std::vector<char>(
      std::istreambuf_iterator<char>(in),
      std::istreambuf_iterator<char>());
}


}
