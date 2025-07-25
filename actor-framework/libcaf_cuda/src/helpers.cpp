#include "caf/cuda/helpers.hpp"


//gets a random number
int random_number() {
    static std::random_device rd;
    static std::mt19937 gen(rd());
    static std::uniform_int_distribution<int> distrib(INT_MIN, INT_MAX);
    return distrib(gen);  
}


