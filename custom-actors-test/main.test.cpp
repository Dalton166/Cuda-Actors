
#include "main.test.hpp"

#include <caf/all.hpp>
#include <caf/cuda/all.hpp>
#include <iostream>
#include <iomanip>
#include <vector>
#include <chrono>
#include <thread>
#include <algorithm>
#include <numeric>
#include "caf/actor_registry.hpp"



using namespace caf;
using namespace std::chrono_literals;

const char* kernel_code = R"(
extern "C" __global__
void compare_strings(const char* a, const char* b, int* result, int * length) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < * length) {
        result[idx] = (a[idx] == b[idx]) ? 1 : 0;
    }
}
)";

const char* matrixMulKernel2 = R"(
extern "C" __global__
void matrixMul(const int* a, const int* b, int* c, int *N_val) {
    int N = *N_val;
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (row < N && col < N) {
        int temp = 0;
        for (int k = 0; k < N; ++k) {
            temp += a[row * N + k] * b[k * N + col];
        }
        c[row * N + col] = temp;
    }
}
)";


const char* matrixMulKernel = R"(
extern "C" __global__
void matrixMul(const int* a, const int* b, int* c, int N) {
    //printf("%d\n",N);
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (row < N && col < N) {
        int temp = 0;
        for (int k = 0; k < N; ++k) {
            temp += a[row * N + k] * b[k * N + col];
        }
        c[row * N + col] = temp;
    }
}
)";


//commands classes used to launch kernels 
using mmulCommand = caf::cuda::command_runner<in<int>,in<int>,out<int>,int<int>>;
using mmulGenCommand = caf::cuda::command_runner<in_out<int>,in<int>,in<int>,in<int>>;



// Define atoms for CPU and GPU
using cpu_atom = caf::atom_constant<caf::atom("cpu")>;
using gpu_atom = caf::atom_constant<caf::atom("gpu")>;

// Actor state
struct my_actor_state {
  static inline const char* name = "my_actor";
  int last_N = 0; // example state variable
};

// Stateful actor behavior
caf::behavior my_actor_fun(caf::stateful_actor<my_actor_state>* self) {
  return {
    // 1st handler: Just int N
    [=](int N) {
      self->state.last_N = N;
      aout(self) << "[my_actor] Received N = " << N << "\n";
    },

    // 2nd handler: GPU atom + matrices + N
    [=](gpu_atom, const std::vector<int>& matrixA,
        const std::vector<int>& matrixB, int N) {
      aout(self) << "[my_actor] GPU task: N = " << N
                 << ", matrixA size = " << matrixA.size()
                 << ", matrixB size = " << matrixB.size() << "\n";
      // TODO: implement GPU logic here
    },

    // 3rd handler: CPU atom + matrices + N
    [=](cpu_atom, const std::vector<int>& matrixA,
        const std::vector<int>& matrixB, int N) {
      aout(self) << "[my_actor] CPU task: N = " << N
                 << ", matrixA size = " << matrixA.size()
                 << ", matrixB size = " << matrixB.size() << "\n";
      // TODO: implement CPU logic here
    }
  };
}





void caf_main(caf::actor_system& sys) {
  caf::cuda::manager::init(sys);


}




CAF_MAIN()
