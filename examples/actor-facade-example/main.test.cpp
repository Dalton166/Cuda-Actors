
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

//verification of matrix multiplication on the gpu
void serial_matrix_multiply(const std::vector<int>& a,
                            const std::vector<int>& b,
                            std::vector<int>& c,
                            int N) {
  

 for (int i = 0; i < N; ++i) {
    for (int j = 0; j < N; ++j) {
      int sum = 0;
      for (int k = 0; k < N; ++k) {
        sum += a[i * N + k] * b[k * N + j];
      }
      c[i * N + j] = sum;
    }
  }
}


void test_mmul_from_cubin(caf::actor_system& sys, int N) {
  std::cout << "[TEST] Starting test_mmul_from_cubin\n";

  caf::cuda::manager& mgr = caf::cuda::manager::get();

  int THREADS = 32;
  int BLOCKS = (N + THREADS - 1) / THREADS;

  caf::cuda::nd_range dim(BLOCKS, BLOCKS, 1, THREADS, THREADS, 1);

  std::vector<int> h_a(N * N);
  std::vector<int> h_b(N * N);
  std::vector<int> h_c(N * N, 0);
  std::vector<int> h_ref(N * N, 0);
  std::vector<int> h_n(1, N);

  std::generate(h_a.begin(), h_a.end(), []() { return rand() % 10; });
  std::generate(h_b.begin(), h_b.end(), []() { return rand() % 10; });


  serial_matrix_multiply(h_a, h_b, h_ref, N);

  auto arg1 = caf::cuda::create_in_arg(h_a);
  auto arg2 = caf::cuda::create_in_arg(h_b);
  auto arg3 = caf::cuda::create_out_arg(h_c);
  auto arg4 = caf::cuda::create_in_arg(N);

  // Spawn actor from precompiled cubin file
  auto gpuActor = mgr.spawnFromCUBIN("../mmul.cubin", "matrixMul", dim,
                                  in<int>{}, in<int>{}, out<int>{}, in<int>{});

  sys.spawn([=](caf::event_based_actor* self_actor) {
    auto start = std::chrono::high_resolution_clock::now();

    self_actor->mail(gpuActor, arg1, arg2, arg3, arg4)
      .request(gpuActor, std::chrono::seconds(10))
      .then([=](const std::vector<output_buffer>& outputs) {
        auto end = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double> elapsed = end - start;

        std::vector<int> result;
        for (const auto& out : outputs) {
          std::visit([&](const auto& vec) {
            using T = std::decay_t<decltype(vec)>;
            if constexpr (std::is_same_v<T, std::vector<int>>) {
              result = vec;
            }
          }, out.data);
        }

        // Compare result with reference
        bool match = (result == h_ref);
        std::cout << "[INFO] Kernel round-trip time: " << elapsed.count() << " seconds\n";
        std::cout << (match ? "[PASS] GPU result matches reference\n" : "[FAIL] Mismatch in GPU result\n");

        self_actor->send_exit(gpuActor, caf::exit_reason::user_shutdown);
        self_actor->quit();
      });
  });

  sys.await_all_actors_done();
}




void caf_main(caf::actor_system& sys) {
  caf::cuda::manager::init(sys);
   test_mmul_from_cubin(sys,100);

   //test_mmul_from_cubin(sys,50);
   //test_mmul_from_cubin(sys,1024);

}




CAF_MAIN()
