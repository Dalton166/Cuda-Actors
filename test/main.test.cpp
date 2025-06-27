/*
 * This file contains tests for the CUDA actor facade, including a concurrent matrix multiplication benchmark.
 */

#include "main.test.hpp"

#include <caf/all.hpp>
#include <caf/cuda/all.hpp>
#include <iostream>
#include <iomanip>
#include <vector>
#include <chrono>
#include <thread>

using namespace caf;
using namespace std::chrono_literals;

const char* kernel_code = R"(
extern "C" __global__
void compare_strings(const char* a, const char* b, int* result, int length) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < length) {
        result[idx] = (a[idx] == b[idx]) ? 1 : 0;
    }
}
)";

const char* matrixMulKernel = R"(
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

void actor_facade_launch_kernel_test(actor_system& sys) {
  caf::cuda::manager& mgr = caf::cuda::manager::get();

  int length = 10;
  std::vector<char> str1(length, 'A');  // example content
  std::vector<char> str2(length, 'A');
  std::vector<int> result(length);
  std::vector<int> len(1, length);

  // 1D grid/block dimensions
  caf::cuda::nd_range dim((length + 255) / 256, 1, 1, 256, 1, 1);

  // Spawn CUDA actor
  auto gpuActor = mgr.spawn(kernel_code, "compare_strings", dim,
                            in<char>{}, in<char>{}, out<int>{}, in<int>{});

  // Spawn an event-based actor to send the message and handle response
  scoped_actor self{sys};

  // Compose message arguments
  auto arg1 = caf::cuda::create_in_arg(str1);
  auto arg2 = caf::cuda::create_in_arg(str2);
  auto arg3 = caf::cuda::create_out_arg(result);
  auto arg4 = caf::cuda::create_in_arg(len);

  sys.spawn([=](event_based_actor* self_actor) {
    self_actor->mail(gpuActor, arg1, arg2, arg3, arg4)
      .request(gpuActor, 10s).then(
        [=](const std::vector<output_buffer>& outputs) {
          self_actor->println("Kernel finished, results:");
          for (size_t i = 0; i < outputs.size(); ++i) {
            std::visit([&](const auto& vec) {
              self_actor->println("Output[{}]: [", i);
              for (const auto& val : vec) {
                self_actor->println("{} ", val);
              }
              self_actor->println("]");
            }, outputs[i].data);
          }
          self_actor->println("");
          self_actor->send_exit(gpuActor, exit_reason::user_shutdown);
          self_actor->quit();
        });
  });

  sys.await_all_actors_done();
}

void test_mmul(caf::actor_system& sys) {
  caf::cuda::manager& mgr = caf::cuda::manager::get();

  int N = 1024;
  int THREADS = 32;
  int BLOCKS = (N + THREADS - 1) / THREADS;

  caf::cuda::nd_range dim(BLOCKS, BLOCKS, 1, THREADS, THREADS, 1);

  auto gpuActor = mgr.spawn(matrixMulKernel, "matrixMul", dim,
                            in<int>{}, in<int>{}, out<int>{}, in<int>{});

  std::vector<int> h_a(N * N);
  std::vector<int> h_b(N * N);
  std::vector<int> h_c(N * N, 0);
  std::vector<int> h_n(1, N);

  std::generate(h_a.begin(), h_a.end(), []() { return rand() % 10; });
  std::generate(h_b.begin(), h_b.end(), []() { return rand() % 10; });

  auto arg1 = caf::cuda::create_in_arg(h_a);
  auto arg2 = caf::cuda::create_in_arg(h_b);
  auto arg3 = caf::cuda::create_out_arg(h_c);
  auto arg4 = caf::cuda::create_in_arg(h_n);

  sys.spawn([=](event_based_actor* self_actor) {
    auto start = std::chrono::high_resolution_clock::now();

    self_actor->mail(gpuActor, arg1, arg2, arg3, arg4)
      .request(gpuActor, 30s).then(
        [=](const std::vector<output_buffer>& outputs) {
          auto end = std::chrono::high_resolution_clock::now();
          std::chrono::duration<double> duration = end - start;

          std::cout << std::fixed << std::setprecision(6);
          std::cout << "[INFO] Kernel round-trip time: " << duration.count() << " seconds\n";

          std::vector<int> result;
          bool got_output = false;

          for (const auto& out : outputs) {
            std::visit([&](const auto& vec) {
        if constexpr (std::is_same_v<std::decay_t<decltype(vec)>, std::vector<int>>) {
                result = vec;
                got_output = true;
              }
            }, out.data);
          }

          if (!got_output) {
            std::cout << "[ERROR] No output data received!\n";
          }

          self_actor->send_exit(gpuActor, exit_reason::user_shutdown);
          self_actor->quit();
        });
  });

  sys.await_all_actors_done();
}

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

void serial_matrix_multiply_test() {
  int N = 1024;
  std::vector<int> h_a(N * N);
  std::vector<int> h_b(N * N);
  std::vector<int> h_c(N * N, 0);

  std::generate(h_a.begin(), h_a.end(), []() { return rand() % 10; });
  std::generate(h_b.begin(), h_b.end(), []() { return rand() % 10; });

  std::cout << "[INFO] Starting serial matrix multiplication...\n";

  auto start = std::chrono::high_resolution_clock::now();

  serial_matrix_multiply(h_a, h_b, h_c, N);

  auto end = std::chrono::high_resolution_clock::now();
  std::chrono::duration<double> duration = end - start;

  std::cout << std::fixed << std::setprecision(6);
  std::cout << "[INFO] Serial matrix multiplication time: "
            << duration.count() << " seconds\n";
}

void test_concurrent_mmul(caf::actor_system& sys) {
  caf::cuda::manager& mgr = caf::cuda::manager::get();

  int N = 1024;
  int THREADS = 32;
  int BLOCKS = (N + THREADS - 1) / THREADS;

  caf::cuda::nd_range dim(BLOCKS, BLOCKS, 1, THREADS, THREADS, 1);

  // Spawn two CUDA actors
  auto gpuActorA = mgr.spawn(matrixMulKernel, "matrixMul", dim,
                             in<int>{}, in<int>{}, out<int>{}, in<int>{});
  auto gpuActorB = mgr.spawn(matrixMulKernel, "matrixMul", dim,
                             in<int>{}, in<int>{}, out<int>{}, in<int>{});

  // Prepare data for actor A
  std::vector<int> h_a1(N * N);
  std::vector<int> h_b1(N * N);
  std::vector<int> h_c1(N * N, 0);
  std::vector<int> h_n1(1, N);

  // Prepare data for actor B
  std::vector<int> h_a2(N * N);
  std::vector<int> h_b2(N * N);
  std::vector<int> h_c2(N * N, 0);
  std::vector<int> h_n2(1, N);

  // Initialize input matrices with random values
  std::generate(h_a1.begin(), h_a1.end(), []() { return rand() % 10; });
  std::generate(h_b1.begin(), h_b1.end(), []() { return rand() % 10; });
  std::generate(h_a2.begin(), h_a2.end(), []() { return rand() % 10; });
  std::generate(h_b2.begin(), h_b2.end(), []() { return rand() % 10; });

  // Create arguments
  auto arg1_a = caf::cuda::create_in_arg(h_a1);
  auto arg2_a = caf::cuda::create_in_arg(h_b1);
  auto arg3_a = caf::cuda::create_out_arg(h_c1);
  auto arg4_a = caf::cuda::create_in_arg(h_n1);

  auto arg1_b = caf::cuda::create_in_arg(h_a2);
  auto arg2_b = caf::cuda::create_in_arg

(h_b2);
  auto arg3_b = caf::cuda::create_out_arg(h_c2);
  auto arg4_b = caf::cuda::create_in_arg(h_n2);

  // Spawn two event-based actors to send messages concurrently
  sys.spawn([=](event_based_actor* self_actor) {
    auto start = std::chrono::high_resolution_clock::now();
    self_actor->mail(gpuActorA, arg1_a, arg2_a, arg3_a, arg4_a)
      .request(gpuActorA, 30s).then(
        [=](const std::vector<output_buffer>& outputs) {
          auto end = std::chrono::high_resolution_clock::now();
          std::chrono::duration<double> duration = end - start;
          std::cout << std::fixed << std::setprecision(6);
          std::cout << "[INFO] Actor A round-trip time: " << duration.count() << " seconds\n";
          self_actor->send_exit(gpuActorA, exit_reason::user_shutdown);
          self_actor->quit();
        });
  });

  sys.spawn([=](event_based_actor* self_actor) {
    auto start = std::chrono::high_resolution_clock::now();
    self_actor->mail(gpuActorB, arg1_b, arg2_b, arg3_b, arg4_b)
      .request(gpuActorB, 30s).then(
        [=](const std::vector<output_buffer>& outputs) {
          auto end = std::chrono::high_resolution_clock::now();
          std::chrono::duration<double> duration = end - start;
          std::cout << std::fixed << std::setprecision(6);
          std::cout << "[INFO] Actor B round-trip time: " << duration.count() << " seconds\n";
          self_actor->send_exit(gpuActorB, exit_reason::user_shutdown);
          self_actor->quit();
        });
  });

  sys.await_all_actors_done();
}

void caf_main(caf::actor_system& sys) {
  caf::cuda::manager::init(sys);
  actor_facade_launch_kernel_test(sys);
  serial_matrix_multiply_test();
  test_mmul(sys);
  test_concurrent_mmul(sys);
}

CAF_MAIN()
