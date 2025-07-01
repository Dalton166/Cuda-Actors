#include "main.test.hpp"

#include <caf/all.hpp>
#include <caf/cuda/all.hpp>
#include <iostream>
#include <iomanip>
#include <vector>
#include <chrono>
#include <thread>
#include <algorithm>

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
  std::cout << "[TEST] Starting actor_facade_launch_kernel_test\n";

  caf::cuda::manager& mgr = caf::cuda::manager::get();

  int length = 10;
  std::vector<char> str1(length, 'A');
  std::vector<char> str2(length, 'A');
  std::vector<int> result(length);
  std::vector<int> len(1, length);

  caf::cuda::nd_range dim((length + 255) / 256, 1, 1, 256, 1, 1);

  auto gpuActor = mgr.spawn(kernel_code, "compare_strings", dim,
                            in<char>{}, in<char>{}, out<int>{}, in<int>{});

  auto arg1 = caf::cuda::create_in_arg(str1);
  auto arg2 = caf::cuda::create_in_arg(str2);
  auto arg3 = caf::cuda::create_out_arg(result);
  auto arg4 = caf::cuda::create_in_arg(len);

  sys.spawn([=](event_based_actor* self_actor) {
    auto start = std::chrono::high_resolution_clock::now();
    self_actor->mail(gpuActor, arg1, arg2, arg3, arg4)
      .request(gpuActor, 10s).then(
        [=](const std::vector<output_buffer>& outputs) {
          auto end = std::chrono::high_resolution_clock::now();
          std::chrono::duration<double> elapsed = end - start;
          std::cout << "[INFO] Kernel round-trip time: " << elapsed.count() << " seconds\n";

          for (size_t i = 0; i < outputs.size(); ++i) {
            std::visit([&](const auto& vec) {
              std::cout << "Output[" << i << "]: ";
              for (const auto& val : vec) {
                std::cout << val << " ";
              }
              std::cout << "\n";
            }, outputs[i].data);
          }

          self_actor->send_exit(gpuActor, exit_reason::user_shutdown);
          self_actor->quit();
        });
  });

  std::this_thread::sleep_for(5s);
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
  std::cout << "[TEST] Starting serial_matrix_multiply_test\n";

  int N = 1024;
  std::vector<int> h_a(N * N);
  std::vector<int> h_b(N * N);
  std::vector<int> h_c(N * N, 0);

  std::generate(h_a.begin(), h_a.end(), []() { return rand() % 10; });
  std::generate(h_b.begin(), h_b.end(), []() { return rand() % 10; });

  auto start = std::chrono::high_resolution_clock::now();

  serial_matrix_multiply(h_a, h_b, h_c, N);

  auto end = std::chrono::high_resolution_clock::now();
  std::chrono::duration<double> duration = end - start;

  std::cout << std::fixed << std::setprecision(6);
  std::cout << "[INFO] Serial matrix multiplication time: "
            << duration.count() << " seconds\n";

  std::this_thread::sleep_for(5s);
}

void test_mmul(caf::actor_system& sys) {
  std::cout << "[TEST] Starting test_mmul\n";

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
  std::vector<int> h_ref(N * N, 0);
  std::vector<int> h_n(1, N);

  std::generate(h_a.begin(), h_a.end(), []() { return rand() % 10; });
  std::generate(h_b.begin(), h_b.end(), []() { return rand() % 10; });

  //serial_matrix_multiply(h_a, h_b, h_ref, N);

  auto arg1 = caf::cuda::create_in_arg(h_a);
  auto arg2 = caf::cuda::create_in_arg(h_b);
  auto arg3 = caf::cuda::create_out_arg(h_c);
  auto arg4 = caf::cuda::create_in_arg(h_n);

  sys.spawn([=](event_based_actor* self_actor) {
    auto start = std::chrono::high_resolution_clock::now();
    self_actor->mail(gpuActor, arg1, arg2, arg3, arg4)
      .request(gpuActor, 10s).then(
        [=](const std::vector<output_buffer>& outputs) {
          auto end = std::chrono::high_resolution_clock::now();
          std::chrono::duration<double> elapsed = end - start;
          std::vector<int> result;
          for (const auto& out : outputs) {
            std::visit([&](const auto& vec) {
              if constexpr (std::is_same_v<std::decay_t<decltype(vec)>, std::vector<int>>) {
                result = vec;
              }
            }, out.data);
          }

          bool match = result == h_ref;
          std::cout << "[INFO] Kernel round-trip time: " << elapsed.count() << " seconds\n";
          std::cout << (match ? "[PASS] GPU result matches reference\n" : "[FAIL] Mismatch in GPU result\n");
          self_actor->send_exit(gpuActor, exit_reason::user_shutdown);
          self_actor->quit();
        });
  });

  //std::this_thread::sleep_for(5s);
  //caf::cuda::manager::shutdown();
  sys.await_all_actors_done();
}

void test_concurrent_mmul(caf::actor_system& sys) {
  std::cout << "[TEST] Starting test_concurrent_mmul\n";

  caf::cuda::manager& mgr = caf::cuda::manager::get();

  int N = 1024;
  int THREADS = 32;
  int BLOCKS = (N + THREADS - 1) / THREADS;

  caf::cuda::nd_range dim(BLOCKS, BLOCKS, 1, THREADS, THREADS, 1);

  auto gpuActorA = mgr.spawn(matrixMulKernel, "matrixMul", dim,
                             in<int>{}, in<int>{}, out<int>{}, in<int>{});
  auto gpuActorB = mgr.spawn(matrixMulKernel, "matrixMul", dim,
                             in<int>{}, in<int>{}, out<int>{}, in<int>{});

  std::vector<int> h_a1(N * N);
  std::vector<int> h_b1(N * N);
  std::vector<int> h_c1(N * N, 0);
  std::vector<int> h_ref1(N * N, 0);
  std::vector<int> h_n1(1, N);

  std::vector<int> h_a2(N * N);
  std::vector<int> h_b2(N * N);
  std::vector<int> h_c2(N * N, 0);
  std::vector<int> h_ref2(N * N, 0);
  std::vector<int> h_n2(1, N);

  std::generate(h_a1.begin(), h_a1.end(), []() { return rand() % 10; });
  std::generate(h_b1.begin(), h_b1.end(), []() { return rand() % 10; });
  std::generate(h_a2.begin(), h_a2.end(), []() { return rand() % 10; });
  std::generate(h_b2.begin(), h_b2.end(), []() { return rand() % 10; });

  serial_matrix_multiply(h_a1, h_b1, h_ref1, N);
  serial_matrix_multiply(h_a2, h_b2, h_ref2, N);

  auto arg1a = caf::cuda::create_in_arg(h_a1);
  auto arg2a = caf::cuda::create_in_arg(h_b1);
  auto arg3a = caf::cuda::create_out_arg(h_c1);
  auto arg4a = caf::cuda::create_in_arg(h_n1);

  auto arg1b = caf::cuda::create_in_arg(h_a2);
  auto arg2b = caf::cuda::create_in_arg(h_b2);
  auto arg3b = caf::cuda::create_out_arg(h_c2);
  auto arg4b = caf::cuda::create_in_arg(h_n2);

  sys.spawn([=](event_based_actor* self_actor) {
    auto start = std::chrono::high_resolution_clock::now();
    self_actor->mail(gpuActorA, arg1a, arg2a, arg3a, arg4a)
      .request(gpuActorA, 10s).then(
        [=](const std::vector<output_buffer>& outputs) {
          auto end = std::chrono::high_resolution_clock::now();
          std::chrono::duration<double> elapsed = end - start;
          std::vector<int> result;
          for (const auto& out : outputs) {
            std::visit([&](const auto& vec) {
              if constexpr (std::is_same_v<std::decay_t<decltype(vec)>, std::vector<int>>) {
                result = vec;
              }
            }, out.data);
          }
          bool match = result == h_ref1;
          std::cout << "[INFO] Actor A round-trip time: " << elapsed.count() << " seconds\n";
          std::cout << (match ? "[PASS] Actor A result OK\n" : "[FAIL] Actor A result mismatch\n");
          self_actor->send_exit(gpuActorA, exit_reason::user_shutdown);
          self_actor->quit();
        });
  });

  sys.spawn([=](event_based_actor* self_actor) {
    auto start = std::chrono::high_resolution_clock::now();
    self_actor->mail(gpuActorB, arg1b, arg2b, arg3b, arg4b)
      .request(gpuActorB, 10s).then(
        [=](const std::vector<output_buffer>& outputs) {
          auto end = std::chrono::high_resolution_clock::now();
          std::chrono::duration<double> elapsed = end - start;
          std::vector<int> result;
          for (const auto& out : outputs) {
            std::visit([&](const auto& vec) {
              if constexpr (std::is_same_v<std::decay_t<decltype(vec)>, std::vector<int>>) {
                result = vec;
              }
            }, out.data);
          }
          bool match = result == h_ref2;
          std::cout << "[INFO] Actor B round-trip time: " << elapsed.count() << " seconds\n";
          std::cout << (match ? "[PASS] Actor B result OK\n" : "[FAIL] Actor B result mismatch\n");
          self_actor->send_exit(gpuActorB, exit_reason::user_shutdown);
          self_actor->quit();
        });
  });

  //std::this_thread::sleep_for(5s);
  sys.await_all_actors_done();
}

void caf_main(caf::actor_system& sys) {
  caf::cuda::manager::init(sys);
  //actor_facade_launch_kernel_test(sys);
  //serial_matrix_multiply_test();
  test_mmul(sys);
  test_concurrent_mmul(sys);
}

CAF_MAIN()

