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




void actor_facade_launch_kernel_test(actor_system& sys) {
  std::cout << "[TEST] Starting actor_facade_launch_kernel_test\n";

  caf::cuda::manager& mgr = caf::cuda::manager::get();

  int length = 10;
  std::vector<char> str1(length, 'A');
  std::vector<char> str2(length, 'A');
  std::vector<int> result(length);
  std::vector<int> len(1, length);

  caf::cuda::nd_range dim(10, 1, 1, 1, 1, 1);

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
  std::cout << "[TEST] Starting serial_matrix_multiply_test\n";

  int N = 32000;
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
}

void test_mmul(caf::actor_system& sys,int N) {
  std::cout << "[TEST] Starting test_mmul\n";

  caf::cuda::manager& mgr = caf::cuda::manager::get();

  //int N = 32000;
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

  serial_matrix_multiply(h_a, h_b, h_ref, N);

  auto arg1 = caf::cuda::create_in_arg(h_a);
  auto arg2 = caf::cuda::create_in_arg(h_b);
  auto arg3 = caf::cuda::create_out_arg(h_c);
  auto arg4 = caf::cuda::create_in_arg(N);

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

          bool match = result == h_ref ;
          std::cout << "[INFO] Kernel round-trip time: " << elapsed.count() << " seconds\n";
          std::cout << (match ? "[PASS] GPU result matches reference\n" : "[FAIL] Mismatch in GPU result\n");
          self_actor->send_exit(gpuActor, exit_reason::user_shutdown);
          self_actor->quit();
        });
  });

  sys.await_all_actors_done();
}


void test_mmul_plain(caf::actor_system& sys,int N) {
  std::cout << "[TEST] Starting test_mmul\n";

  caf::cuda::manager& mgr = caf::cuda::manager::get();

  //int N = 32000;
  int THREADS = 32;
  int BLOCKS = (N + THREADS - 1) / THREADS;

  caf::cuda::nd_range dim(BLOCKS, BLOCKS, 1, THREADS, THREADS, 1);

  auto gpuActor = mgr.spawn(matrixMulKernel2, "matrixMul", dim,
                            in<int>{}, in<int>{}, out<int>{}, in<int>{});

  std::vector<int> h_a(N * N);
  std::vector<int> h_b(N * N);
  std::vector<int> h_c(N * N, 0);
  std::vector<int> h_ref(N * N, 0);
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

          bool match = result == h_ref ;
          std::cout << "[INFO] Kernel round-trip time: " << elapsed.count() << " seconds\n";
          //std::cout << (match ? "[PASS] GPU result matches reference\n" : "[FAIL] Mismatch in GPU result\n");
          self_actor->send_exit(gpuActor, exit_reason::user_shutdown);
          self_actor->quit();
        });
  });

  sys.await_all_actors_done();
}

void test_mmul_large(caf::actor_system& sys) {
  std::cout << "[TEST] Starting test_mmul\n";

  caf::cuda::manager& mgr = caf::cuda::manager::get();

  int N = 10000;
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

  serial_matrix_multiply(h_a, h_b, h_ref, N);

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

  sys.await_all_actors_done();
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





void test_mmul_from_ptx(caf::actor_system& sys, int N) {
  std::cout << "[TEST] Starting test_mmul_from_ptx\n";

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

  // Spawn actor from precompiled PTX file
  auto gpuActor = mgr.spawnFromPTX("../mmul.ptx", "matrixMul", dim,
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



void test_mmul_raw_data(caf::actor_system& sys) {
  std::cout << "[TEST] Starting test_mmul_raw_data\n";

  caf::cuda::manager& mgr = caf::cuda::manager::get();

  int N = 1024;
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

  auto gpuActor = mgr.spawn(matrixMulKernel, "matrixMul", dim,
                            in<int>{}, in<int>{}, out<int>{}, in<int>{});

  sys.spawn([=](caf::event_based_actor* self) {
    auto start = std::chrono::high_resolution_clock::now();
    self->mail(gpuActor, h_a, h_b, h_c, h_n).request(gpuActor, 10s).then(
      [=](const std::vector<output_buffer>& outputs) {
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

        bool match = result == h_ref;
        std::cout << "[INFO] Kernel round-trip time: " << elapsed.count() << " seconds\n";
        std::cout << (match ? "[PASS] GPU result matches reference\n"
                            : "[FAIL] Mismatch in GPU result\n");

        self->send_exit(gpuActor, caf::exit_reason::user_shutdown);
        self->quit();
      });
  });

  sys.await_all_actors_done();
}
#include <caf/all.hpp>
#include <caf/cuda/all.hpp>
#include <chrono>
#include <numeric>
#include <vector>
#include <random>
#include <iostream>

using Clock = std::chrono::high_resolution_clock;

struct supervisor_state {
  caf::actor gpu_actor;
  std::vector<int> h_a;
  std::vector<int> h_b;
  std::vector<int> h_c;
  std::vector<int> h_n;

  std::vector<double> kernel_times;
  std::vector<double> full_times;
  int count = 0;
  int id = 0;
  int N = 0;
};

caf::behavior supervisor_fun(caf::stateful_actor<supervisor_state>* self, int id, int N) {
  auto& st = self->state();
  st.id = id;
  st.N = N;

  // Generate matrices once
  st.h_a.resize(st.N * st.N);
  st.h_b.resize(st.N * st.N);
  st.h_c.resize(st.N * st.N, 0);
  st.h_n = {st.N};

  std::generate(st.h_a.begin(), st.h_a.end(), [] { return rand() % 10; });
  std::generate(st.h_b.begin(), st.h_b.end(), [] { return rand() % 10; });

  const int THREADS = 32;
  const int BLOCKS = (N + THREADS - 1) / THREADS;
  caf::cuda::nd_range dims(BLOCKS, 1, 1, THREADS, 1, 1);

  st.gpu_actor = caf::cuda::manager::get().spawn(matrixMulKernel, "matrixMul", dims,
                                                 in<int>{}, in<int>{}, out<int>{}, in<int>{});

  auto run_iteration = [self]() {
    auto& st_ref = self->state();

    auto iteration_start = Clock::now();

    auto arg1 = caf::cuda::create_in_arg(st_ref.h_a);
    auto arg2 = caf::cuda::create_in_arg(st_ref.h_b);
    auto arg3 = caf::cuda::create_out_arg(st_ref.h_c);
    auto arg4 = caf::cuda::create_in_arg(st_ref.h_n);

    auto kernel_start = Clock::now();

    self->mail(st_ref.gpu_actor, arg1, arg2, arg3, arg4)
      .request(st_ref.gpu_actor, std::chrono::seconds(100))
      .then(
        [self, iteration_start, kernel_start](const std::vector<output_buffer>&) {
          auto& st_ref = self->state();
          auto kernel_end = Clock::now();
          auto iteration_end = Clock::now();

          double kernel_time = std::chrono::duration<double>(kernel_end - kernel_start).count();
          double full_time = std::chrono::duration<double>(iteration_end - iteration_start).count();

          std::cout << "[INFO] Supervisor " << st_ref.id
                    << " Iteration " << st_ref.count
                    << " Kernel round-trip: " << kernel_time << " s, "
                    << "Full iteration time: " << full_time << " s\n";

          st_ref.kernel_times.push_back(kernel_time);
          st_ref.full_times.push_back(full_time);
          ++st_ref.count;

          if (st_ref.count < 20) {
            self->delayed_send(self, std::chrono::milliseconds(0), std::string("start"));
          } else {
            double kernel_avg = std::accumulate(st_ref.kernel_times.begin(), st_ref.kernel_times.end(), 0.0) / st_ref.kernel_times.size();
            double full_avg = std::accumulate(st_ref.full_times.begin(), st_ref.full_times.end(), 0.0) / st_ref.full_times.size();

            std::cout << "[INFO] Supervisor " << st_ref.id
                      << " Kernel average: " << kernel_avg << " s, "
                      << "Full iteration average: " << full_avg << " s\n";

            self->send_exit(st_ref.gpu_actor, caf::exit_reason::user_shutdown);
            self->quit();
          }
        },
        [self](caf::error& err) {
          std::cerr << "[ERROR] Kernel execution failed: " << caf::to_string(err) << std::endl;
          self->quit(err);
        });
  };

  return {
    [=](const std::string& msg) {
      if (msg == "start") {
        run_iteration();
      }
    }
  };
}


// Driver function
inline void run_concurrent_mmul_test(caf::actor_system& sys,
                                     int num_supervisors,
                                     int matrix_size) {
  auto start = Clock::now();

  for (int i = 0; i < num_supervisors; ++i) {
    auto sup = sys.spawn(supervisor_fun, i, matrix_size);
    caf::anon_send(sup, std::string("start"));
  }

  sys.await_all_actors_done();

  auto end = Clock::now();
  std::chrono::duration<double> duration = end - start;
  std::cout << "[TIMER] run_concurrent_mmul_test took: "
            << duration.count() << " seconds\n";
}





caf::behavior supervisor_fun_validate(caf::stateful_actor<supervisor_state>* self, int id, int N) {
  auto& st = self->state();
  st.id = id;
  st.N = N;

  st.h_a.resize(st.N * st.N);
  st.h_b.resize(st.N * st.N);
  st.h_c.resize(st.N * st.N, 0);
  st.h_n = {st.N};


  const int THREADS = 32;
  const int BLOCKS = (N + THREADS - 1) / THREADS;
  caf::cuda::nd_range dims(BLOCKS, BLOCKS, 1, THREADS, THREADS, 1);

  st.gpu_actor = caf::cuda::manager::get().spawnFromCUBIN("../mmul.cubin", "matrixMul", dims,
                                                 in<int>{}, in<int>{}, out<int>{}, in<int>{});

  auto run_iteration = [&,self]() {
    auto& st_ref = self->state();

    auto iteration_start = Clock::now();

    //generate new matrix 
    std::generate(st.h_a.begin(), st.h_a.end(), [] { return rand() % 10; });
    std::generate(st.h_b.begin(), st.h_b.end(), [] { return rand() % 10; });
    auto arg1 = caf::cuda::create_in_arg(st.h_a);
    auto arg2 = caf::cuda::create_in_arg(st.h_b);
    auto arg3 = caf::cuda::create_out_arg(st.h_c);
    auto arg4 = caf::cuda::create_in_arg(st.N); //i have no idea why this needs to be squared but it does

    auto kernel_start = Clock::now();

    self->mail(st_ref.gpu_actor, arg1, arg2, arg3, arg4)
      .request(st_ref.gpu_actor, std::chrono::seconds(100))
      .then(
        [&,self, iteration_start, kernel_start](const std::vector<output_buffer>& outputs) {
          auto& st_ref = self->state();
          auto kernel_end = Clock::now();
          auto iteration_end = Clock::now();

          double kernel_time = std::chrono::duration<double>(kernel_end - kernel_start).count();
          double full_time = std::chrono::duration<double>(iteration_end - iteration_start).count();

          std::cout << "[INFO] Supervisor " << st_ref.id
                    << " Iteration " << st_ref.count
                    << " Kernel round-trip: " << kernel_time << " s, "
                    << "Full iteration time: " << full_time << " s\n";

          st_ref.kernel_times.push_back(kernel_time);
          st_ref.full_times.push_back(full_time);
          ++st_ref.count;
        
	  std::vector<int> result;
        
	  for (const auto& out : outputs) {
          std::visit([&](const auto& vec) {
            using T = std::decay_t<decltype(vec)>;
            if constexpr (std::is_same_v<T, std::vector<int>>) {
              result = vec;
            }
          }, out.data);
        }

std::vector<int> h_ref(st.N * st.N, 0);
serial_matrix_multiply(st.h_a, st.h_b, h_ref, st.N);
bool match = result == h_ref;

if (match) {
  std::cout << "[PASS] GPU result matches reference\n";
} else {
  std::cout << "[FAIL] Mismatch in GPU result\n";

  std::cout << "Expected (h_ref): ";
  for (int val : h_ref)
    std::cout << val << ' ';
  std::cout << '\n';

  std::cout << "Actual (result):  ";
  for (int val : result)
    std::cout << val << ' ';
  std::cout << '\n';
}

          if (st_ref.count < 20) {
            self->delayed_send(self, std::chrono::milliseconds(0), std::string("start"));
          } else {
            double kernel_avg = std::accumulate(st_ref.kernel_times.begin(), st_ref.kernel_times.end(), 0.0) / st_ref.kernel_times.size();
            double full_avg = std::accumulate(st_ref.full_times.begin(), st_ref.full_times.end(), 0.0) / st_ref.full_times.size();

            std::cout << "[INFO] Supervisor " << st_ref.id
                      << " Kernel average: " << kernel_avg << " s, "
                      << "Full iteration average: " << full_avg << " s\n";

            self->send_exit(st_ref.gpu_actor, caf::exit_reason::user_shutdown);
            self->quit();
          }
        },
        [self](caf::error& err) {
          std::cerr << "[ERROR] Kernel execution failed: " << caf::to_string(err) << std::endl;
          self->quit(err);
        });
  };

  return {
    [=](const std::string& msg) {
      if (msg == "start") {
        run_iteration();
      }
    }
  };
}




// Driver function
inline void run_concurrent_mmul_validate_test(caf::actor_system& sys,
                                     int num_supervisors,
                                     int matrix_size) {
  auto start = Clock::now();

  for (int i = 0; i < num_supervisors; ++i) {
    auto sup = sys.spawn(supervisor_fun_validate, i, matrix_size);
    caf::anon_send(sup, std::string("start"));
  }

  sys.await_all_actors_done();

  auto end = Clock::now();
  std::chrono::duration<double> duration = end - start;
  std::cout << "[TIMER] run_concurrent_mmul_test took: "
            << duration.count() << " seconds\n";
}



// === Global matrices for shared use ===
std::vector<int> global_a;
std::vector<int> global_b;
std::vector<std::vector<int>> global_cs;
std::vector<int> global_c; //yes each actor gets the same output buffer
			   //this shouldnt matter anyways due to just performance testing
			   //and gpu actors dont share state

// === New Supervisor (uses global matrix data) ===
caf::behavior supervisor_global_fun(caf::stateful_actor<supervisor_state>* self, int id, int N) {
  auto& st = self->state();
  st.id = id;
  st.N = N;

  const int THREADS = 32;
  const int BLOCKS = (N + THREADS - 1) / THREADS;
  caf::cuda::nd_range dims(BLOCKS, 1, 1, THREADS, 1, 1);

  st.gpu_actor = caf::cuda::manager::get().spawnFromCUBIN("../mmul.cubin", "matrixMul", dims,
                                                 in<int>{}, in<int>{}, out<int>{}, in<int>{});

  auto run_iteration = [self]() {
    auto& st_ref = self->state();
    int N_val = st_ref.N;

    auto iteration_start = Clock::now();

    auto arg1 = caf::cuda::create_in_arg(global_a);
    auto arg2 = caf::cuda::create_in_arg(global_b);
    auto arg3 = caf::cuda::create_out_arg(global_c);
    auto arg4 = caf::cuda::create_in_arg(N_val); // local N

    auto kernel_start = Clock::now();

    self->mail(st_ref.gpu_actor, arg1, arg2, arg3, arg4)
      .request(st_ref.gpu_actor, std::chrono::seconds(1000))
      .then(
        [self, iteration_start, kernel_start](const std::vector<output_buffer>&) {
          auto& st_ref = self->state();
          auto kernel_end = Clock::now();
          auto iteration_end = Clock::now();

          double kernel_time = std::chrono::duration<double>(kernel_end - kernel_start).count();
          double full_time = std::chrono::duration<double>(iteration_end - iteration_start).count();

          std::cout << "[INFO] [GPU GLOBAL] Supervisor " << st_ref.id
                    << " Iteration " << st_ref.count
                    << " Kernel round-trip: " << kernel_time << " s, "
                    << "Full iteration time: " << full_time << " s\n";

          st_ref.kernel_times.push_back(kernel_time);
          st_ref.full_times.push_back(full_time);
          ++st_ref.count;

          if (st_ref.count < 20) {
            self->delayed_send(self, std::chrono::milliseconds(0), std::string("start"));
          } else {
            double kernel_avg = std::accumulate(st_ref.kernel_times.begin(), st_ref.kernel_times.end(), 0.0) / st_ref.kernel_times.size();
            double full_avg = std::accumulate(st_ref.full_times.begin(), st_ref.full_times.end(), 0.0) / st_ref.full_times.size();

            std::cout << "[INFO] [GPU GLOBAL] Supervisor " << st_ref.id
                      << " Kernel average: " << kernel_avg << " s, "
                      << "Full iteration average: " << full_avg << " s\n";

            self->send_exit(st_ref.gpu_actor, caf::exit_reason::user_shutdown);
            self->quit();
          }
        },
        [self](caf::error& err) {
          std::cerr << "[ERROR] [GPU GLOBAL] Kernel execution failed: " << caf::to_string(err) << std::endl;
          self->quit(err);
        });
  };

  return {
    [=](const std::string& msg) {
      if (msg == "start") {
        run_iteration();
      }
    }
  };
}

// === New Test Function ===
inline void run_concurrent_mmul_test_global(caf::actor_system& sys,
                                            int num_supervisors,
                                            int matrix_size) {
  auto start = Clock::now();

  int N = matrix_size;
  size_t matrix_elements = static_cast<size_t>(N) * N;

  // Global inputs
  global_a.assign(matrix_elements, 0);
  global_b.assign(matrix_elements, 0);
  global_c.assign(matrix_elements,0);
  //global_cs.resize(num_supervisors);
  //for (int i = 0; i < num_supervisors; ++i)
    //global_cs[i].assign(matrix_elements, 0);

  // Optional: Populate input with actual data
  //std::generate(global_a.begin(), global_a.end(), [] { return rand() % 10; });
  //std::generate(global_b.begin(), global_b.end(), [] { return rand() % 10; });

  // Spawn supervisors
  for (int i = 0; i < num_supervisors; ++i) {
    auto sup = sys.spawn(supervisor_global_fun, i, N);
    caf::anon_send(sup, std::string("start"));
  }

  sys.await_all_actors_done();

  auto end = Clock::now();
  std::chrono::duration<double> duration = end - start;
  std::cout << "[TIMER] run_concurrent_mmul_test_global took: "
            << duration.count() << " seconds\n";
}


// === Global matrices for CPU serial multiply ===
std::vector<int> cpu_global_a;
std::vector<int> cpu_global_b;
std::vector<int> cpu_global_c; // Shared output buffer

// === Messages ===
using matrix_msg = caf::message; // We'll send references via vector<int> const&

// === Worker Actor: Does the serial multiply on request ===
caf::behavior cpu_worker_fun(caf::event_based_actor* self) {
  return {
    [=](const std::vector<int>& a, const std::vector<int>& b, std::vector<int>& c, int N) {
      serial_matrix_multiply(a, b, c, N);
      // Reply with empty message or some confirmation (could send duration, etc.)
      //self->send(self->current_sender());
    }
  };
}

// === Supervisor State ===
struct cpu_supervisor_state {
  caf::actor worker;
  int id = 0;
  int N = 0;
  int count = 0;
  std::vector<double> run_times;
};

// === Supervisor Actor ===
caf::behavior cpu_supervisor_global_fun(caf::stateful_actor<cpu_supervisor_state>* self, int id, int N) {
  auto& st = self->state();
  st.id = id;
  st.N = N;
  st.count = 0;

  // Spawn worker actor once
  st.worker = self->spawn(cpu_worker_fun);

  auto run_iteration = [self]() {
    auto& st_ref = self->state();
    auto start = Clock::now();

    // Send global buffers to worker actor for multiplication
    self->request(st_ref.worker, std::chrono::seconds(1000),
                  cpu_global_a, cpu_global_b, cpu_global_c, st_ref.N)
      .then(
        [self, start]() {
          auto end = Clock::now();
          double duration = std::chrono::duration<double>(end - start).count();

          auto& st_ref = self->state();
          std::cout << "[INFO] [CPU GLOBAL] Supervisor " << st_ref.id
                    << " Iteration " << st_ref.count
                    << " Serial multiply time: " << duration << " s\n";

          st_ref.run_times.push_back(duration);
          ++st_ref.count;

          if (st_ref.count < 20) {
            self->delayed_send(self, std::chrono::milliseconds(0), std::string("start"));
          } else {
            double avg = std::accumulate(st_ref.run_times.begin(), st_ref.run_times.end(), 0.0) / st_ref.run_times.size();
            std::cout << "[INFO] [CPU GLOBAL] Supervisor " << st_ref.id
                      << " Average serial multiply time: " << avg << " s\n";

            self->send_exit(st_ref.worker, caf::exit_reason::user_shutdown);
            self->quit();
          }
        },
        [self](caf::error& err) {
          std::cerr << "[ERROR] [CPU GLOBAL] Worker call failed: " << caf::to_string(err) << std::endl;
          self->quit(err);
        });
  };

  return {
    [=](const std::string& msg) {
      if (msg == "start") {
        run_iteration();
      }
    }
  };
}

// === CPU Global Matrix Test Function with Worker ===
inline void run_concurrent_serial_mmul_test_global_with_worker(caf::actor_system& sys,
                                                               int num_supervisors,
                                                               int matrix_size) {
  auto start = Clock::now();

  int N = matrix_size;
  size_t matrix_elements = static_cast<size_t>(N) * N;

  // Initialize global matrices once
  cpu_global_a.assign(matrix_elements, 0);
  cpu_global_b.assign(matrix_elements, 0);
  cpu_global_c.assign(matrix_elements, 0);

  // Optional: fill inputs with some data
  std::generate(cpu_global_a.begin(), cpu_global_a.end(), [] { return rand() % 10; });
  std::generate(cpu_global_b.begin(), cpu_global_b.end(), [] { return rand() % 10; });

  // Spawn supervisors (which spawn workers internally)
  for (int i = 0; i < num_supervisors; ++i) {
    auto sup = sys.spawn(cpu_supervisor_global_fun, i, N);
    caf::anon_send(sup, std::string("start"));
  }

  sys.await_all_actors_done();

  auto end = Clock::now();
  std::chrono::duration<double> duration = end - start;
  std::cout << "[TIMER] run_concurrent_serial_mmul_test_global_with_worker took: "
            << duration.count() << " seconds\n";
}




void run_all_concurrent_tests(caf::actor_system& sys) {
  std::vector<int> matrix_sizes = {1024, 2048, 4096};
  std::vector<int> actor_counts = {1, 50, 200};

  for (auto N : matrix_sizes) {
    for (auto num_actors : actor_counts) {
      std::cout << "\n=== Running tests for N = " << N
                << ", num_actors = " << num_actors << " ===\n";

      std::cout << "[RUN] GPU concurrent test (global matrices)...\n";
      run_concurrent_mmul_test_global(sys, num_actors, N);

      std::cout << "[RUN] CPU concurrent test with worker (global matrices)...\n";
      run_concurrent_serial_mmul_test_global_with_worker(sys, num_actors, N);
    }
  }
}


void caf_main(caf::actor_system& sys) {
  caf::cuda::manager::init(sys);
  //test_main(sys);
  //actor_facade_launch_kernel_test(sys);
   //test_mmul(sys,1024);
   //test_mmul_from_ptx(sys,1024);
   //test_mmul_from_cubin(sys,1024);
   //test_mmul_plain(sys,1024);
  //test_mmul_large(sys);
  //run_concurrent_mmul_test(sys,200,1024);
  //run_concurrent_mmul_test_global(sys,500,1024);
 //run_concurrent_serial_mmul_test_global_with_worker(sys,2,1024);
  run_concurrent_mmul_validate_test(sys,100,60);
 //run_all_concurrent_tests(sys);

}




CAF_MAIN()
