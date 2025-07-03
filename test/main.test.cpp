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

void test_concurrent_supervisor_mmul(actor_system& sys) {
  std::cout << "[TEST] Starting test_concurrent_supervisor_mmul\n";

  // Supervisor actor behavior
  auto supervisor_actor = [](event_based_actor* self, int id) {
    caf::cuda::manager& mgr = caf::cuda::manager::get();
    const int N = 1024;
    const int THREADS = 32;
    const int BLOCKS = (N + THREADS - 1) / THREADS;
    caf::cuda::nd_range dim(BLOCKS, BLOCKS, 1, THREADS, THREADS, 1);

    // Spawn a unique GPU actor for this supervisor
    auto gpuActor = mgr.spawn(matrixMulKernel, "matrixMul", dim,
                              in<int>{}, in<int>{}, out<int>{}, in<int>{});

    int count = 0;
    std::vector<double> times;

    // Function to start or continue an iteration
    auto start_iteration = [self, gpuActor, &count, &times, id, N]() mutable {
     //std::cout << "times is " <<  N << "\n"; 
     if (count < 20) {
        // Generate random 1024x1024 matrices
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

        auto start = std::chrono::high_resolution_clock::now();

	std::cout << "Sending message to gpu actor\n";
        // Send message to GPU actor and wait for response
        self->mail(gpuActor, arg1, arg2, arg3, arg4)
          .request(gpuActor, 10s)
          .then([self, start, gpuActor, &count, &times, id](const std::vector<output_buffer>&) mutable {
            auto end = std::chrono::high_resolution_clock::now();
            std::chrono::duration<double> elapsed = end - start;
            std::cout << "[INFO] Supervisor " << id << " Iteration " << count
                      << " round-trip time: " << elapsed.count() << " seconds\n";
            times.push_back(elapsed.count());
            count++;

            if (count < 20) {
              // Trigger the next iteration
              self->send(self, std::string("start"));
            } else {
              // All iterations complete, report average time
              double avg_time = std::accumulate(times.begin(), times.end(), 0.0) / times.size();
              std::cout << "[INFO] Supervisor " << id << " Average round-trip time: "
                        << avg_time << " seconds\n";
              self->send_exit(gpuActor, exit_reason::user_shutdown);
              self->quit();
            }
          });
      }
    };

    // Define actor behavior to handle the "start" message
    self->become([start_iteration](const std::string& msg) mutable {
      if (msg == "start") {
        start_iteration();
      }
    });

    // Kick off the first iteration
    self->send(self, std::string("start"));
  };

  // Spawn 5 supervisor actors to test concurrency
  for (int i = 0; i < 1; ++i) {
    sys.spawn(supervisor_actor, i);
  }

  sys.await_all_actors_done();
}


#include <caf/all.hpp>
//#include <caf/io/all.hpp>
#include <caf/cuda/all.hpp>
#include <chrono>
#include <numeric>
#include <vector>
#include <random>

using Clock = std::chrono::high_resolution_clock;
struct supervisor_state {
  caf::actor gpu_actor;
  std::vector<double> times;
  int count = 0;   // track iteration count
  int id = 0;      // supervisor id
  int N = 0;       // matrix size
};
;
caf::behavior supervisor_fun(caf::stateful_actor<supervisor_state>* self, int id, int N) {
  auto& st = self->state();
  st.id = id;
  st.N = N;

  const int THREADS = 32;
  const int BLOCKS = (N + THREADS - 1) / THREADS;
  caf::cuda::nd_range dims(BLOCKS, BLOCKS, 1, THREADS, THREADS, 1);

  st.gpu_actor = caf::cuda::manager::get().spawn(matrixMulKernel, "matrixMul", dims,
                                                 in<int>{}, in<int>{}, out<int>{}, in<int>{});

  auto run_iteration = [self]() {
    auto& st_ref = self->state();

    std::vector<int> h_a(st_ref.N * st_ref.N);
    std::vector<int> h_b(st_ref.N * st_ref.N);
    std::vector<int> h_c(st_ref.N * st_ref.N, 0);
    std::vector<int> h_n(1, st_ref.N);

    std::generate(h_a.begin(), h_a.end(), [] { return rand() % 10; });
    std::generate(h_b.begin(), h_b.end(), [] { return rand() % 10; });

    auto arg1 = caf::cuda::create_in_arg(h_a);
    auto arg2 = caf::cuda::create_in_arg(h_b);
    auto arg3 = caf::cuda::create_out_arg(h_c);
    auto arg4 = caf::cuda::create_in_arg(h_n);

    auto start_time = std::chrono::high_resolution_clock::now();

    // Here is the key part: mail().request() chain
    self->mail(st_ref.gpu_actor, arg1, arg2, arg3, arg4)
      .request(st_ref.gpu_actor, std::chrono::seconds(10))
      .then(
        [self, start_time](const std::vector<output_buffer>&) {
          auto& st_ref = self->state();
          auto end_time = std::chrono::high_resolution_clock::now();
          double elapsed = std::chrono::duration<double>(end_time - start_time).count();

          std::cout << "[INFO] Supervisor " << st_ref.id
                    << " Iteration " << st_ref.count
                    << " Round-trip: " << elapsed << " s\n";

          st_ref.times.push_back(elapsed);
          ++st_ref.count;

          if (st_ref.count < 20) {
            self->mail(std::string("start")).send(self);  // trigger next iteration
          } else {
            double avg = std::accumulate(st_ref.times.begin(), st_ref.times.end(), 0.0) / st_ref.times.size();
            std::cout << "[INFO] Supervisor " << st_ref.id
                      << " Average time: " << avg << " s\n";

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
  for (int i = 0; i < num_supervisors; ++i) {
    auto sup = sys.spawn(supervisor_fun, i, matrix_size);
    caf::anon_send(sup, std::string("start"));
  }
  sys.await_all_actors_done();
}





void caf_main(caf::actor_system& sys) {
  caf::cuda::manager::init(sys);
  //actor_facade_launch_kernel_test(sys);
  //test_mmul(sys);
  //test_mmul_raw_data(sys);
  //test_concurrent_mmul(sys);
  //serial_matrix_multiply_test();
  //test_concurrent_supervisor_mmul(sys);
  run_concurrent_mmul_test(sys,10,1024);
}

CAF_MAIN()
