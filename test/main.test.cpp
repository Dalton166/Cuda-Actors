

/*
 * this file does absolutely nothing right now 
 * should however test the actor facade spawn function 
 */

#include "main.test.hpp"

const char* kernel_code = R"(
extern "C" __global__
void compare_strings(const char* a, const char* b, int* result, int length) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < length) {
        result[idx] = (a[idx] == b[idx]) ? 1 : 0;
    }
}
)";

#include <caf/all.hpp>
#include <caf/cuda/all.hpp>
#include <iostream>
#include <vector>
#include <chrono>

using namespace caf;
using namespace std::chrono_literals;
void actor_facade_launch_kernel_test(actor_system& sys) {
  caf::cuda::manager& mgr = caf::cuda::manager::get();

  int length = 10;
  std::vector<char> str1(length, 'A');  // example content
  std::vector<char> str2(length, 'A');
  std::vector<int> result(length);
  std::vector<int> len(1, length);

  // 1D grid/block dimensions
  caf::cuda::nd_range dim(1, 1, 1, 1, 1, 1);

  // Spawn CUDA actor
  auto gpuActor = mgr.spawn(kernel_code, "compare_strings", dim,
                            in<char>{}, in<char>{}, out<int>{}, in<int>{});

  // Spawn an event_based_actor to send the message and handle response
  scoped_actor self{sys};

  // Compose message arguments
  auto arg1 = caf::cuda::create_in_arg(str1);
  auto arg2 = caf::cuda::create_in_arg(str2);
  auto arg3 = caf::cuda::create_out_arg(result);
  auto arg4 = caf::cuda::create_in_arg(len);

  // Send message and handle response
  sys.spawn([=](event_based_actor* self_actor) {
    self_actor->mail(gpuActor, arg1, arg2, arg3, arg4)
      .request(gpuActor, 10s).then(
        [self_actor](const std::vector<output_buffer>& outputs) {
          aout(self_actor) << "Kernel finished, results: ";
          for (size_t i = 0; i < outputs.size(); ++i) {
  std::visit([&](const auto& vec) {
    aout(self_actor) << "Output[" << i << "]: [ ";
    for (const auto& val : vec) {
      aout(self_actor) << val << " ";
    }
    aout(self_actor) << "]\n";
  }, outputs[i].data);
}

          aout(self_actor) << std::endl;
          self_actor->quit();
        }
      );
  });

  // Wait for all actors to finish
  //sys.await_all_actors_done();
  std::this_thread::sleep_for(2s);
}


// ---------------------------------- Matrix multiplication test ----------------


const char* matrixMulKernel = R"(
extern "C" __global__
void matrixMul(const int* a, const int* b, int* c, int *N_val) {
    
    int N = *N_val;
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    //printf("N = %d\n", N);
    if (row < N && col < N) {
        int temp = 0;
        for (int k = 0; k < N; ++k) {
            temp += a[row * N + k] * b[k * N + col];
        }
        c[row * N + col] = temp;
    }
}
)";



// Check result on the CPU
void verify_result(std::vector<int> &a, std::vector<int> &b, std::vector<int> &c, int N) {
  // For every row...
  for (int i = 0; i < N; i++) {
    // For every column...
    for (int j = 0; j < N; j++) {
      // For every element in the row-column pair
      int tmp = 0;
      for (int k = 0; k < N; k++) {
        // Accumulate the partial results
        tmp += a[i * N + k] * b[k * N + j];
      }

      // Check against the CPU result
      assert(tmp == c[i * N + j]);
    }
  }
}
#include <iostream>
#include <iomanip> // for std::setprecision
#include <chrono>

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

  sys.spawn([=](caf::event_based_actor* self_actor) {
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
          } else {
            std::cout << "[INFO] Verifying result...\n";
            std::vector<int> expected(N * N);
            for (int i = 0; i < N; ++i) {
              for (int j = 0; j < N; ++j) {
                int tmp = 0;
                for (int k = 0; k < N; ++k) {
                  tmp += h_a[i * N + k] * h_b[k * N + j];
                }
                expected[i * N + j] = tmp;
              }
            }

            if (std::equal(result.begin(), result.end(), expected.begin())) {
              std::cout << "[SUCCESS] Matrix multiplication result verified successfully.\n";
            } else {
              std::cout << "[FAILURE] Mismatch found in matrix multiplication results.\n";
            }
          }

          self_actor->quit();
        });
  });

  std::this_thread::sleep_for(std::chrono::seconds(5));
}


//serial mmul for reference


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



/*
void actor_facade_spawn_test(caf::actor_system& sys) {

 caf::cuda::manager& mgr = caf::cuda::manager::get();
    int length = 10;
    std::vector<char> str1(length);
    std::vector<char> str2(length);
    std::vector<int> result(length);
    std::vector<int> len;
    len[0] = length;

    //1 dimension for blocks and grids
    caf::cuda::nd_range dim(1,1,1,1,1,1);

    // Pass std::move for vectors you want to forward as rvalues
    auto gpuActor = mgr.spawn(
        kernel_code,
        "compare_strings",
	dim
        );

}

*/






void caf_main(caf::actor_system& sys) {

	  caf::core::init_global_meta_objects();
	  caf::init_global_meta_objects<caf::id_block::cuda>(); // 👈 This is the missing piece


	
	caf::cuda::manager::init(sys);
	//actor_facade_spawn_test(sys);
      	//actor_facade_launch_kernel_test(sys);
        //test_main(sys);

	serial_matrix_multiply_test();
	test_mmul(sys);
	
//	return 0;
}


//CAF_ALLOW_UNSAFE_MESSAGE_TYPE(in<char>)
//CAF_ALLOW_UNSAFE_MESSAGE_TYPE(in<int>)
//:CAF_ALLOW_UNSAFE_MESSAGE_TYPE(out<int>)

CAF_MAIN()

