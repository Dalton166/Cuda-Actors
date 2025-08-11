
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
#include <random>
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
using mmulGenCommand = caf::cuda::command_runner<out<int>,in<int>,in<int>,in<int>>;


// Define atoms for CPU and GPU
using cpu_atom = caf::atom_constant<caf::atom("cpu")>;
using gpu_atom = caf::atom_constant<caf::atom("gpu")>;



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







// Actor state
struct my_actor_state {
  static inline const char* name = "my_actor";
  int last_N = 0; // example state variable
  int id = rand(); //an actors id 
};

// Stateful actor behavior
caf::behavior my_actor_fun(caf::stateful_actor<my_actor_state>* self) {
  return {
    // 1st handler: Just int N, and who to send the matrices to
    [=](int N, std::vector<caf::actor> receivers) {
      self->state.last_N = N;

        //create the program and configure the dimesnions of the kernel
        auto program = mgr.create_program_from_cubin("../mmul.cu","generate_random_matrix");
	int THREADS = 256;
	int BLOCKS = (N*N + THREADS - 1) / THREADS;
  	caf::cuda::nd_range dim(BLOCKS,1, 1, THREADS,1, 1);

	//tag the arguments so that caf::cuda knows what to do with them	
         auto arg1 = caf::cuda::create_out_arg(N*N); //output buffer indicate its size, caf::cuda will handle the rest
          auto arg2 = caf::cuda::create_in_arg(N*N); //matrix size
          auto arg3 = caf::cuda::create_in_arg(1234); //seed
	  auto arg4 = caf::cuda::create_in_arg(9999); //max valux
	  


	  //launch kernels and collect their outputs
	  auto tempA = matrixGenCommand(program,dim, self -> state.id,arg1,arg2,arg3,arg4);
	  auto tempB = matrixGenCommand(program,dim, self -> state.id,arg1,arg2,arg3,arg4);
	  std::vector<int> matrixA =  caf::cuda::collect_vector(tempA);
	  std::vector<int> matrixB = caf::cuda::collect_vector(tempB);

	  //broadcast the result out to receviers.
	  for (auto actor: receivers) {
	  
		  self->mail(gpu_atom,matrixA,matrixB,N).send(actor);
	  }

    },

    // 2nd handler: GPU atom + matrices + N
    [=](gpu_atom, const std::vector<int> matrixA,
        const std::vector<int> matrixB, int N) {
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
