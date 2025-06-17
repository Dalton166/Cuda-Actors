

/*
 * this file does absolutely nothing right now 
 * should however test the actor facade spawn function 
 */

#include <caf/all.hpp>  // Includes most CAF essentials

#include "caf/cuda/actor_facade.hpp"
#include "caf/cuda/manager.hpp"
#include "caf/cuda/nd_range.hpp"
#include "caf/cuda/all.hpp"
#include "caf/cuda/utility.hpp"
#include <caf/type_id.hpp>


CAF_BEGIN_TYPE_ID_BLOCK(cuda_test, caf::first_custom_type_id)

  CAF_ADD_TYPE_ID(cuda_test, (std::vector<char>))
  CAF_ADD_TYPE_ID(cuda_test, (std::vector<int>))
  CAF_ADD_TYPE_ID(cuda_test, (in<int>))
  CAF_ADD_TYPE_ID(cuda_test, (in<char>))
  CAF_ADD_TYPE_ID(cuda_test, (out<int>))

CAF_END_TYPE_ID_BLOCK(cuda_test)

//using namespace caf;

//runs a test to ensure the actor facade can spawn in
//correctly 

const char* kernel_code = R"(
extern "C" __global__
void compare_strings(const char* a, const char* b, int* result, int length) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < length) {
        result[idx] = (a[idx] == b[idx]) ? 1 : 0;
    }
}
)";

void actor_facade_launch_kernel_test(caf::actor_system& sys) {
    caf::cuda::manager& mgr = caf::cuda::manager::get();
    int length = 10;
    std::vector<char> str1(length);
    std::vector<char> str2(length);
    std::vector<int> result(length);
    std::vector<int> len(1);
    len[0] = length;

    // 1 dimension for blocks and grids
    caf::cuda::nd_range dim(1,1,1,1,1,1);

    // Spawn the CUDA actor
    auto gpuActor = mgr.spawn(kernel_code, "compare_strings", dim);

    // Create the necessary input/output args
    auto arg1 = caf::cuda::create_in_arg(str1);
    auto arg2 = caf::cuda::create_in_arg(str2);
    auto arg3 = caf::cuda::create_out_arg(result);
    auto arg4 = caf::cuda::create_in_arg(len);

    // Send a message with the args to the actor
    // The actor_facade is expected to handle this message and call run_kernel internally
     anon_mail(gpuActor, arg1, arg2, arg3, arg4);
     //anon_mail(gpuActor, str1, str2, result, length);
     //anon_mail(gpuActor, std::vector<char>{str1}, std::vector<char>{str2}, std::vector<int>{result}, length);


    // Optionally, you can wait for a response or schedule followup steps
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

	caf::cuda::manager::init(sys);
//	actor_facade_spawn_test(sys);

      actor_facade_launch_kernel_test(sys);
	
//	return 0;
}


CAF_MAIN()

