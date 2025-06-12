

/*
 * this file does absolutely nothing right now 
 * should however test the actor facade spawn function 
 */

#include <caf/all.hpp>  // Includes most CAF essentials

#include "caf/cuda/actor_facade.hpp"
#include "caf/cuda/manager.hpp"

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

    // Pass std::move for vectors you want to forward as rvalues
    auto gpuActor = mgr.spawn(
        kernel_code,
        "compare_strings",
        std::move(str1),
        std::move(str2),
        std::move(length),
        std::move(result));
}

/*
void actor_facade_spawn_test(caf::actor_system& sys) {

	caf::cuda::manager mgr{sys};

	int x = 1;
	auto gpuActor = mgr.spawn(x);


 // actor_system_config cfg;
  //actor_system system{cfg};

  // Spawn actor_facade<int, std::string> passing 42 and "hello"
  //auto my_actor = system.spawn<caf::opencl::actor_facade<false,int, std::string>>(42, "hello");

}

*/


void caf_main(caf::actor_system& sys) {

	caf::cuda::manager::init(sys);
//	actor_facade_spawn_test(sys);

      actor_facade_launch_kernel_test(sys);
	
//	return 0;
}


CAF_MAIN()

