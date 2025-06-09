

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




	caf::cuda::manager mgr{sys};
	int length = 10;
	const char str1[length];
	const char str2[length];
	int result[10];

	auto gpuActor = mgr.spawn(kernel_code,"myKernel",str1,str2,length,result);




}


void actor_facade_spawn_test(caf::actor_system& sys) {

	caf::cuda::manager mgr{sys};

	int x = 1;
	auto gpuActor = mgr.spawn(x);


 // actor_system_config cfg;
  //actor_system system{cfg};

  // Spawn actor_facade<int, std::string> passing 42 and "hello"
  //auto my_actor = system.spawn<caf::opencl::actor_facade<false,int, std::string>>(42, "hello");

}




void caf_main(caf::actor_system& sys) {

	cuInit(0);
	actor_facade_spawn_test(sys);

//	return 0;
}


CAF_MAIN()

