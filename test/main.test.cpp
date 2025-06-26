

/*
 * this file does absolutely nothing right now 
 * should however test the actor facade spawn function 
 */

#include "main.test.hpp"

/*
CAF_BEGIN_TYPE_ID_BLOCK(cuda_test, caf::first_custom_type_id)

  CAF_ADD_TYPE_ID(cuda_test, (std::vector<char>))
  CAF_ADD_TYPE_ID(cuda_test, (std::vector<int>))
  CAF_ADD_TYPE_ID(cuda_test, (in<int>))
  CAF_ADD_TYPE_ID(cuda_test, (in<char>))
  CAF_ADD_TYPE_ID(cuda_test, (out<int>))

CAF_END_TYPE_ID_BLOCK(cuda_test)
*/


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
//	actor_facade_spawn_test(sys);

      actor_facade_launch_kernel_test(sys);
//       test_main(sys);
	
//	return 0;
}


//CAF_ALLOW_UNSAFE_MESSAGE_TYPE(in<char>)
//CAF_ALLOW_UNSAFE_MESSAGE_TYPE(in<int>)
//:CAF_ALLOW_UNSAFE_MESSAGE_TYPE(out<int>)

CAF_MAIN()

