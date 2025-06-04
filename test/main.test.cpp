

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

void actor_facade_spawn_test(caf::actor_system& sys) {

// Call this once before creating actor_system
   // caf::init_global_meta_objects<>();

	//caf::actor_system_config cfg;

	//caf::actor_system sys{cfg};

	//caf::opencl::actor_facade facade{cfg};
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

