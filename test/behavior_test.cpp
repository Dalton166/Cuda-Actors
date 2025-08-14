#include <caf/all.hpp>
#include <caf/cuda/all.hpp>
#include <iostream>
#include <iomanip>
#include <vector>
#include <chrono>
#include <thread>
#include <algorithm>
#include <numeric>
#include "caf/actor_registry.hpp"


//behavior that can generate 2 random matrices and launch the broadcast the results to other actors
//expects a message of size N and generates 2 N^2 matrices and broadcasts them out 

//note that the template corrosponds the the command parameters and not the message types
class randomMatrixBehavior : caf::cuda::SynchronousMulticastBehavior<in_out<int> in<int>,in<int>,in<int>> {

	randomMatrixBehavior(std::vector<caf::actor> targets) {	
        caf::cuda::manager& mgr = caf::cuda::manager::get();

	auto program = mgr.create_program_from_cubin("../mmul.cu","generate_random_matrix");
	//create a junk dim vec to pass to the superclass we wont be using this anyways
        caf::cuda::nd_range dim(1, 1, 1, 1,1, 1);


	//may need to forward the templates here 
	super::("randomMatrix",
			std::move(program),
			std::move(dim),
			nullptr,
			nullptr,
			targets);
	}

	protected:

	//expects the message to be in format of an int N and will generate 2 
	//N *N matrices 
	void execute(const caf::message& msg,
               int actor_id,
               caf::actor self) override {
		int N = msg.get_as<int>(0);

		//create a new message with the correct wrappers and call
		//execute_command_impl twice 
		
		//TODO maybe give an option to generate an std::vector of a size

		auto arg1 = caf::cuda::create_out_arg(N*N); //output buffer indicate its size, caf::cuda will handle the rest
		auto arg2 = caf::cuda::create_in_arg(N*N); //matrix size 
		auto arg3 = caf::cuda::create_in_arg(1234); //seed
		auto arg4 = caf::cuda::create_in_arg(9999); //max value
	        caf::message random_kernel_message = caf::make_message(arg1,arg2,arg3,arg4);


		//generate the 2 random matrices
		std::vector<output_buffer> output1 = super::execute_command(random_kernel_message,actor_id);
  
		std::vector<output_buffer> output2 = super::execute_command(random_kernel_message,actor_id);

		//collect their outputs 	
	 std::vector<int> matrixA;
          for (const auto& out : outputs1) {
            std::visit([&](const auto& vec) {
              if constexpr (std::is_same_v<std::decay_t<decltype(vec)>, std::vector<int>>) {
                matrixA = vec;
              }
            }, out.data);
          }	
	
	  std::vector<int> matrixB;
          for (const auto& out : outputs2) {
            std::visit([&](const auto& vec) {
              if constexpr (std::is_same_v<std::decay_t<decltype(vec)>, std::vector<int>>) {
                matrixB = vec;
              }
            }, out.data);
          }

	//package them into a message and broadcast the result
	auto matrix1 = caf::cuda::create_in_arg(matrixA);
	auto matrix2 = caf::cuda::create_in_arg(matrixB);
	auto matrix3 = caf::cuda::create_out_arg(N*N);
	auto size = caf::cuda::create_in_arg(N);
	caf::message multicast_msg = make_message(matrix1,matrix2,matrix3,size);

	//get the multicast behavior to broadcast this out for us
	super::reply(multicast_msg,self);

	}

}	






 //helper function to create a mmul behavior for actors 
 auto createMMulBehavior() {
  caf::cuda::manager& mgr = caf::cuda::manager::get();
  auto program = mgr.create_program_from_cubin("../mmul.cubin","matrixMul"); 
  caf::cuda::SynchronousUnicastBehavior<in<int>,in<int>,out<int>,in<int>> behavior("mmulBehavior",program,dim,nullptr,nullptr,self_actor,in<int>{},in<int>{} ,out<int>{},in<int>{});
  return behavior;
 }



void test_mmul_sync(caf::actor_system& sys, int N) {
  std::cout << "[TEST] Starting test_mmul_sync\n";


  caf::cuda::manager& mgr = caf::cuda::manager::get();
  auto program = mgr.create_program_from_cubin("../mmul.cubin","matrixMul");
  sys.spawn([&](caf::stateful_actor<mmul_sync_state>* self_actor) {



  int THREADS = 32;
  int BLOCKS = (N + THREADS - 1) / THREADS;

  caf::cuda::nd_range dim(BLOCKS, BLOCKS, 1, THREADS, THREADS, 1);


  caf::cuda::SynchronousUnicastBehavior<in<int>,in<int>,out<int>,in<int>> behavior("mmulBehavior",program,dim,nullptr,nullptr,self_actor,in<int>{},in<int>{} ,out<int>{},in<int>{});

  caf::cuda::behavior_ptr ptr = std::make_shared<decltype(behavior)>(behavior);
  auto gpuActor = mgr.spawnFromBehavior(
    ptr,in<int>{}, in<int>{}, out<int>{}, in<int>{}
  );

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

    auto& st = self_actor->state();
    st.gpu_actor = gpuActor;

    // Register lifecycle hooks
    self_actor->set_exit_handler([=](const caf::exit_msg& msg) {
      std::cout << "[EXIT HANDLER] test_mmul_sync received exit from actor: "
                << to_string(msg.source) << ", reason: " << caf::to_string(msg.reason) << "\n";
      if (msg.source == st.gpu_actor) {
        std::cerr << "[ERROR] GPU actor crashed or terminated unexpectedly!\n";
        self_actor->quit(msg.reason);
      }
    });

    self_actor->monitor(st.gpu_actor);

    self_actor->attach_functor([=](const caf::error& reason) {
      std::cout << "[EXIT] test_mmul_sync terminated with reason: "
                << caf::to_string(reason) << "\n";
    });

    // Send synchronous message
    st.start_time = Clock::now();
    self_actor->mail(arg1, arg2, arg3, arg4).send(gpuActor);

    return caf::behavior{
      [=](const std::vector<output_buffer>& outputs) {
        auto& st_ref = self_actor->state();
        auto end = Clock::now();
        std::chrono::duration<double> elapsed = end - st_ref.start_time;

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

        self_actor->send_exit(st_ref.gpu_actor, caf::exit_reason::user_shutdown);
        self_actor->quit();
      },
      [=](caf::error& err) {
        std::cerr << "[ERROR] test_mmul_sync kernel execution failed: "
                  << caf::to_string(err) << "\n";
        self_actor->quit(err);
      }
    };
  });

  sys.await_all_actors_done();
}


