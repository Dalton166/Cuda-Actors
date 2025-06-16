#pragma once

#include <cuda.h>
#include "caf/cuda/types.hpp"
#include "caf/cuda/manager.hpp"



/*
 * Most of these functions are just helper functions for 
 * requesting a resource or operation from the manager
 */
namespace caf::cuda {

// This helper calls manager singleton's get_context_by_id
inline CUcontext getContextById(int device_id, int context_id) {
  auto& mgr = manager::get();
  return mgr.get_context_by_id(device_id, context_id);
}



//helper function to launch a kernel for the command class 
inline void launch_kernel(program_ptr program,
                   const caf::cuda::nd_range& range,
                   std::tuple<mem_ptr<T>> args,
		   int stream_id) {

	
	int device_id = program -> get_device_id();
	int context_id = program -> get_context_id();
	CUfunction kernel = program -> get_kernel();
	auto& mgr = manager::get();
	device_ptr dev =  mgr.find_device(device_id);
	dev -> launch_kernel(kernel,range,args,stream_id,context_id);

}



template< typename T>
inline mem_ptr makeArg(int device_id,int context_id,in<T> arg) {

	auto& mgr = manager::get();
	device_ptr dev =  mgr.find_device(device_id);

	return dev -> make_arg(arg);
}	



template< typename T>
inline mem_ptr makeArg(int device_id,int context_id,in_out<T> arg) {

	auto& mgr = manager::get();
	device_ptr dev =  mgr.find_device(device_id);

	return dev -> make_arg(arg);
}	


template <typename T>
inline mem_ptr makeArg(int device_id,int context_id,out<T> arg) {

	auto& mgr = manager::get();
	device_ptr dev =  mgr.find_device(device_id);

	return dev -> make_arg(arg);
}	

template <typename T>
inline in<T> create_in_arg(std::vector<T> buffer) {

	in<T> arg;
	arg.buffer = buffer;
	return arg;
}

template <typename T>
inline in_out<T> create_in_out_arg(std::vector<T> buffer) {

	in_out<T> arg;
	arg.buffer = buffer;
	return arg;
}

template <typename T>
inline out<T> create_out_arg(std::vector<T> buffer) {

	out<T> arg;
	arg.buffer = buffer;
	return arg;
}



} // namespace caf::cuda

