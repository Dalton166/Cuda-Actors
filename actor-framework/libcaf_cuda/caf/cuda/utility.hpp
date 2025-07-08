#pragma once

#include <cuda.h>
#include "caf/cuda/types.hpp"
#include "caf/cuda/manager.hpp"
#include <type_traits>




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



/*
//helper function to launch a kernel for the command class 
template <typename ... Ts>
inline void launch_kernel(program_ptr program,
                   const caf::cuda::nd_range& range,
                   std::tuple<mem_ptr<Ts> ...> args,
		   int stream_id) {

	
	int device_id = program -> get_device_id();
	int context_id = program -> get_context_id();
	CUfunction kernel = program -> get_kernel();
	auto& mgr = manager::get();
	device_ptr dev =  mgr.find_device(device_id);
	dev -> launch_kernel(kernel,range,args,stream_id,context_id);

}
*/


template< typename T>
inline mem_ptr<T> makeArg(int device_id,int context_id,in<T> arg) {

	auto& mgr = manager::get();
	device_ptr dev =  mgr.find_device(device_id);

	return dev -> make_arg(arg);
}	



template< typename T>
inline mem_ptr<T> makeArg(int device_id,int context_id,in_out<T> arg) {

	auto& mgr = manager::get();
	device_ptr dev =  mgr.find_device(device_id);

	return dev -> make_arg(arg);
}	


template <typename T>
inline mem_ptr <T> makeArg(int device_id,int context_id,out<T> arg) {

	auto& mgr = manager::get();
	device_ptr dev =  mgr.find_device(device_id);

	return dev -> make_arg(arg);
}	



// in<T>
template <typename T>
in<T> create_in_arg(const std::vector<T>& buffer) {
  in<T> arg;
  if constexpr (!is_scalar_v<T>) {
    arg.buffer = buffer;
  }
  return arg;
}

template <typename T>
in<T> create_in_arg(T val) {
  static_assert(is_scalar_v<T>, "create_in_arg(T) is only allowed for scalar types.");
  return in<T>{val};
}

// out<T>
template <typename T>
out<T> create_out_arg(const std::vector<T>& buffer) {
  out<T> arg;
  if constexpr (!is_scalar_v<T>) {
    arg.buffer = buffer;
  }
  return arg;
}

template <typename T>
out<T> create_out_arg(T val) {
  static_assert(is_scalar_v<T>, "create_out_arg(T) is only allowed for scalar types.");
  return out<T>{val};
}

// in_out<T>
template <typename T>
in_out<T> create_in_out_arg(const std::vector<T>& buffer) {
  in_out<T> arg;
  if constexpr (!is_scalar_v<T>) {
    arg.buffer = buffer;
  }
  return arg;
}

template <typename T>
in_out<T> create_in_out_arg(T val) {
  static_assert(is_scalar_v<T>, "create_in_out_arg(T) is only allowed for scalar types.");
  return in_out<T>{val};
}


}

