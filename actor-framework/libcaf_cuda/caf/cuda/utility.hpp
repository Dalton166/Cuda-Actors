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

// ====================
// in<T> factory helpers
// ====================

// Construct in<T> from a scalar (e.g. int, float)
template <typename T>
inline in<T> create_in_arg(T scalar_val) {
  static_assert(is_scalar_v<T>, "create_in_arg(T) expects a scalar type.");
  return in<T>{scalar_val}; // Will resolve to in_impl<T, true>
}

// Construct in<T> from a vector (only for non-scalar T)
template <typename T>
inline in<T> create_in_arg(const std::vector<T>& buffer) {
  static_assert(!is_scalar_v<T>, "create_in_arg(std::vector<T>) is not allowed for scalar types.");
  in<T> arg;                 // Will resolve to in_impl<T, false>
  arg.buffer = buffer;
  return arg;
}

// ======================
// out<T> factory helpers
// ======================

// Construct out<T> from a scalar
template <typename T>
inline out<T> create_out_arg(T scalar_val) {
  static_assert(is_scalar_v<T>, "create_out_arg(T) expects a scalar type.");
  return out<T>{scalar_val};
}

// Construct out<T> from a vector (only for non-scalar T)
template <typename T>
inline out<T> create_out_arg(const std::vector<T>& buffer) {
  static_assert(!is_scalar_v<T>, "create_out_arg(std::vector<T>) is not allowed for scalar types.");
  out<T> arg;
  arg.buffer = buffer;
  return arg;
}

// ==========================
// in_out<T> factory helpers
// ==========================

// Construct in_out<T> from a scalar
template <typename T>
inline in_out<T> create_in_out_arg(T scalar_val) {
  static_assert(is_scalar_v<T>, "create_in_out_arg(T) expects a scalar type.");
  return in_out<T>{scalar_val};
}

// Construct in_out<T> from a vector (only for non-scalar T)
template <typename T>
inline in_out<T> create_in_out_arg(const std::vector<T>& buffer) {
  static_assert(!is_scalar_v<T>, "create_in_out_arg(std::vector<T>) is not allowed for scalar types.");
  in_out<T> arg;
  arg.buffer = buffer;
  return arg;
}

} // namespace caf::cuda

