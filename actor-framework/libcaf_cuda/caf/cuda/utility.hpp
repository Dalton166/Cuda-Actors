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


template< typename T>
inline mem_ptr makeArg(int device_id,int context_id,out<T> arg) {

	auto& mgr = manager::get();
	device_ptr dev =  mgr.find_device(device_id);

	return dev -> make_arg(arg);
}	





} // namespace caf::cuda

