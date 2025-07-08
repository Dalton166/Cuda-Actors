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

// create_in_arg overloads
template <typename T>
inline typename std::enable_if<is_scalar_v<T>, in<T>>::type
create_in_arg(T scalar_val) {
    return in<T>{scalar_val}; // in<T> resolves to in_impl<T, true>
}

template <typename T>
inline typename std::enable_if<is_scalar_v<T>, in<T>>::type
create_in_arg(const std::vector<T>& vec) {
    in_impl<T, false> arg;
    arg.buffer = vec;
    return arg; // in<T> resolves to in_impl<T, true>, but we return vector version
}

template <typename T>
inline typename std::enable_if<!is_scalar_v<T>, in<T>>::type
create_in_arg(const std::vector<T>& vec) {
    in<T> arg;
    arg.buffer = vec;
    return arg; // in<T> resolves to in_impl<T, false>
}


// create_in_out_arg overloads
template <typename T>
inline typename std::enable_if<is_scalar_v<T>, in_out<T>>::type
create_in_out_arg(T scalar_val) {
    return in_out<T>{scalar_val};
}

template <typename T>
inline typename std::enable_if<is_scalar_v<T>, in_out<T>>::type
create_in_out_arg(const std::vector<T>& vec) {
    in_out_impl<T, false> arg;
    arg.buffer = vec;
    return arg;
}

template <typename T>
inline typename std::enable_if<!is_scalar_v<T>, in_out<T>>::type
create_in_out_arg(const std::vector<T>& vec) {
    in_out<T> arg;
    arg.buffer = vec;
    return arg;
}


// create_out_arg overloads
template <typename T>
inline typename std::enable_if<is_scalar_v<T>, out<T>>::type
create_out_arg(T scalar_val) {
    return out<T>{scalar_val};
}

template <typename T>
inline typename std::enable_if<is_scalar_v<T>, out<T>>::type
create_out_arg(const std::vector<T>& vec) {
    out_impl<T, false> arg;
    arg.buffer = vec;
    return arg;
}

template <typename T>
inline typename std::enable_if<!is_scalar_v<T>, out<T>>::type
create_out_arg(const std::vector<T>& vec) {
    out<T> arg;
    arg.buffer = vec;
    return arg;
}


} // namespace caf::cuda

