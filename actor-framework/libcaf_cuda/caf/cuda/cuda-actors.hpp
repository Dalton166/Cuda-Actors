#pragma once

#include <type_traits>

namespace cuda_actors {

// Forward declaration of mem_ref<T>
template <class T>
class mem_ref {
public:
    using value_type = T;
    mem_ref(T* device_ptr) : ptr_(device_ptr) {}
private:
    T* ptr_;
};

// Helper trait to determine if a type is a valid CUDA argument
template <class T, class = void>
struct is_cuda_arg_helper : std::false_type {};

// Specialization for mem_ref<T>
template <class T>
struct is_cuda_arg_helper<mem_ref<T>> : std::true_type {};

// Specialization for arithmetic types
template <class T>
struct is_cuda_arg_helper<T, typename std::enable_if<std::is_arithmetic<T>::value>::type> : std::true_type {};

// Primary type trait: delegates to helper
template <class T>
struct is_cuda_arg : is_cuda_arg_helper<T> {};

} // namespace cuda_actors
