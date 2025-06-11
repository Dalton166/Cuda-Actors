#pragma once

#include <stdexcept>





namespace caf::cuda {

template <class T>
class mem_ref {
public:
  using value_type = T;

  mem_ref() = default;

  /*
  mem_ref(void*, void*, size_t) {
    throw std::runtime_error("CUDA support disabled: mem_ref ctor");
  }
  */


  //this is a default constructor that will likely be changed in the future
  mem_ref(size_t num_elements, ,
          CUdeviceptr memory, int access,
          )
    : num_elements_{num_elements},
      access_{access},
      memory_{memory} {
    // nop
  }

  ~mem_ref() { 
	  reset();
  }
  mem_ref(mem_ref&&) noexcept = default;
  mem_ref& operator=(mem_ref&&) noexcept = default;

  void* queue() const { return nullptr; }
  void* mem() const { return nullptr; }
  size_t size() const { return 0; }

  void reset() {
  
	  CuMemFree(memory_);
	  num_elements = 0;
	  memory_ = 0;
	  access = NOT_IN_USE; 
  }


private:
  CUdeviceptr memory_;
  int num_elements;
  int access_;

};

} // namespace caf::cuda
