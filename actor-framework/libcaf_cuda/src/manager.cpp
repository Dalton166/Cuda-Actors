#include "caf/cuda/manager.hpp"
#include <stdexcept>

namespace caf::cuda {

	manager* manager::instance_ = nullptr;
	std::mutex manager::mutex_;




device_ptr manager::find_device(std::size_t) const {
  throw std::runtime_error("OpenCL support disabled: manager::find_device");
}

device_ptr manager::find_device(int id) {

	return platform_ -> getDevice(id);

}

program_ptr manager::create_program(const char * kernel,
                                    const std::string& name,
                                    device_ptr device) {
  

	CUdevice current_device = device -> getDevice();;

	int d_id = device -> getId();
	int c_id = device -> getContextId();
	int s_id = device -> getStreamId();


	//the compiled program can be accessed via ptx.data() afterwards
	std::vector<char> ptx;
	compile_nvrtc_program(kernel,current_device,ptx);
	program_ptr prog = make_counted<program>(name,device,d_id,c_id,s_id, ptx);
	return prog;
}

program_ptr manager::create_program_from_file(const std::string&,
                                              const char*,
                                              device_ptr) {
  throw std::runtime_error("OpenCL support disabled: manager::create_program_from_file");
}



// Helper: Get compute architecture string for nvrtc (e.g. "--gpu-architecture=compute_75")
std::string manager::get_computer_architecture_string(CUdevice device) {
    int major = 0, minor = 0;

    CUresult res1 = cuDeviceGetAttribute(&major, CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR, device);
    CUresult res2 = cuDeviceGetAttribute(&minor, CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR, device);

//std::cout << nvrtcVersion(&major,&minor) << "\n";
    if (res1 != CUDA_SUCCESS || res2 != CUDA_SUCCESS) {
        std::cerr << "Failed to get compute capability for device\n";
        return "";
    }

    return "--gpu-architecture=compute_" + std::to_string(major) + std::to_string(minor);
}

// Helper: Compile CUDA source to PTX for a specific device
// Returns true on success; on failure prints log and returns false
bool manager::compile_nvrtc_program(const char* source, CUdevice device, std::vector<char>& ptx_out) {
    // 1. Create NVRTC program
    nvrtcProgram prog;
    nvrtcResult res = nvrtcCreateProgram(&prog, source, "kernel.cu", 0, nullptr, nullptr);
    if (res != NVRTC_SUCCESS) {
        std::cerr << "nvrtcCreateProgram failed: " << nvrtcGetErrorString(res) << "\n";
        return false;
    }

    // 2. Get architecture string for device
    std::string arch = get_computer_architecture_string(device);
    if (arch.empty()) {
        nvrtcDestroyProgram(&prog);
        return false;
    }

    const char* options[] = {
    arch.c_str(),           // e.g., "--gpu-architecture=compute_70"
    "--std=c++11",          // Avoids too-modern C++ features
    "--fmad=false",         // Optional: disables fused multiply-add
    "--device-as-default-execution-space" // Optional compatibility
};
    

    // 3. Compile program
    res = nvrtcCompileProgram(prog, 1, options);

    // 4. Print compile log regardless of success/failure
    size_t logSize;
    nvrtcGetProgramLogSize(prog, &logSize);
    if (logSize > 1) {
        std::vector<char> log(logSize);
        nvrtcGetProgramLog(prog, log.data());
        std::cout << "NVRTC Compile Log:\n" << log.data() << "\n";
    }

    if (res != NVRTC_SUCCESS) {
        std::cerr << "nvrtcCompileProgram failed: " << nvrtcGetErrorString(res) << "\n";
        nvrtcDestroyProgram(&prog);
        return false;
    }

    // 5. Get PTX size
    size_t ptxSize;
    res = nvrtcGetPTXSize(prog, &ptxSize);
    if (res != NVRTC_SUCCESS) {
        std::cerr << "nvrtcGetPTXSize failed: " << nvrtcGetErrorString(res) << "\n";
        nvrtcDestroyProgram(&prog);
        return false;
    }

    // 6. Extract PTX
    ptx_out.resize(ptxSize);
    res = nvrtcGetPTX(prog, ptx_out.data());
    if (res != NVRTC_SUCCESS) {
        std::cerr << "nvrtcGetPTX failed: " << nvrtcGetErrorString(res) << "\n";
        nvrtcDestroyProgram(&prog);
        return false;
    }

    // 7. Clean up
    nvrtcDestroyProgram(&prog);
    return true;
}


 CUcontext manager::get_context_by_id(int device_id, int context_id) {
  device_ptr dev = find_device(device_id);
  if (!dev) {
    throw std::runtime_error("No CUDA device found with id: " + std::to_string(device_id));
  }
  return dev->getContext();
}



} // namespace caf::cuda
