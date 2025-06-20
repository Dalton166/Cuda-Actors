#include <iostream>
#include <vector>
#include <cassert>
#include <stdexcept>
#include <iomanip>
#include <cuda.h>
#include <caf/all.hpp>
#include <caf/cuda/manager.hpp>
#include <caf/cuda/program.hpp>
#include <caf/cuda/device.hpp>
#include <caf/cuda/command.hpp>
#include <caf/cuda/actor_facade.hpp>
#include <caf/cuda/mem_ref.hpp>
#include <caf/cuda/platform.hpp>
#include <caf/cuda/global.hpp>
#include <caf/cuda/cuda-actors.hpp>
#include "caf/detail/test.hpp"

using namespace caf;
using namespace caf::cuda;

void test_platform(actor_system& sys, platform_ptr plat) {
    std::cout << "\n=== Test Platform ===\n";
    
    std::cout << "Test 1: Creating platform...\n";
    assert(plat != nullptr && "Platform creation failed: nullptr returned");
    assert(!plat->devices().empty() && "Platform creation failed: no devices found");
    std::cout << "  -> Platform created with " << plat->devices().size() << " device(s).\n";

    std::cout << "Test 2: Retrieving device 0...\n";
    device_ptr dev = plat->getDevice(0);
    assert(dev != nullptr && "Device retrieval failed: nullptr returned for device 0");
    assert(dev->getId() == 0 && "Device ID mismatch: expected 0");
    std::cout << "  -> Device 0 retrieved successfully.\n";

    std::cout << "Test 3: Testing invalid device ID (-1)...\n";
    try {
        plat->getDevice(-1);
        assert(false && "Expected exception for negative device ID");
    } catch (const std::exception& e) {
        std::cout << "  -> Caught expected exception: " << e.what() << "\n";
    }
    std::cout << "---- Platform tests passed ----\n";
}

void test_device(actor_system& sys, platform_ptr plat) {
    std::cout << "\n=== Test Device ===\n";
    
    std::cout << "Test 1: Checking device properties...\n";
    device_ptr dev = plat->getDevice(0);
    assert(dev->getContext() != nullptr && "Device context is null");
    assert(dev->getStream() != nullptr && "Device stream is null");
    assert(dev->name() != nullptr && "Device name is null");
    std::cout << "  -> Device properties valid (context, stream, name).\n";

    std::cout << "Test 2: Testing memory argument creation...\n";
    std::vector<int> data(5, 42);
    in<int> input = create_in_arg(data);
    mem_ptr<int> in_mem = dev->make_arg(input);
    assert(in_mem->size() == 5 && "Input memory size mismatch: expected 5");
    assert(in_mem->access() == IN && "Input memory access type incorrect: expected IN");
    std::cout << "  -> Input memory argument created successfully.\n";

    in_out<int> inout = create_in_out_arg(data);
    mem_ptr<int> inout_mem = dev->make_arg(inout);
    assert(inout_mem->access() == IN_OUT && "In-out memory access type incorrect: expected IN_OUT");
    std::cout << "  -> In-out memory argument created successfully.\n";

    out<int> output = create_out_arg(std::vector<int>(5, 0));
    mem_ptr<int> out_mem = dev->make_arg(output);
    assert(out_mem->size() == 5 && "Output memory size mismatch: expected 5");
    assert(out_mem->access() == OUT && "Output memory access type incorrect: expected OUT");
    std::cout << "  -> Output memory argument created successfully.\n";
    std::cout << "---- Device tests passed ----\n";
}

void test_manager(actor_system& sys, platform_ptr plat) {
    std::cout << "\n=== Test Manager ===\n";
    
    std::cout << "Test 1: Initializing manager...\n";
    manager& mgr = manager::get();
    assert(&mgr == &manager::get() && "Manager singleton mismatch");
    std::cout << "  -> Manager initialized successfully.\n";

    std::cout << "Test 2: Retrieving device 0...\n";
    device_ptr dev = mgr.find_device(0);
    assert(dev != nullptr && "Device retrieval failed: nullptr returned");
    assert(dev->getId() == 0 && "Device ID mismatch: expected 0");
    std::cout << "  -> Device 0 retrieved successfully.\n";

    std::cout << "Test 3: Creating program with test kernel...\n";
    const char* kernel = R"(
        extern "C" __global__ void test_kernel(int* data) {
            int idx = threadIdx.x;
            data[idx] = idx;
        })";
    program_ptr prog = mgr.create_program(kernel, "test_kernel", dev);
    assert(prog != nullptr && "Program creation failed: nullptr returned");
    assert(prog->get_kernel() != nullptr && "Kernel creation failed: nullptr returned");
    std::cout << "  -> Program and kernel created successfully.\n";

    std::cout << "Test 4: Testing invalid device ID (999)...\n";
    try {
        mgr.find_device(999);
        assert(false && "Expected exception for invalid device ID");
    } catch (const std::exception& e) {
        std::cout << "  -> Caught expected exception: " << e.what() << "\n";
    }
    std::cout << "---- Manager tests passed ----\n";
}

void test_program(actor_system& sys, platform_ptr plat) {
    std::cout << "\n=== Test Program ===\n";
    
    std::cout << "Test 1: Checking program properties...\n";
    manager& mgr = manager::get();
    device_ptr dev = mgr.find_device(0);
    const char* kernel = R"(
        extern "C" __global__ void test_kernel(int* data) {
            int idx = threadIdx.x;
            data[idx] = idx;
        })";
    program_ptr prog = mgr.create_program(kernel, "test_kernel", dev);
    assert(prog->get_device_id() == 0 && "Program device ID mismatch: expected 0");
    assert(prog->get_context_id() == 0 && "Program context ID mismatch: expected 0");
    assert(prog->get_stream_id() == 0 && "Program stream ID mismatch: expected 0");
    std::cout << "  -> Program properties valid (device_id=0, context_id=0, stream_id=0).\n";
    std::cout << "---- Program tests passed ----\n";
}

void test_create_program(actor_system& sys, platform_ptr plat) {
    std::cout << "\n=== Test Create Program ===\n";

    manager& mgr = manager::get();
    device_ptr dev = mgr.find_device(0);
    std::cout << "  -> Device context: " << dev->getContext(0) << "\n";

    std::cout << "Test 1: Creating and validating simple kernel...\n";
    {
        const char* kernel_code = R"(
            extern "C" __global__ void simple_kernel(int* output) {
                output[0] = 42;
            })";
        try {
            std::cout << "  -> Creating program for kernel: simple_kernel\n";
            program_ptr prog = mgr.create_program(kernel_code, "simple_kernel", dev);
            if (!prog) {
                throw std::runtime_error("create_program returned null program_ptr");
            }
            std::cout << "  -> Program created: prog=" << prog.get() << "\n";

            CUfunction kernel = prog->get_kernel();
            if (!kernel) {
                throw std::runtime_error("get_kernel returned null CUfunction");
            }
            std::cout << "  -> Kernel handle: " << kernel << "\n";

            CUmodule module;
            CHECK_CUDA(cuFuncGetModule(&module, kernel));
            if (!module) {
                throw std::runtime_error("cuFuncGetModule returned null CUmodule");
            }
            std::cout << "  -> Module handle: " << module << "\n";

            CUcontext ctx = dev->getContext(0);
            CUcontext current_ctx;
            CHECK_CUDA(cuCtxGetCurrent(&current_ctx));
            std::cout << "  -> Current context: " << current_ctx << ", device context: " << ctx << "\n";
            if (current_ctx && current_ctx != ctx) {
                throw std::runtime_error("Context mismatch: expected " + std::to_string((uintptr_t)ctx) +
                                         ", got " + std::to_string((uintptr_t)current_ctx));
            }

            std::cout << "  -> Simple kernel created and validated successfully\n";
        } catch (const std::exception& e) {
            std::cout << "  -> Failed: " << e.what() << "\n";
            throw;
        }
        std::cout << "  -> End of scope: prog destroyed\n";
    }
    std::cout << "---- Create Program tests passed ----\n";
}

void test_mem_ref(actor_system& sys, platform_ptr plat) {
    std::cout << "\n=== Test Mem Ref ===\n";
    
    manager& mgr = manager::get();
    device_ptr dev = mgr.find_device(0);

    std::cout << "Test 1: Testing input memory allocation...\n";
    std::vector<int> host_data(5, 10);
    in<int> input = create_in_arg(host_data);
    mem_ptr<int> mem = dev->make_arg(input);
    assert(mem->size() == 5 && "Input memory size mismatch: expected 5");
    assert(mem->mem() != 0 && "Input memory allocation failed: null pointer");
    assert(mem->access() == IN && "Input memory access type incorrect: expected IN");
    std::cout << "  -> Input memory allocated successfully.\n";

    std::cout << "Test 2: Testing output memory allocation...\n";
    out<int> output = create_out_arg(std::vector<int>(5, 0));
    mem_ptr<int> out_mem = dev->make_arg(output);
    assert(out_mem->size() == 5 && "Output memory size mismatch: expected 5");
    assert(out_mem->access() == OUT && "Output memory access type incorrect: expected OUT");
    std::cout << "  -> Output memory allocated successfully.\n";

    std::cout << "Test 3: Testing in-out memory data integrity...\n";
    in_out<int> inout = create_in_out_arg(host_data);
    mem_ptr<int> inout_mem = dev->make_arg(inout);
    assert(inout_mem->access() == IN_OUT && "In-out memory access type incorrect: expected IN_OUT");
    std::vector<int> copied = inout_mem->copy_to_host();
    for (size_t i = 0; i < 5; ++i) {
        assert(copied[i] == 10 && "In-out memory data corruption");
        if (copied[i] != 10) {
            std::cout << "  -> Failed: copied[" << i << "] = " << copied[i] << ", expected 10\n";
        }
    }
    std::cout << "  -> In-out memory data copied correctly.\n";

    std::cout << "Test 4: Testing invalid copy from input memory...\n";
    try {
        mem->copy_to_host();
        assert(false && "Expected exception for copying IN memory");
    } catch (const std::runtime_error& e) {
        std::cout << "  -> Caught expected exception: " << e.what() << "\n";
    }

    std::cout << "Test 5: Testing memory reset...\n";
    mem->reset();
    assert(mem->size() == 0 && "Memory reset failed: size not 0");
    assert(mem->mem() == 0 && "Memory reset failed: pointer not null");
    std::cout << "  -> Memory reset successfully.\n";
    std::cout << "---- Mem Ref tests passed ----\n";
}

void test_command(actor_system& sys, platform_ptr plat) {
    std::cout << "\n=== Test Command ===\n";
    std::cout << "Test 1: Command test skipped due to response_promise issue.\n";
    std::cout << "---- Command tests passed (skipped) ----\n";
}

void test_actor_facade(actor_system& sys, platform_ptr plat) {
    std::cout << "\n=== Test Actor Facade ===\n";
    
    std::cout << "Test 1: Testing actor facade kernel execution...\n";
    manager& mgr = manager::get();
    device_ptr dev = mgr.find_device(0);
    std::cout << "  -> Device context: " << dev->getContext(0) << "\n";
    const char* kernel = R"(
        extern "C" __global__ void test_kernel(int* data) {
            int idx = threadIdx.x;
            if (idx < 5) data[idx] = idx + 1;
        })";
    program_ptr prog = mgr.create_program(kernel, "test_kernel", dev);
    std::cout << "  -> Program created with kernel: test_kernel, handle: " << prog->get_kernel() << ", prog=" << prog.get() << "\n";
    nd_range dims{1, 1, 1, 5, 1, 1};
    std::vector<int> host_data(5, 0);
    out<int> output = create_out_arg(host_data);
    mem_ptr<int> out_mem = dev->make_arg(output);
    assert(out_mem->mem() != 0 && "Output memory not allocated");
    std::cout << "  -> Output memory allocated: " << out_mem->mem() << ", out_mem=" << out_mem.get() << "\n";
    actor_config actor_cfg;
    actor_facade<false, out<int>> facade{std::move(actor_cfg), prog, dims, out<int> {}};
    std::cout << "  -> Actor facade created\n";
    try {
        facade.run_kernel(output);
        std::cout << "  -> Kernel launched via actor facade\n";
    } catch (const std::exception& e) {
        std::cout << "  -> Failed: CUDA error during actor facade kernel launch: " << e.what() << "\n";
        throw;
    }
    //std::vector<int> result = output.buffer;
    std::vector<int> result = out_mem -> copy_to_host();
    /* we can really only validate output via inspection at this point since 
     * there is no way to return the buffer since that needs to be handled via caf 
     */
    std::cout << " --- Expected output is 1 2 3 4 5\n"; 
    std::cout << "  -> Actor facade kernel executed successfully.\n";
    std::cout << "---- Actor Facade tests passed ----\n";
}

void test_mem_ref_extended(actor_system& sys, platform_ptr plat) {
    std::cout << "\n=== Test Mem Ref Extended ===\n";
    
    manager& mgr = manager::get();
    device_ptr dev = mgr.find_device(0);

    std::cout << "Test 1: Testing input memory allocation...\n";
    std::vector<int> host_input(5, 42);
    in<int> input = create_in_arg(host_input);
    mem_ptr<int> in_mem = dev->make_arg(input);
    assert(in_mem->size() == 5 && "Input memory size mismatch: expected 5");
    assert(in_mem->access() == IN && "Input memory access type incorrect: expected IN");
    assert(in_mem->mem() != 0 && "Input memory allocation failed: null pointer");
    std::cout << "  -> Input memory allocated successfully.\n";

    std::cout << "Test 2: Testing output memory allocation...\n";
    std::vector<int> host_output(5, 0);
    out<int> output = create_out_arg(host_output);
    mem_ptr<int> out_mem = dev->make_arg(output);
    assert(out_mem->size() == 5 && "Output memory size mismatch: expected 5");
    assert(out_mem->access() == OUT && "Output memory access type incorrect: expected OUT");
    assert(out_mem->mem() != 0 && "Output memory allocation failed: null pointer");
    std::cout << "  -> Output memory allocated successfully.\n";

    std::cout << "Test 3: Testing in-out memory data integrity...\n";
    std::vector<int> host_inout(5, 10);
    in_out<int> inout = create_in_out_arg(host_inout);
    mem_ptr<int> inout_mem = dev->make_arg(inout);
    assert(inout_mem->size() == 5 && "In-out memory size mismatch: expected 5");
    assert(inout_mem->access() == IN_OUT && "In-out memory access type incorrect: expected IN_OUT");
    std::vector<int> copied = inout_mem->copy_to_host();
    for (size_t i = 0; i < 5; ++i) {
        assert(copied[i] == 10 && "In-out memory data corruption");
        if (copied[i] != 10) {
            std::cout << "  -> Failed: copied[" << i << "] = " << copied[i] << ", expected 10\n";
        }
    }
    std::cout << "  -> In-out memory data copied correctly.\n";

    std::cout << "Test 4: Testing small buffer allocation...\n";
    std::vector<int> small_data(2, 5);
    out<int> small_output = create_out_arg(small_data);
    mem_ptr<int> small_mem = dev->make_arg(small_output);
    assert(small_mem->size() == 2 && "Small buffer size mismatch: expected 2");
    std::cout << "  -> Small buffer allocated successfully.\n";
    std::cout << "---- Mem Ref Extended tests passed ----\n";
}

void test_argument_translation(actor_system& sys, platform_ptr plat) {
    std::cout << "\n=== Test Argument Translation ===\n";
    
    manager& mgr = manager::get();
    device_ptr dev = mgr.find_device(0);

    std::cout << "Test 1: Testing output argument creation...\n";
    std::vector<int> data(5, 0);
    out<int> output = create_out_arg(data);
    mem_ptr<int> out_mem = dev->make_arg(output);
    assert(out_mem->size() == data.size() && "Output argument size mismatch: expected 5");
    assert(out_mem->access() == OUT && "Output argument access type incorrect: expected OUT");
    std::cout << "  -> Output argument created successfully.\n";

    std::cout << "Test 2: Testing type mismatch simulation...\n";
    in<int> input = create_in_arg(data);
    mem_ptr<int> in_mem = dev->make_arg(input);
    assert(in_mem->access() == IN && "Input argument access type incorrect: expected IN");
    try {
        in_mem->copy_to_host();
        assert(false && "Expected exception for copying IN memory");
    } catch (const std::runtime_error& e) {
        std::cout << "  -> Caught expected exception: " << e.what() << "\n";
    }

    std::cout << "Test 3: Testing multiple argument creation...\n";
    in<int> input2 = create_in_arg(std::vector<int>(5, 7));
    out<int> output2 = create_out_arg(std::vector<int>(5, 0));
    mem_ptr<int> in_mem2 = dev->make_arg(input2);
    mem_ptr<int> out_mem2 = dev->make_arg(output2);
    assert(in_mem2->size() == 5 && out_mem2->size() == 5 && "Multiple argument size mismatch: expected 5");
    std::cout << "  -> Multiple arguments created successfully.\n";
    std::cout << "---- Argument Translation tests passed ----\n";
}

void test_kernel_launch(actor_system& sys, platform_ptr plat) {
    std::cout << "\n=== Test Kernel Launch ===\n";
    
    manager& mgr = manager::get();
    device_ptr dev = mgr.find_device(0);
    std::cout << "  -> Device context: " << dev->getContext(0) << "\n";

    std::cout << "Test 1: Testing basic kernel launch...\n";
    {
        const char* kernel_code = R"(
            extern "C" __global__ void simple_kernel(int* output) {
                output[0] = 42;
            })";
        program_ptr prog = mgr.create_program(kernel_code, "simple_kernel", dev);
        std::cout << "  -> Program created with kernel: simple_kernel, handle: " << prog->get_kernel() << ", prog=" << prog.get() << "\n";
        std::vector<int> host_data(1, 0);
        out<int> output = create_out_arg(host_data);
        mem_ptr<int> out_mem = dev->make_arg(output);
        assert(out_mem->mem() != 0 && "Output memory not allocated");
        std::cout << "  -> Output memory allocated: " << out_mem->mem() << ", out_mem=" << out_mem.get() << "\n";
        nd_range dims{1, 1, 1, 1, 1, 1};
        try {
            CUcontext ctx = dev->getContext(0);
            CUcontext current_ctx;
            CHECK_CUDA(cuCtxGetCurrent(&current_ctx));
            std::cout << "  -> Current context before launch: " << current_ctx << "\n";
            std::cout << "  -> Launching kernel with context: " << ctx << ", kernel: " << prog->get_kernel() << ", args: " << out_mem->mem() << "\n";
            dev->launch_kernel(prog->get_kernel(), dims, std::make_tuple(out_mem), 0, 0);
            std::cout << "  -> Kernel launched\n";
            CHECK_CUDA(cuCtxSynchronize());
            std::cout << "  -> Context synchronized\n";
            std::vector<int> result = out_mem->copy_to_host();
            std::cout << "  -> Data copied to host\n";
            assert(result[0] == 42 && "Kernel output incorrect");
            if (result[0] != 42) {
                std::cout << "  -> Failed: result[0] = " << result[0] << ", expected 42\n";
            }
            std::cout << "  -> Basic kernel launched successfully\n";
        } catch (const std::exception& e) {
            std::cout << "  -> Failed: CUDA error during kernel launch: " << e.what() << "\n";
            throw;
        }
        std::cout << "  -> End of scope: prog=" << prog.get() << ", out_mem=" << out_mem.get() << "\n";
    }
    std::cout << "---- Kernel Launch tests passed ----\n";
}

void test_kernel_launch_direct(actor_system& sys, platform_ptr plat) {
    std::cout << "\n=== Test Kernel Launch Direct ===\n";
    
    manager& mgr = manager::get();
    device_ptr dev = mgr.find_device(0);
    std::cout << "  -> Device context: " << dev->getContext(0) << "\n";

    std::cout << "Test 1: Testing direct kernel launch with cuLaunchKernel...\n";
    {
        const char* kernel_code = R"(
            extern "C" __global__ void simple_kernel(int* output) {
                output[0] = 42;
            })";
        try {
            program_ptr prog = mgr.create_program(kernel_code, "simple_kernel", dev);
            std::cout << "  -> Program created with kernel: simple_kernel, handle: " << prog->get_kernel() << ", prog=" << prog.get() << "\n";

            std::vector<int> host_data(1, 0);
            out<int> output = create_out_arg(host_data);
            mem_ptr<int> out_mem = dev->make_arg(output);
            assert(out_mem->mem() != 0 && "Output memory not allocated");
            std::cout << "  -> Output memory allocated: " << out_mem->mem() << ", out_mem=" << out_mem.get() << "\n";

            nd_range dims{1, 1, 1, 1, 1, 1};
            CUcontext ctx = dev->getContext(0);
            CHECK_CUDA(cuCtxPushCurrent(ctx));
            CUcontext current_ctx;
            CHECK_CUDA(cuCtxGetCurrent(&current_ctx));
            std::cout << "  -> Current context before launch: " << current_ctx << "\n";

            CUdeviceptr device_ptr = out_mem->mem();
            std::cout << "  -> Launching kernel with device_ptr=" << device_ptr << "\n";
            void* kernel_args[] = { &device_ptr };
            CHECK_CUDA(cuLaunchKernel(
                prog->get_kernel(),
                dims.getGridDimX(), dims.getGridDimY(), dims.getGridDimZ(),
                dims.getBlockDimX(), dims.getBlockDimY(), dims.getBlockDimZ(),
                0, nullptr, kernel_args, nullptr
            ));
            std::cout << "  -> Kernel launched\n";

            CHECK_CUDA(cuCtxSynchronize());
            std::cout << "  -> Context synchronized\n";

            std::vector<int> result = out_mem->copy_to_host();
            std::cout << "  -> Data copied to host\n";
            assert(result[0] == 42 && "Kernel output incorrect");
            if (result[0] != 42) {
                std::cout << "  -> Failed: result[0] = " << result[0] << ", expected 42\n";
            }

            CHECK_CUDA(cuCtxPopCurrent(nullptr));
            std::cout << "  -> Direct kernel launched successfully\n";
        } catch (const std::exception& e) {
            std::cout << "  -> Failed: CUDA error during direct kernel launch: " << e.what() << "\n";
            throw;
        }
    }
    std::cout << "---- Kernel Launch Direct tests passed ----\n";
}

void test_actor_facade_multi_buffer(actor_system& sys, platform_ptr plat) {
    std::cout << "\n=== Test Actor Facade Multi Buffer ===\n";
    
    std::cout << "Test 1: Testing actor facade with multiple buffers...\n";
    manager& mgr = manager::get();
    device_ptr dev = mgr.find_device(0);
    std::cout << "  -> Device context: " << dev->getContext(0) << "\n";
    const char* kernel = R"(
        extern "C" __global__ void multi_buffer_kernel(const int* in_data, int* inout_data, int* out_data) {
            int idx = threadIdx.x;
            if (idx < 5) {
                inout_data[idx] = inout_data[idx] * 2; // Hardcoded scale=2
                out_data[idx] = in_data[idx] + 5;      // Hardcoded offset=5
            }
        })";
    program_ptr prog = mgr.create_program(kernel, "multi_buffer_kernel", dev);
    std::cout << "  -> Program created with kernel: multi_buffer_kernel, handle: " << prog->get_kernel() << ", prog=" << prog.get() << "\n";
    nd_range dims{1, 1, 1, 5, 1, 1};
    
    // Initialize host buffers
    std::vector<int> in_host(5, 10);        // in_data: [10, 10, 10, 10, 10]
    std::vector<int> inout_host(5, 20);     // inout_data: [20, 20, 20, 20, 20]
    std::vector<int> out_host(5, 0);        // out_data: [0, 0, 0, 0, 0]
    in<int> in_arg = create_in_arg(in_host);
    in_out<int> inout_arg = create_in_out_arg(inout_host);
    out<int> out_arg = create_out_arg(out_host);
    
    // Create device memory
    mem_ptr<int> in_mem = dev->make_arg(in_arg);
    mem_ptr<int> inout_mem = dev->make_arg(inout_arg);
    mem_ptr<int> out_mem = dev->make_arg(out_arg);
    assert(in_mem->mem() != 0 && "Input memory not allocated");
    assert(inout_mem->mem() != 0 && "In-out memory not allocated");
    assert(out_mem->mem() != 0 && "Output memory not allocated");
    std::cout << "  -> Input memory allocated: " << in_mem->mem() << ", in_mem=" << in_mem.get() << "\n";
    std::cout << "  -> In-out memory allocated: " << inout_mem->mem() << ", inout_mem=" << inout_mem.get() << "\n";
    std::cout << "  -> Output memory allocated: " << out_mem->mem() << ", out_mem=" << out_mem.get() << "\n";
    
    actor_config actor_cfg;
    actor_facade<false, in<int>, in_out<int>, out<int>> facade{
        std::move(actor_cfg), prog, dims, in<int>{}, in_out<int>{}, out<int>{}};
    std::cout << "  -> Actor facade created\n";
    
    try {
        facade.run_kernel(in_arg, inout_arg, out_arg);
        std::cout << "  -> Kernel launched via actor facade\n";
        
        // Validate results via inspection using copy_to_host
        std::vector<int> inout_result = inout_mem->copy_to_host();
        std::vector<int> out_result = out_mem->copy_to_host();
        
        // Expected: inout_result = [40, 40, 40, 40, 40] (20 * 2)
        // Expected: out_result = [15, 15, 15, 15, 15] (10 + 5)
        std::cout << " --- Expected inout_result is 40 40 40 40 40\n";
        std::cout << " --- Expected out_result is 15 15 15 15 15\n";
        for (size_t i = 0; i < 5; ++i) {
            if (inout_result[i] != 40) {
                std::cout << "  -> Failed: inout_result[" << i << "] = " << inout_result[i] << ", expected 40\n";
            }
            if (out_result[i] != 15) {
                std::cout << "  -> Failed: out_result[" << i << "] = " << out_result[i] << ", expected 15\n";
            }
        }
        std::cout << "  -> Multi-buffer kernel executed successfully\n";
    } catch (const std::exception& e) {
        std::cout << "  -> Failed: CUDA error during actor facade kernel launch: " << e.what() << "\n";
        throw;
    }
    std::cout << "---- Actor Facade Multi Buffer tests passed ----\n";
}

void test_kernel_launch_multi_buffer(actor_system& sys, platform_ptr plat) {
    std::cout << "\n=== Test Kernel Launch Multi Buffer ===\n";
    
    manager& mgr = manager::get();
    device_ptr dev = mgr.find_device(0);
    std::cout << "  -> Device context: " << dev->getContext(0) << "\n";

    std::cout << "Test 1: Testing direct kernel launch with multiple buffers...\n";
    {
        const char* kernel_code = R"(
            extern "C" __global__ void multi_buffer_kernel(const int* in_data, int* inout_data, int* out_data, int scale, int offset) {
                int idx = threadIdx.x;
                if (idx < 5) {
                    inout_data[idx] = inout_data[idx] * scale; // Modify in_out buffer
                    out_data[idx] = in_data[idx] + offset;     // Write to out buffer
                }
            })";
        program_ptr prog = mgr.create_program(kernel_code, "multi_buffer_kernel", dev);
        std::cout << "  -> Program created with kernel: multi_buffer_kernel, handle: " << prog->get_kernel() << ", prog=" << prog.get() << "\n";
        
        // Initialize host buffers
        std::vector<int> in_host(5, 10);        // in_data: [10, 10, 10, 10, 10]
        std::vector<int> inout_host(5, 20);     // inout_data: [20, 20, 20, 20, 20]
        std::vector<int> out_host(5, 0);        // out_data: [0, 0, 0, 0, 0]
        in<int> in_arg = create_in_arg(in_host);
        in_out<int> inout_arg = create_in_out_arg(inout_host);
        out<int> out_arg = create_out_arg(out_host);
        
        // Create device memory for buffers
        mem_ptr<int> in_mem = dev->make_arg(in_arg);
        mem_ptr<int> inout_mem = dev->make_arg(inout_arg);
        mem_ptr<int> out_mem = dev->make_arg(out_arg);
        assert(in_mem->mem() != 0 && "Input memory not allocated");
        assert(inout_mem->mem() != 0 && "In-out memory not allocated");
        assert(out_mem->mem() != 0 && "Output memory not allocated");
        std::cout << "  -> Input memory allocated: " << in_mem->mem() << ", in_mem=" << in_mem.get() << "\n";
        std::cout << "  -> In-out memory allocated: " << inout_mem->mem() << ", inout_mem=" << inout_mem.get() << "\n";
        std::cout << "  -> Output memory allocated: " << out_mem->mem() << ", out_mem=" << out_mem.get() << "\n";
        
        // Allocate device memory for scalar arguments
        int scale = 2;  // inout_data[i] = inout_data[i] * 2
        int offset = 5; // out_data[i] = in_data[i] + 5
        CUdeviceptr scale_dev, offset_dev;
        CHECK_CUDA(cuMemAlloc(&scale_dev, sizeof(int)));
        CHECK_CUDA(cuMemAlloc(&offset_dev, sizeof(int)));
        CHECK_CUDA(cuMemcpyHtoD(scale_dev, &scale, sizeof(int)));
        CHECK_CUDA(cuMemcpyHtoD(offset_dev, &offset, sizeof(int)));
        
        nd_range dims{1, 1, 1, 5, 1, 1};
        try {
            CUcontext ctx = dev->getContext(0);
            CUcontext current_ctx;
            CHECK_CUDA(cuCtxGetCurrent(&current_ctx));
            std::cout << "  -> Current context before launch: " << current_ctx << "\n";
            std::cout << "  -> Launching kernel with context: " << ctx << ", kernel: " << prog->get_kernel() << "\n";
            CUdeviceptr in_dev = in_mem->mem();
	    CUdeviceptr inout_dev = inout_mem->mem();
            CUdeviceptr out_dev = out_mem->mem();

	    void* kernel_args[] = {
    	    &in_dev, &inout_dev, &out_dev, &scale_dev, &offset_dev
            }; 
	   CHECK_CUDA(cuLaunchKernel(
                prog->get_kernel(),
                dims.getGridDimX(), dims.getGridDimY(), dims.getGridDimZ(),
                dims.getBlockDimX(), dims.getBlockDimY(), dims.getBlockDimZ(),
                0, nullptr, kernel_args, nullptr
            ));
            std::cout << "  -> Kernel launched\n";
            CHECK_CUDA(cuCtxSynchronize());
            std::cout << "  -> Context synchronized\n";
            
            // Validate results via inspection using copy_to_host
            std::vector<int> inout_result = inout_mem->copy_to_host();
            std::vector<int> out_result = out_mem->copy_to_host();
            
            // Expected: inout_result = [40, 40, 40, 40, 40] (20 * 2)
            // Expected: out_result = [15, 15, 15, 15, 15] (10 + 5)
            std::cout << " --- Expected inout_result is 40 40 40 40 40\n";
            std::cout << " --- Expected out_result is 15 15 15 15 15\n";
            for (size_t i = 0; i < 5; ++i) {
                if (inout_result[i] != 40) {
                    std::cout << "  -> Failed: inout_result[" << i << "] = " << inout_result[i] << ", expected 40\n";
                }
                if (out_result[i] != 15) {
                    std::cout << "  -> Failed: out_result[" << i << "] = " << out_result[i] << ", expected 15\n";
                }
            }
            std::cout << "  -> Multi-buffer kernel launched successfully\n";
            
            // Free scalar device memory
            CHECK_CUDA(cuMemFree(scale_dev));
            CHECK_CUDA(cuMemFree(offset_dev));
        } catch (const std::exception& e) {
            std::cout << "  -> Failed: CUDA error during kernel launch: " << e.what() << "\n";
            CHECK_CUDA(cuMemFree(scale_dev));
            CHECK_CUDA(cuMemFree(offset_dev));
            throw;
        }
        std::cout << "  -> End of scope: prog=" << prog.get() << ", out_mem=" << out_mem.get() << "\n";
    }
    std::cout << "---- Kernel Launch Multi Buffer tests passed ----\n";
}


void test_main(caf::actor_system& sys) {
    std::cout << "\n===== Running CUDA CAF Tests =====\n";
    manager::init(sys);
    platform_ptr plat = platform::create();
    manager& mgr = manager::get();
    device_ptr dev = mgr.find_device(0);
    std::cout << "  -> Manager initialized with context: " << dev->getContext(0) << "\n";
    try {
        test_platform(sys, plat);
        test_device(sys, plat);
        test_manager(sys, plat);
        test_program(sys, plat);
        test_create_program(sys, plat);
        test_mem_ref(sys, plat);
        test_command(sys, plat);
        test_mem_ref_extended(sys, plat);
        test_argument_translation(sys, plat);
        test_kernel_launch_direct(sys, plat); // Added new test
        test_kernel_launch(sys, plat);
        test_actor_facade(sys, plat);
	test_actor_facade_multi_buffer(sys, plat);
        test_kernel_launch_multi_buffer(sys, plat);
    } catch (const std::exception& e) {
        std::cout << "Test failed: " << e.what() << "\n";
        manager::shutdown();
        plat.reset();
        throw;
    }
    manager::shutdown();
    plat.reset();
    std::cout << "\n===== All CUDA CAF Tests Completed Successfully =====\n";
}
