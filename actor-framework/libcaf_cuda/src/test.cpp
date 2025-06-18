#include <iostream>
#include <vector>
#include <cassert>
#include <stdexcept>
#include <caf/all.hpp>
#include "caf/cuda/manager.hpp"
#include "caf/cuda/program.hpp"
#include "caf/cuda/device.hpp"
#include "caf/cuda/command.hpp"
#include "caf/cuda/actor_facade.hpp"
#include "caf/cuda/mem_ref.hpp"
#include "caf/cuda/platform.hpp"

using namespace caf;
using namespace caf::cuda;

// Helper function to check CUDA errors
void check_cuda_error(CUresult result, const char* operation) {
    if (result != CUDA_SUCCESS) {
        const char* error_string;
        cuGetErrorString(result, &error_string);
        std::cerr << "CUDA Error during " << operation << ": " << error_string << std::endl;
        assert(false);
    }
}

// ### Test Cases for platform.hpp
void test_platform(actor_system& sys) {
    // Test 1: Platform creation and basic properties
    platform_ptr plat = platform::create();
    assert(plat != nullptr && "Platform should be created successfully");
    assert(!plat->devices().empty() && "Platform should have at least one device");
    std::cout << "Platform creation test passed.\n";

    // Test 2: Device retrieval by ID (valid case)
    device_ptr dev = plat->getDevice(0);
    assert(dev != nullptr && "Device 0 should exist");
    assert(dev->getId() == 0 && "Device ID should match requested ID");

    // Test 3: Device retrieval with invalid ID (equivalence class: out-of-bounds)
    try {
        plat->getDevice(-1); // Should throw
        assert(false && "Expected exception for negative device ID");
    } catch (const std::exception&) {
        std::cout << "Platform invalid device ID test passed.\n";
    }
}

// ### Test Cases for device.hpp
void test_device(actor_system& sys) {
    platform_ptr plat = platform::create();
    device_ptr dev = plat->getDevice(0);

    // Test 1: Basic device properties
    assert(dev->getDevice() != 0 && "Device handle should be valid");
    assert(dev->getContext() != nullptr && "Context should be valid");
    assert(dev->getStream() != nullptr && "Stream should be valid");
    assert(dev->name() != nullptr && "Device should have a name");
    std::cout << "Device properties test passed.\n";

    // Test 2: Memory argument creation (in, in_out, out equivalence classes)
    std::vector<int> data(5, 42);
    in<int> input{data};
    mem_ptr<int> in_mem = dev->make_arg(input);
    assert(in_mem->size() == 5 && "Input memory size should match");
    assert(in_mem->access() == IN && "Access type should be IN");

    in_out<int> inout{data};
    mem_ptr<int> inout_mem = dev->make_arg(inout);
    assert(inout_mem->access() == IN_OUT && "Access type should be IN_OUT");

    out<int> output{std::vector<int>(5)};
    mem_ptr<int> out_mem = dev->make_arg(output);
    assert(out_mem->access() == OUT && "Access type should be OUT");
    std::cout << "Device memory argument test passed.\n";
}

// ### Test Cases for manager.hpp and manager.cpp
void test_manager(actor_system& sys) {
    // Test 1: Manager initialization and singleton access
    manager::init(sys);
    manager& mgr = manager::get();
    assert(&mgr == &manager::get() && "Manager should be singleton");
    std::cout << "Manager initialization test passed.\n";

    // Test 2: Device lookup
    device_ptr dev = mgr.find_device(0);
    assert(dev != nullptr && "Manager should find device 0");
    assert(dev->getId() == 0 && "Device ID should be 0");

    // Test 3: Program creation with kernel
    const char* kernel = R"(
        extern "C" __global__ void test_kernel(int* data) {
            int idx = threadIdx.x;
            data[idx] = idx;
        }
    )";
    program_ptr prog = mgr.create_program(kernel, "test_kernel", dev);
    assert(prog != nullptr && "Program should be created");
    assert(prog->get_kernel() != nullptr && "Kernel should be loaded");
    std::cout << "Manager program creation test passed.\n";

    // Test 4: Invalid device ID (equivalence class)
    try {
        mgr.find_device(999); // Likely invalid
        assert(false && "Expected exception for invalid device ID");
    } catch (const std::exception&) {
        std::cout << "Manager invalid device test passed.\n";
    }

    manager::shutdown(); // Cleanup
}

// ### Test Cases for program.hpp
void test_program(actor_system& sys) {
    manager::init(sys);
    manager& mgr = manager::get();
    device_ptr dev = mgr.find_device(0);

    // Test 1: Program creation and properties
    const char* kernel = R"(
        extern "C" __global__ void test_kernel(int* data) {
            data[threadIdx.x] = threadIdx.x;
        }
    )";
    program_ptr prog = mgr.create_program(kernel, "test_kernel", dev);
    assert(prog->get_device_id() == 0 && "Device ID should match");
    assert(prog->get_context_id() == 0 && "Context ID should be 0");
    assert(prog->get_stream_id() == 0 && "Stream ID should be 0");
    std::cout << "Program properties test passed.\n";

    manager::shutdown();
}

// ### Test Cases for mem_ref.hpp
void test_mem_ref(actor_system& sys) {
    manager::init(sys);
    manager& mgr = manager::get();
    device_ptr dev = mgr.find_device(0);

    // Test 1: Input memory reference
    std::vector<int> host_data(5, 10);
    in<int> input{host_data};
    mem_ptr<int> mem = dev->make_arg(input);
    assert(mem->size() == 5 && "Memory size should match input");
    assert(mem->mem() != 0 && "Device memory should be allocated");
    assert(mem->access() == IN && "Access should be IN");

    // Test 2: Output memory reference and copy
    out<int> output{std::vector<int>(5, 0)};
    mem_ptr<int> out_mem = dev->make_arg(output);
    assert(out_mem->access() == OUT && "Access should be OUT");
    std::vector<int> copied = out_mem->copy_to_host(); // Should succeed for OUT
    assert(copied.size() == 5 && "Copied data size should match");

    // Test 3: Invalid copy attempt (equivalence class: IN-only)
    try {
        mem->copy_to_host(); // Should throw since IN
        assert(false && "Expected exception for copying IN memory");
    } catch (const std::runtime_error&) {
        std::cout << "Mem_ref invalid copy test passed.\n";
    }

    // Test 4: Reset functionality
    mem->reset();
    assert(mem->size() == 0 && "Size should be 0 after reset");
    assert(mem->mem() == 0 && "Memory should be freed");
    std::cout << "Mem_ref reset test passed.\n";

    manager::shutdown();
}

// ### Test Cases for command.hpp and command.cpp
void test_command(actor_system& sys) {
    manager::init(sys);
    manager& mgr = manager::get();
    device_ptr dev = mgr.find_device(0);

    // Test 1: Command creation and enqueue
    const char* kernel = R"(
        extern "C" __global__ void test_kernel(int* data) {
            int idx = threadIdx.x;
            data[idx] = idx;
        }
    )";
    program_ptr prog = mgr.create_program(kernel, "test_kernel", dev);
    nd_range dims{1, 1, 1, 5, 1, 1}; // 5 threads
    std::vector<int> host_data(5, 0);
    out<int> output{host_data};
    response_promise rp = sys.dummy_response_promise();
    command<actor, out<int>> cmd{rp, prog, dims, output};
    cmd.enqueue(); // Should run without crashing
    std::vector<int> result = std::get<0>(cmd.mem_refs)->copy_to_host();
    for (int i = 0; i < 5; ++i) {
        assert(result[i] == i && "Kernel should write thread ID");
    }
    std::cout << "Command enqueue and output test passed.\n";

    // Test 2: Reference counting
    auto cmd_ptr = make_counted<command<actor, out<int>>>(rp, prog, dims, output);
    cmd_ptr->intrusive_ptr_add_ref(cmd_ptr.get());
    assert(cmd_ptr->ref_count == 1 && "Ref count should increment");
    cmd_ptr->intrusive_ptr_release(cmd_ptr.get());
    std::cout << "Command ref counting test passed.\n";

    manager::shutdown();
}

// ### Test Cases for actor_facade.hpp
void test_actor_facade(actor_system& sys) {
    manager::init(sys);
    manager& mgr = manager::get();
    device_ptr dev = mgr.find_device(0);

    // Test 1: Actor creation and kernel execution
    const char* kernel = R"(
        extern "C" __global__ void test_kernel(int* data) {
            int idx = threadIdx.x;
            data[idx] = idx + 1;
        }
    )";
    program_ptr prog = mgr.create_program(kernel, "test_kernel", dev);
    nd_range dims{1, 1, 1, 5, 1, 1};
    std::vector<int> host_data(5, 0);
    out<int> output{host_data};
    actor_config actor_cfg;
    actor_facade<false, out<int>> facade{std::move(actor_cfg), prog, dims};
    facade.run_kernel(output);
    std::vector<int> result = output.buffer;
    for (int i = 0; i < 5; ++i) {
        assert(result[i] == i + 1 && "Kernel should write thread ID + 1");
    }
    std::cout << "Actor_facade kernel execution test passed.\n";

    // Test 2: Actor creation via system
    actor act = actor_facade<false, out<int>>::create(&sys, std::move(actor_cfg), prog, dims, output);
    assert(act != nullptr && "Actor should be created");
    std::cout << "Actor_facade creation test passed.\n";

    //manager::shutdown();
}

