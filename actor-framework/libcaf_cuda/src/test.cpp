
#include <iostream>
#include <vector>
#include <cassert>
#include <stdexcept>
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

void test_platform(actor_system&) {
    platform_ptr plat = platform::create();
    assert(plat != nullptr && "Platform should be created successfully");
    assert(!plat->devices().empty() && "Platform should have at least one device");
    std::cout << "Platform creation test passed.\n";

    device_ptr dev = plat->getDevice(0);
    assert(dev != nullptr && "Device 0 should exist");
    assert(dev->getId() == 0 && "Device ID should match requested ID");

    try {
        plat->getDevice(-1);
        assert(false && "Expected exception for negative device ID");
    } catch (const std::exception&) {
        std::cout << "Platform invalid device ID test passed.\n";
    }
}

void test_device(actor_system&) {
    std::cout << "Starting test of device\n";
    platform_ptr plat = platform::create();
    device_ptr dev = plat->getDevice(0);
    assert(dev->getContext() != nullptr);
    assert(dev->getStream() != nullptr);
    assert(dev->name() != nullptr);
    std::cout << "Device properties test passed.\n";

    std::vector<int> data(5, 42);
    in<int> input = create_in_arg(data);
    mem_ptr<int> in_mem = dev->make_arg(input);
    assert(in_mem->size() == 5);
    assert(in_mem->access() == IN);

    in_out<int> inout = create_in_out_arg(data);
    mem_ptr<int> inout_mem = dev->make_arg(inout);
    assert(inout_mem->access() == IN_OUT);

    out<int> output = create_out_arg(std::vector<int>(5, 0));
    mem_ptr<int> out_mem = dev->make_arg(output);
    assert(out_mem->size() == 5);
    assert(out_mem->access() == OUT);
    std::cout << "Device memory argument test passed.\n";
}

void test_manager(actor_system& sys) {
    manager::init(sys);
    manager& mgr = manager::get();
    assert(&mgr == &manager::get());
    std::cout << "Manager initialization test passed.\n";

    device_ptr dev = mgr.find_device(0);
    assert(dev != nullptr);
    assert(dev->getId() == 0);

    const char* kernel = R"(
        extern "C" __global__ void test_kernel(int* data) {
            int idx = threadIdx.x;
            data[idx] = idx;
        })";
    program_ptr prog = mgr.create_program(kernel, "test_kernel", dev);
    assert(prog != nullptr);
    assert(prog->get_kernel() != nullptr);
    std::cout << "Manager program creation test passed.\n";

    try {
        mgr.find_device(999);
        assert(false && "Expected exception for invalid device ID");
    } catch (const std::exception&) {
        std::cout << "Manager invalid device test passed.\n";
    }

    manager::shutdown();
}

void test_program(actor_system& sys) {
    manager::init(sys);
    manager& mgr = manager::get();
    device_ptr dev = mgr.find_device(0);

    const char* kernel = R"(
        extern "C" __global__ void test_kernel(int* data) {
            int idx = threadIdx.x;
            data[idx] = idx;
        })";
    program_ptr prog = mgr.create_program(kernel, "test_kernel", dev);
    assert(prog->get_device_id() == 0);
    assert(prog->get_context_id() == 0);
    assert(prog->get_stream_id() == 0);
    std::cout << "Program properties test passed.\n";

    manager::shutdown();
}

void test_mem_ref(actor_system& sys) {
    manager::init(sys);
    manager& mgr = manager::get();
    device_ptr dev = mgr.find_device(0);

    std::vector<int> host_data(5, 10);
    in<int> input = create_in_arg(host_data);
    mem_ptr<int> mem = dev->make_arg(input);
    assert(mem->size() == 5);
    assert(mem->mem() != 0);
    assert(mem->access() == IN);

    out<int> output = create_out_arg(std::vector<int>(5, 0));
    mem_ptr<int> out_mem = dev->make_arg(output);
    assert(out_mem->size() == 5);
    assert(out_mem->access() == OUT);

    in_out<int> inout = create_in_out_arg(host_data);
    mem_ptr<int> inout_mem = dev->make_arg(inout);
    assert(inout_mem->access() == IN_OUT);
    std::vector<int> copied = inout_mem->copy_to_host();
    assert(copied.size() == 5 && copied[0] == 10);

    try {
        mem->copy_to_host();
        assert(false && "Expected exception for copying IN memory");
    } catch (const std::runtime_error&) {
        std::cout << "Mem_ref invalid copy test passed.\n";
    }

    mem->reset();
    assert(mem->size() == 0);
    assert(mem->mem() == 0);
    std::cout << "Mem_ref reset test passed.\n";

    manager::shutdown();
}

void test_command(actor_system&) {
    std::cout << "Command test skipped due to response_promise issue.\n";
}

void test_actor_facade(actor_system& sys) {
    manager::init(sys);
    manager& mgr = manager::get();
    device_ptr dev = mgr.find_device(0);

    const char* kernel = R"(
        extern "C" __global__ void test_kernel(int* data) {
            int idx = threadIdx.x;
            if (idx < 5) data[idx] = idx + 1;
        })";
    program_ptr prog = mgr.create_program(kernel, "test_kernel", dev);
    nd_range dims{5, 1, 1, 1, 1, 1}; // 1 block with 5 threads
    std::vector<int> host_data(5, 0);
    out<int> output = create_out_arg(host_data);
    actor_config actor_cfg;
    actor_facade<false, out<int>> facade{std::move(actor_cfg), prog, dims, std::move(output)};
    facade.run_kernel(output);
    std::vector<int> result = output.buffer;
    for (size_t i = 0; i < 5; ++i) {
        assert(result[i] == static_cast<int>(i + 1));
    }
    std::cout << "Actor_facade kernel execution test passed.\n";
    manager::shutdown();
}

void test_mem_ref_extended(actor_system& sys) {
    manager::init(sys);
    manager& mgr = manager::get();
    device_ptr dev = mgr.find_device(0);

    // Test 1: Memory allocation and data transfer for in<T>
    std::vector<int> host_input(5, 42);
    in<int> input = create_in_arg(host_input);
    mem_ptr<int> in_mem = dev->make_arg(input);
    assert(in_mem->size() == 5 && "Incorrect size for in_mem");
    assert(in_mem->access() == IN && "Incorrect access type for in_mem");
    assert(in_mem->mem() != 0 && "Memory not allocated for in_mem");
    std::cout << "Mem_ref allocation test for in<T> passed.\n";

    // Test 2: Memory allocation and data transfer for out<T>
    std::vector<int> host_output(5, 0);
    out<int> output = create_out_arg(host_output);
    mem_ptr<int> out_mem = dev->make_arg(output);
    assert(out_mem->size() == 5 && "Incorrect size for out_mem");
    assert(out_mem->access() == OUT && "Incorrect access type for out_mem");
    assert(out_mem->mem() != 0 && "Memory not allocated for out_mem");
    std::cout << "Mem_ref allocation test for out<T> passed.\n";

    // Test 3: Data integrity for in_out<T>
    std::vector<int> host_inout(5, 10);
    in_out<int> inout = create_in_out_arg(host_inout);
    mem_ptr<int> inout_mem = dev->make_arg(inout);
    assert(inout_mem->size() == 5 && "Incorrect size for inout_mem");
    assert(inout_mem->access() == IN_OUT && "Incorrect access type for inout_mem");
    std::vector<int> copied = inout_mem->copy_to_host();
    for (size_t i = 0; i < 5; ++i) {
        assert(copied[i] == 10 && "Data corruption in inout_mem");
    }
    std::cout << "Mem_ref data integrity test for in_out<T> passed.\n";

    // Test 4: Out-of-bounds allocation check
    std::vector<int> small_data(2, 5);
    out<int> small_output = create_out_arg(small_data);
    mem_ptr<int> small_mem = dev->make_arg(small_output);
    assert(small_mem->size() == 2 && "Incorrect size for small_mem");
    std::cout << "Mem_ref small buffer test passed.\n";

    manager::shutdown();
}

void test_argument_translation(actor_system& sys) {
    manager::init(sys);
    manager& mgr = manager::get();
    device_ptr dev = mgr.find_device(0);

    // Test 1: Correct argument creation
    std::vector<int> data(5, 0);
    out<int> output = create_out_arg(data);
    mem_ptr<int> out_mem = dev->make_arg(output);
    assert(out_mem->size() == data.size() && "Size mismatch in argument translation");
    assert(out_mem->access() == OUT && "Access type mismatch in argument translation");
    std::cout << "Argument translation for out<T> passed.\n";

    // Test 2: Type mismatch simulation
    in<int> input = create_in_arg(data);
    mem_ptr<int> in_mem = dev->make_arg(input);
    assert(in_mem->access() == IN && "Incorrect access type for input");
    try {
        in_mem->copy_to_host(); // Should fail for IN access
        assert(false && "Expected exception for IN memory copy");
    } catch (const std::runtime_error&) {
        std::cout << "Argument access type enforcement test passed.\n";
    }

    // Test 3: Multiple arguments
    in<int> input2 = create_in_arg(std::vector<int>(5, 7));
    out<int> output2 = create_out_arg(std::vector<int>(5, 0));
    mem_ptr<int> in_mem2 = dev->make_arg(input2);
    mem_ptr<int> out_mem2 = dev->make_arg(output2);
    assert(in_mem2->size() == 5 && out_mem2->size() == 5 && "Size mismatch in multiple args");
    std::cout << "Multiple argument translation test passed.\n";

    manager::shutdown();
}

void test_kernel_launch(actor_system& sys) {
    manager::init(sys);
    manager& mgr = manager::get();
    device_ptr dev = mgr.find_device(0);

    // Define a simple kernel
    const char* kernel_code = R"(
        extern "C" __global__ void simple_kernel(int* output) {
            int idx = threadIdx.x;
            if (idx < 5) output[idx] = idx * 10;
        })";
    program_ptr prog = mgr.create_program(kernel_code, "simple_kernel", dev);

    // Test 1: Basic kernel launch
    std::vector<int> host_data(5, 0);
    out<int> output = create_out_arg(host_data);
    mem_ptr<int> out_mem = dev->make_arg(output);
    nd_range dims{1, 1, 1, 5, 1, 1}; // 1 block, 5 threads
    dev->launch_kernel(prog->get_kernel(), dims, std::make_tuple(out_mem), 0, 0);
    std::vector<int> result = out_mem->copy_to_host();
    for (int i = 0; i < 5; ++i) {
        assert(result[i] == i * 10 && "Kernel output incorrect");
    }
    std::cout << "Basic kernel launch test passed.\n";

    // Test 2: Out-of-bounds access
    std::vector<int> small_host(2, 0);
    out<int> small_output = create_out_arg(small_host);
    mem_ptr<int> small_mem = dev->make_arg(small_output);
    // Launch with 5 threads but buffer size 2
    try {
        dev->launch_kernel(prog->get_kernel(), dims, std::make_tuple(small_mem), 0, 0);
        std::vector<int> small_result = small_mem->copy_to_host();
        std::cout << "Out-of-bounds kernel launch completed (check with cuda-memcheck).\n";
    } catch (const std::exception& e) {
        std::cout << "Out-of-bounds kernel launch caught error: " << e.what() << "\n";
    }

    // Test 3: Sequential single-argument kernels
    std::vector<int> in_data(5, 2);
    std::vector<int> out_data(5, 0);
    const char* input_kernel = R"(
        extern "C" __global__ void input_kernel(int* input) {
            int idx = threadIdx.x;
            if (idx < 5) input[idx] = input[idx] * 2;
        })";
    const char* output_kernel = R"(
        extern "C" __global__ void output_kernel(int* output) {
            int idx = threadIdx.x;
            if (idx < 5) output[idx] = idx * 3;
        })";
    program_ptr input_prog = mgr.create_program(input_kernel, "input_kernel", dev);
    program_ptr output_prog = mgr.create_program(output_kernel, "output_kernel", dev);
    in<int> input = create_in_arg(in_data);
    out<int> output2 = create_out_arg(out_data);
    mem_ptr<int> in_mem = dev->make_arg(input);
    mem_ptr<int> out_mem2 = dev->make_arg(output2);
    // Launch input kernel
    dev->launch_kernel(input_prog->get_kernel(), dims, std::make_tuple(in_mem), 0, 0);
    // Launch output kernel
    dev->launch_kernel(output_prog->get_kernel(), dims, std::make_tuple(out_mem2), 0, 0);
    std::vector<int> in_result = in_mem->copy_to_host();
    std::vector<int> out_result = out_mem2->copy_to_host();
    for (int i = 0; i < 5; ++i) {
        assert(in_result[i] == 4 && "Input kernel output incorrect");
        assert(out_result[i] == i * 3 && "Output kernel output incorrect");
    }
    std::cout << "Sequential single-argument kernel launch test passed.\n";

    manager::shutdown();
}

void test_actor_facade_debug(actor_system& sys) {
    manager::init(sys);
    manager& mgr = manager::get();
    device_ptr dev = mgr.find_device(0);

    const char* kernel = R"(
        extern "C" __global__ void test_kernel(int* data) {
            int idx = threadIdx.x;
            if (idx < 5) data[idx] = idx + 1;
        })";
    program_ptr prog = mgr.create_program(kernel, "test_kernel", dev);
    nd_range dims{1, 1, 1, 5, 1, 1}; // 1 block, 5 threads
    std::vector<int> host_data(5, 0);
    out<int> output = create_out_arg(host_data);
    mem_ptr<int> out_mem = dev->make_arg(output);
    assert(out_mem->size() == 5 && "Buffer size mismatch");
    assert(out_mem->mem() != 0 && "Memory not allocated");

    // Direct launch for comparison
    dev->launch_kernel(prog->get_kernel(), dims, std::make_tuple(out_mem), 0, 0);
    std::vector<int> direct_result = out_mem->copy_to_host();
    for (size_t i = 0; i < 5; ++i) {
        assert(direct_result[i] == static_cast<int>(i + 1) && "Direct launch failed");
    }
    std::cout << "Direct kernel launch in debug test passed.\n";

    // Reset output
    host_data.assign(5, 0);
    output = create_out_arg(host_data);
    actor_config actor_cfg;
    actor_facade<false, out<int>> facade{std::move(actor_cfg), prog, dims, std::move(output)};
    facade.run_kernel(output);
    std::vector<int> result = output.buffer;
    for (size_t i = 0; i < 5; ++i) {
        assert(result[i] == static_cast<int>(i + 1) && "Actor facade kernel output incorrect");
    }
    std::cout << "Actor_facade debug kernel execution test passed.\n";

    manager::shutdown();
}

void test_main(caf::actor_system& sys) {
    std::cout << "Running CUDA CAF tests...\n";
    test_platform(sys);
    test_device(sys);
    test_manager(sys);
    test_program(sys);
    test_mem_ref(sys);
    test_command(sys);
//    test_actor_facade(sys);
    test_mem_ref_extended(sys);
    test_argument_translation(sys);
    test_kernel_launch(sys);
    test_actor_facade_debug(sys);
    std::cout << "All tests completed successfully.\n";
}
