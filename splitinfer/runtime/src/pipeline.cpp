/* splitinfer/runtime/src/pipeline.cpp
 * Pipeline orchestrator: iterates manifest layers, dispatches to GPU/FPGA
 * executors, handles device-boundary transfers, and records telemetry.
 */

#include "splitinfer/pipeline.h"

#include <chrono>
#include <cstdio>
#include <unordered_map>
#include <vector>

namespace splitinfer {

using Clock = std::chrono::steady_clock;
using Ms    = std::chrono::duration<double, std::milli>;

static double elapsed_ms(Clock::time_point start) {
    return std::chrono::duration_cast<Ms>(Clock::now() - start).count();
}

Pipeline::Pipeline(const Manifest& manifest, GpuExecutorBase& gpu, FpgaExecutorBase& fpga)
    : manifest_(manifest), gpu_(gpu), fpga_(fpga) {}

bool Pipeline::run() {
    // Build set of transfer trigger points: after which layer a transfer fires.
    // Map: after_layer name → transfer index.
    struct XferTrigger {
        size_t xfer_idx;
    };
    std::unordered_multimap<std::string, size_t> xfer_map;
    for (size_t i = 0; i < manifest_.transfers.size(); ++i) {
        xfer_map.emplace(manifest_.transfers[i].after_layer, i);
    }

    auto pass_start = Clock::now();
    double gpu_ms   = 0.0;
    double fpga_ms  = 0.0;
    int64_t xfer_bytes = 0;

    // Scratch buffers — in a real implementation these would be pinned/device memory.
    // For orchestration purposes we use a single zeroed buffer reused for all layers.
    constexpr size_t SCRATCH_SIZE = 4 * 1024 * 1024; // 4 MiB
    static thread_local std::vector<uint8_t> input_buf(SCRATCH_SIZE, 0);
    static thread_local std::vector<uint8_t> output_buf(SCRATCH_SIZE, 0);

    bool all_ok = true;

    for (const auto& layer : manifest_.layers) {
        size_t in_bytes  = static_cast<size_t>(layer.output_tensor_bytes); // previous output size
        size_t out_bytes = static_cast<size_t>(layer.output_tensor_bytes);

        if (in_bytes  > input_buf.size())  input_buf.resize(in_bytes,  0);
        if (out_bytes > output_buf.size()) output_buf.resize(out_bytes, 0);

        bool layer_ok = false;

        if (layer.device == Device::GPU) {
            auto t0 = Clock::now();
            layer_ok = gpu_.execute(layer.name,
                                    input_buf.data(),  in_bytes,
                                    output_buf.data(), out_bytes);
            gpu_ms += elapsed_ms(t0);
        } else {
            // FPGA: NMC operation code 0 used as a generic placeholder.
            auto t0 = Clock::now();
            layer_ok = fpga_.execute(layer.name, /*nmc_op=*/0,
                                     input_buf.data(),  in_bytes,
                                     output_buf.data(), out_bytes);
            fpga_ms += elapsed_ms(t0);
        }

        if (!layer_ok) {
            std::fprintf(stderr, "Pipeline: layer '%s' executor returned error\n",
                         layer.name.c_str());
            all_ok = false;
        }

        // Device-boundary transfers triggered after this layer.
        auto range = xfer_map.equal_range(layer.name);
        for (auto it = range.first; it != range.second; ++it) {
            const auto& xfer = manifest_.transfers[it->second];
            xfer_bytes += xfer.tensor_bytes;

            if (xfer.from_device == Device::GPU && xfer.to_device == Device::FPGA) {
                // GPU → FPGA: ship output_buf contents to FPGA memory address 0.
                size_t len = std::min(static_cast<size_t>(xfer.tensor_bytes),
                                      output_buf.size());
                fpga_.transfer_to_fpga(output_buf.data(), len, /*fpga_addr=*/0u);
            } else if (xfer.from_device == Device::FPGA && xfer.to_device == Device::GPU) {
                // FPGA → GPU: retrieve from FPGA memory into input_buf.
                size_t len = std::min(static_cast<size_t>(xfer.tensor_bytes),
                                      input_buf.size());
                fpga_.transfer_to_host(/*fpga_addr=*/0u, len, input_buf.data());
            }
        }
    }

    double total_ms = elapsed_ms(pass_start);
    telemetry_.record_inference(total_ms, gpu_ms, fpga_ms, xfer_bytes);

    return all_ok;
}

} // namespace splitinfer
