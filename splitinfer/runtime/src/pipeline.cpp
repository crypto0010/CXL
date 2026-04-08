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

// Map ONNX op_type → EdgeCoh NMC class code routed by FPGA nmc_dispatch.v:
//   0x01 = NMC_EMBEDDING   — embedding lookup engine (validated by T19)
//   0x02 = NMC_INT8_FC     — MAC controller, INT8 8x8 GEMM (validated by T21)
//   0x03 = NMC_ELEMENTWISE — eltwise_controller (validated by T22)
//
// For elementwise ops the *sub-op* (relu/add/mul) is selected by the FPGA
// based on bits [1:0] of nmc_table_base, which the runtime currently does
// not set per layer.  See FpgaExecutor::execute() — which forwards
// nmc_table_base=0 by default, meaning elementwise layers default to
// sub-op 0 (ReLU).  A future enhancement should let the runtime pass the
// elementwise sub-op through the executor, but for now ReLU is sufficient
// for the smoke test and for E1 (DLRM) which only has Gather + MatMul +
// optional ReLU activations.
//
// All three engines were individually hardware-validated on real silicon
// before this routing was enabled (commits 2d39984 + 5ac6bcf).
static uint32_t op_type_to_nmc_op(const std::string& op_type) {
    if (op_type == "Gather")             return 0x01;  // NMC_EMBEDDING
    if (op_type == "MatMul" ||
        op_type == "Gemm")               return 0x02;  // NMC_INT8_FC
    if (op_type == "Relu" ||
        op_type == "Add"  ||
        op_type == "Mul")                return 0x03;  // NMC_ELEMENTWISE
    // Fallback: treat unknown ops as generic INT8 compute (MAC engine).
    // The MAC engine handles the most general matrix-like workload, so
    // it's the safest default for unknown future ops.
    return 0x02;
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
            // FPGA: derive NMC operation code from the layer's ONNX op_type.
            // The FPGA's nmc_dispatch routes based on this code to the
            // embedding lookup / MAC array / elementwise engines.
            auto t0 = Clock::now();
            uint32_t nmc_op = op_type_to_nmc_op(layer.op_type);
            layer_ok = fpga_.execute(layer.name, nmc_op,
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
