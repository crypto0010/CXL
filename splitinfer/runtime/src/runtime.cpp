/* splitinfer/runtime/src/runtime.cpp
 * Main runtime orchestrator: owns manifest, executors, and pipeline.
 */

#include "splitinfer/runtime.h"

#include <cstdio>

namespace splitinfer {

Runtime::Runtime() = default;
Runtime::~Runtime() = default;
Runtime::Runtime(Runtime&&) noexcept = default;
Runtime& Runtime::operator=(Runtime&&) noexcept = default;

bool Runtime::init(const std::string& manifest_path,
                   GpuExecutorBase& gpu,
                   FpgaExecutorBase& fpga) {
    if (!load_manifest(manifest_path, manifest_)) {
        std::fprintf(stderr, "Runtime: failed to load manifest from '%s'\n",
                     manifest_path.c_str());
        return false;
    }

    std::fprintf(stdout, "Runtime: loaded manifest v%s — %d layers (%d GPU, %d FPGA)\n",
                 manifest_.version.c_str(),
                 manifest_.summary.total_layers,
                 manifest_.summary.gpu_layers,
                 manifest_.summary.fpga_layers);

    pipeline_ = std::make_unique<Pipeline>(manifest_, gpu, fpga);
    initialized_ = true;
    return true;
}

bool Runtime::run_inference() {
    if (!initialized_ || !pipeline_) {
        std::fprintf(stderr, "Runtime: not initialized — call init() first\n");
        return false;
    }
    return pipeline_->run();
}

const Telemetry& Runtime::get_telemetry() const {
    return pipeline_->get_telemetry();
}

Telemetry& Runtime::get_telemetry() {
    return pipeline_->get_telemetry();
}

const Manifest& Runtime::get_manifest() const { return manifest_; }

} // namespace splitinfer
