#ifndef SPLITINFER_RUNTIME_H
#define SPLITINFER_RUNTIME_H

#include "splitinfer/manifest.h"
#include "splitinfer/pipeline.h"
#include "splitinfer/gpu_executor.h"
#include "splitinfer/fpga_executor.h"
#include "splitinfer/telemetry.h"

#include <memory>
#include <string>

namespace splitinfer {

/// Main runtime orchestrator: owns manifest, executors, and pipeline.
class Runtime {
public:
    Runtime();
    ~Runtime();

    // Non-copyable, movable.
    Runtime(const Runtime&) = delete;
    Runtime& operator=(const Runtime&) = delete;
    Runtime(Runtime&&) noexcept;
    Runtime& operator=(Runtime&&) noexcept;

    /// Load manifest and initialize executors and pipeline.
    /// @param manifest_path  Path to partition manifest JSON file.
    /// @param gpu            GPU executor implementation (caller retains ownership).
    /// @param fpga           FPGA executor implementation (caller retains ownership).
    /// @return true on success.
    bool init(const std::string& manifest_path,
              GpuExecutorBase& gpu,
              FpgaExecutorBase& fpga);

    /// Execute one forward-pass inference.
    /// @return true if all layers completed successfully.
    bool run_inference();

    /// Read-only access to telemetry.
    const Telemetry& get_telemetry() const;

    /// Mutable access to telemetry (e.g. for reset).
    Telemetry& get_telemetry();

    /// Read-only access to the loaded manifest.
    const Manifest& get_manifest() const;

private:
    Manifest                   manifest_;
    std::unique_ptr<Pipeline>  pipeline_;
    bool                       initialized_ = false;
};

} // namespace splitinfer
#endif
