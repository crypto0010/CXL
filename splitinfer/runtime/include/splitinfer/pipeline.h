#ifndef SPLITINFER_PIPELINE_H
#define SPLITINFER_PIPELINE_H

#include "splitinfer/manifest.h"
#include "splitinfer/gpu_executor.h"
#include "splitinfer/fpga_executor.h"
#include "splitinfer/telemetry.h"

namespace splitinfer {

/// Orchestrates split inference across GPU and FPGA using a parsed Manifest.
class Pipeline {
public:
    /// Construct a pipeline.
    /// @param manifest  Parsed partition manifest (owned externally, must outlive Pipeline).
    /// @param gpu       GPU executor implementation.
    /// @param fpga      FPGA executor implementation.
    Pipeline(const Manifest& manifest, GpuExecutorBase& gpu, FpgaExecutorBase& fpga);

    /// Execute one forward pass.
    /// Iterates layers in manifest order, dispatches each to the appropriate
    /// executor, and performs device-to-device data transfers at boundary points.
    /// @return true if all layers completed without error.
    bool run();

    /// Return the telemetry accumulator (read-only).
    const Telemetry& get_telemetry() const { return telemetry_; }

    /// Return the telemetry accumulator (mutable, e.g. to call reset()).
    Telemetry& get_telemetry() { return telemetry_; }

private:
    const Manifest&   manifest_;
    GpuExecutorBase&  gpu_;
    FpgaExecutorBase& fpga_;
    Telemetry         telemetry_;
};

} // namespace splitinfer
#endif
