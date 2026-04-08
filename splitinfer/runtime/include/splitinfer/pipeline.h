#ifndef SPLITINFER_PIPELINE_H
#define SPLITINFER_PIPELINE_H

#include "splitinfer/manifest.h"
#include "splitinfer/gpu_executor.h"
#include "splitinfer/fpga_executor.h"
#include "splitinfer/telemetry.h"

namespace splitinfer {

/// Orchestrates split inference across GPU and FPGA using a parsed Manifest.
///
/// Execution model:
///   With prefetch enabled (default), the pipeline overlaps each cross-device
///   transfer with the *previous* layer's compute.  Concretely, after layer N
///   finishes, the pipeline issues N's outbound transfer (if any) BEFORE
///   calling layer N+1's execute().  When layer N+1 lives on the destination
///   device, its execute() will pick up the data already pre-staged.  This
///   implements the design spec § 5.4 "double-buffered pipelining" using two
///   rotating scratch buffers (input_buf / output_buf swap roles each iter).
///
///   With prefetch disabled (SPLITINFER_NO_PREFETCH=1 or SPLITINFER_NO_PIPELINE=1),
///   the pipeline runs strictly sequentially: execute -> transfer -> execute.
///   This is the configuration used by E4 ablation experiments to measure
///   the prefetch contribution.
class Pipeline {
public:
    Pipeline(const Manifest& manifest, GpuExecutorBase& gpu, FpgaExecutorBase& fpga);

    /// Enable or disable double-buffered prefetch.  Default: enabled.
    /// E4 ablation toggles this via SPLITINFER_NO_PREFETCH / SPLITINFER_NO_PIPELINE
    /// env vars (read by splitinfer_run.cpp at startup).
    void set_prefetch_enabled(bool enabled) { prefetch_enabled_ = enabled; }
    bool is_prefetch_enabled() const        { return prefetch_enabled_; }

    /// Execute one forward pass.  Returns true if all layers completed.
    bool run();

    const Telemetry& get_telemetry() const { return telemetry_; }
    Telemetry&       get_telemetry()       { return telemetry_; }

private:
    const Manifest&   manifest_;
    GpuExecutorBase&  gpu_;
    FpgaExecutorBase& fpga_;
    Telemetry         telemetry_;
    bool              prefetch_enabled_ = true;
};

} // namespace splitinfer
#endif
