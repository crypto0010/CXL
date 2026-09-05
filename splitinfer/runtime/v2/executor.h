/* Execution modes over one lowered program.
 *
 *   HostExecutor  weights resident in host memory; integer kernels on the
 *                 CPU.  Stand-in for the "gpu" placement's data path (the
 *                 GPU FP16 path is measured separately with TensorRT).
 *   NmcExecutor   weights resident in FPGA DDR2; NMC_EXEC per layer over
 *                 EdgeCoh; only inputs and the final result cross the link.
 *                 Compute moves to memory.
 *   PoolExecutor  weights resident in FPGA DDR2; host computes through a
 *                 cxlwin load/store window, faulting pages in on demand.
 *                 Memory moves to compute.
 *
 * All three produce the same INT32 output vector for the same input, which
 * the CLI checks against the lowering's expected values.
 */
#ifndef SPLITINFER_V2_EXECUTOR_H
#define SPLITINFER_V2_EXECUTOR_H
#include "program.h"
#include <cstdint>
#include <map>
#include <memory>
#include <string>
#include <vector>

struct edgecoh_transport;

namespace splitinfer2 {

struct LayerTiming { std::string name, kind; double ms = 0; };
struct RunMetrics {
    double total_ms = 0, input_ms = 0, output_ms = 0;
    std::vector<LayerTiming> layers;
    uint64_t link_bytes_out = 0, link_bytes_in = 0, link_msgs = 0;
    uint64_t faults = 0, pages_fetched = 0, fetch_calls = 0;
};

class Executor {
public:
    virtual ~Executor() = default;
    virtual const char* mode() const = 0;
    virtual bool prepare(const Program& p, std::string* err) = 0;
    /* raw input bytes per input name (INT8 dense / INT32 index), as stored in vectors.bin */
    virtual bool run(const Program& p, const std::map<std::string, std::vector<uint8_t>>& in,
                     std::vector<int32_t>& out, RunMetrics& m) = 0;
    /* Called between iterations to model a cold pool (drop host copies). */
    virtual void reset_between_iterations() {}
};

/* Integer kernels shared by Host and Pool: operate on pointers, so Pool can
 * hand them pointers INTO the cxlwin window. */
namespace kernels {
void gather(const int8_t* table, int dim_bytes, int dim, int32_t idx, int8_t* out);
void fc_acc(const int8_t* W, int M_pad, int K_pad, const int8_t* x, int32_t* acc);
void epilogue(const int32_t* acc, const int32_t* bias, int M_pad, int mult, int shift, int relu, int8_t* out);
}

std::unique_ptr<Executor> make_host_executor();
std::unique_ptr<Executor> make_nmc_executor(edgecoh_transport* t);
std::unique_ptr<Executor> make_pool_executor(edgecoh_transport* t, unsigned prefetch_pages, bool warm);

} // namespace
#endif
