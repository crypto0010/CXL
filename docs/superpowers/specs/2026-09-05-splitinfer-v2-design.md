# SplitInfer v2 — Design Specification

**Date:** 2026-09-05
**Status:** Approved direction; implementation in progress
**Supersedes:** `2026-04-03-splitinfer-design.md`
**Motivation:** Rejection of the IEEE TC submission. Three reviewers converged on
three charges: (1) the CXL framing is not earned by the mechanism, (2) the
evaluation does not exercise the claimed mechanism end to end, (3) the
contribution beyond engineering effort is not articulated.

---

## 1. What the post-mortem found

Beyond the reviewers' comments, an audit of the implementation found the
headline results rest on code that does not do what the paper says.

| # | Defect | Location | Consequence |
|---|---|---|---|
| D1 | `output_tensor_bytes` hardcoded to 0 | `partitioner/graph.py:29` | Every manifest reports zero activation bytes and zero transfers |
| D2 | Runtime passes 0 bytes to every executor | `runtime/src/pipeline.cpp` | E2's 595.59 ms is 36 *empty* protocol round-trips — no operands, no results |
| D3 | `NMC_EXEC` sent with all address/shape fields zero | `runtime/src/fpga_executor.cpp:110` | The FPGA engines cannot have computed anything |
| D4 | "Dynamic programming solver" is a greedy single pass | `partitioner/solver.py:47` | No optimisation occurs; second branch forces every weighted layer to FPGA |
| D5 | Weight-download cost never priced despite docstring | `partitioner/solver.py` | Streaming capacity model has no latency consequence |
| D6 | Cost model default `usb_bw_mbps=40.0` vs 0.0115 real | `partitioner/cli.py` | 3,478x optimistic; basis of the 463.7 ms E1 estimate |
| D7 | All 8 MAC accumulators receive the same value | `fpga/src/mac_array_8x8.v:82` | It is an 8-wide dot product, not an 8x8 array |
| D8 | Output rows written 32 B wide at 4 B stride | `fpga/src/mac_controller.v` S_WRITE_RES | Consecutive rows overlap and corrupt each other |
| D9 | 180 s timeouts recorded as `status: "ok"` | `results/e2/e2_summary.json` | DLRM b64/b256 "measurements" are timeouts |
| D10 | Reported stats omit CV=250%, p99=11,238 ms | paper Table VII | Selective reporting of a pathological distribution |
| D11 | Power derived from 2 tegrastats samples | `results/e2` B1 FP32 | Baseline power figure is not a measurement |
| D12 | `prefetch_hits` incremented from a flag, not an observation | `runtime/src/pipeline.cpp:139` | Any prefetch ablation would measure nothing |

**Root cause of D1-D3:** a structurally optional dataflow field was never
populated and nothing validated it, so "missing size" silently degraded to
"transfer nothing" while telemetry still reported wall-clock time.

## 2. Thesis change

**Old thesis (rejected):** "We built a CXL-like coherent memory node at the edge."
Not earned — the host never issued a load or store to device memory.

**New thesis:** *When a model does not fit in edge SoC memory, you can either move
the memory to the compute or move the compute to the memory. Which wins is
determined by a per-layer ratio of arithmetic intensity to interconnect
bandwidth, and the crossover is predictable.* SplitInfer implements **both**
paths over one substrate and measures the crossover.

This reframes the 1,083x latency premium from an embarrassment into the
measured extreme point of a model that also predicts where the answer flips.

## 3. New mechanism: `cxlwin` — a host load/store window onto device memory

Answers R2#1(a), R3#1, R3#2, R3#6.

FPGA DDR2 is mapped into the host virtual address space. The host dereferences
ordinary pointers; page faults are serviced by EdgeCoh transfers. `userfaultfd`
is absent from the Tegra kernel (`ENOSYS`), so servicing uses `SIGSEGV` +
`mprotect` — the classic software-DSM technique, which needs no kernel feature
and therefore ports to any edge SoC.

### Page coherence states

Per 4 KiB page, with **transient states that track outstanding responses**
(directly answering R3#6):

| State | Protection | Meaning |
|---|---|---|
| `INVALID` | `PROT_NONE` | No valid host copy; device owns the data |
| `FETCHING` | `PROT_NONE` | *Transient* — a `DATA_READ` is outstanding |
| `SHARED` | `PROT_READ` | Host holds a clean copy; device copy still valid |
| `MODIFIED` | `PROT_READ\|PROT_WRITE` | Host copy dirty; device copy stale |
| `FLUSHING` | `PROT_READ` | *Transient* — a `DATA_WRITE` is outstanding |
| `DEVICE` | `PROT_NONE` | Device bias: an NMC engine owns the page |

Transitions: load/INVALID -> FETCHING -> SHARED; store/SHARED -> MODIFIED;
store/INVALID -> FETCHING -> MODIFIED; `release()` MODIFIED -> FLUSHING ->
DEVICE, SHARED -> DEVICE; `acquire()` DEVICE -> INVALID (refilled on demand).

This is the CXL.mem *bias model* (host bias vs device bias) at page rather than
cache-line granularity. The paper will state that distinction explicitly rather
than claiming equivalence.

### Backends
- `backend_edgecoh` — real FPGA over UART; faults become `DATA_READ`/`DATA_WRITE`.
  **Requires no RTL change**: the existing controller already implements both.
- `backend_emul` — in-process device emulator modelling DDR2 plus the three NMC
  engines with a configurable bandwidth/latency model. Makes the entire system
  testable, and every correctness claim verifiable, without the board.

## 4. The experiment that was missing

One workload, one manifest, two paths through the same substrate:

- **POOL mode** — FPGA DDR2 as pure memory. Weights fault into the host through
  `cxlwin`; compute runs on the GPU under TensorRT. *Memory moves to compute.*
- **NMC mode** — host transfers ownership to device bias and dispatches
  `NMC_EXEC`; only results fault back. *Compute moves to memory.*

Identical inputs, identical numerical output check. This is exactly the
comparison R3#5 asked for and R1#5 found missing, and it is the paper's result.

## 5. Component changes

### 5.1 Partitioner (`partitioner/`)
- `graph.py`: run ONNX shape inference; populate `output_tensor_bytes` for real.
  Fail loudly rather than defaulting to 0.
- `cost_model.py`: parameters calibrated from measurement, not defaults. Price
  per-layer **weight download** under the streaming model. Expose arithmetic
  intensity per layer.
- `solver.py`: replace greedy pass with a genuine **dynamic program** — Viterbi
  over (layer, device), cost = compute + boundary transfer + weight streaming,
  with backtrack. O(N x 2 x 2). Remove the branch that forces weighted layers
  onto the FPGA.

### 5.2 RTL (`fpga/src/`) — needs synthesis
- `mac_array_8x8.v`: eight independent accumulators; a true 8x8 array.
- `mac_controller.v`: correct output stride; eight output rows per pass.
- Golden-vector testbenches under Icarus Verilog.

### 5.3 Runtime (`runtime/`)
- Populate and honour real tensor sizes; validate the manifest on load.
- `POOL` and `NMC` execution modes.
- Replace the fake `prefetch_hits` with an observed-overlap measurement, or
  delete the metric.

### 5.4 Evaluation (`evaluation/`)
- **B3 TensorRT FP16 GPU** and **B4 TensorRT INT8 GPU** baselines. TensorRT
  10.7, `trtexec`, libnvinfer and pycuda are present and functional on the
  Jetson; onnxruntime 1.23.2 there has no CUDA provider, which is the actual
  reason the submitted baselines were CPU-only.
- Numerical correctness harness: NMC output vs FP32 reference.
- Reporting policy: a run that times out or fails is recorded as such and never
  as `ok`; full distributions (CV, p99, max) always reported.

## 6. Claims policy

Every claim in the paper maps to one of:
**MEASURED** (on hardware, with the script that produced it) /
**EMULATED** (device emulator, stated as such) /
**MODELLED** (analytical, with calibration source and validity range).

The 55x figure is stated precisely as *total model size divided by peak FPGA
DDR2 resident footprint* — not as a 55x gain over a Jetson-only deployment (R1#7).

## 7. Hardware gating

Buildable and verifiable on the Jetson without the board: partitioner, `cxlwin`
with the emulator backend, runtime modes, TensorRT baselines, RTL fixes under
Icarus simulation, paper.

Requires the Vivado workstation and the Nexys 4 DDR board: synthesis of the MAC
fixes, and the measured POOL-vs-NMC crossover on real silicon. These will be
batched into one specified session rather than incremental trips.

## 8. Non-goals
- Cache-line-granular hardware coherence. Out of reach; the paper will say so.
- Beating the GPU on latency for models that fit in host memory. The honest
  claim is capacity, plus a predicted crossover.
