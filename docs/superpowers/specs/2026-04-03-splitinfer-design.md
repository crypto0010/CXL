# SplitInfer: CXL-Inspired Heterogeneous Memory-Aware Model Partitioning for Edge ML Inference

**Date:** 2026-04-03
**Target Venue:** Top-tier architecture/systems conference (ISCA, MICRO, ASPLOS, HPCA)
**Status:** Design Spec — Awaiting Review

---

## 1. Problem Statement

Edge ML inference on memory-constrained GPU platforms faces a hard ceiling. The NVIDIA Jetson Orin Nano Super has 8GB LPDDR5 shared between CPU and GPU — as models grow (LLMs, vision transformers, recommendation models), they either don't fit in device memory or consume so much memory that concurrent inference, batching, and pipelining become impossible.

Current solutions — model compression, quantization, pruning — sacrifice accuracy. Simply adding more memory is impossible on fixed SoC platforms.

CXL has enabled memory disaggregation and near-memory compute in data centers, but no work has explored these concepts for edge/embedded heterogeneous systems where GPU and FPGA coexist.

## 2. Research Thesis

A CXL-inspired heterogeneous memory architecture that partitions ML inference across an edge GPU and an FPGA with local DDR2 — executing compute-heavy layers on the GPU and memory-heavy layers as near-memory compute on the FPGA — can enable deployment of larger models at the edge with minimal latency overhead, by co-designing a lightweight coherency protocol and an automated model partitioning algorithm.

## 3. Key Research Questions

1. How should a CXL-like coherency protocol be adapted for low-bandwidth edge interconnects (USB 2.0 ~40 MB/s vs. PCIe Gen5 ~64 GB/s)?
2. What is the optimal partitioning strategy for splitting NN layers across GPU and FPGA given asymmetric compute/memory/bandwidth?
3. What is the achievable accuracy-latency-memory Pareto frontier compared to compression-only baselines?

---

## 4. Hardware Platform

### 4.1 Edge GPU: NVIDIA Jetson Orin Nano Super

- SoC: NVIDIA Orin (Arm Cortex-A78AE 6-core + Ampere GPU 1024 CUDA cores)
- Memory: 8GB LPDDR5 (unified CPU/GPU)
- Software: JetPack 6.x, TensorRT 8.x, CUDA 12.x
- Role: Executes compute-heavy NN layers, hosts SplitInfer runtime

### 4.2 FPGA: Digilent Nexys 4 DDR (Xilinx XC7A100T-1CSG324C)

- Logic: 101,440 LUTs, 240 DSP48E1 slices, 4,860 Kbits Block RAM (~607 KB)
- Memory: DDR2 128MB (Micron MT47H64M16HR-25:H, 16-bit data bus, 1.8V, ~1.3 GB/s peak bandwidth)
- Speed grade: -1 (slowest), target clock ~100 MHz
- Package: CSG324 (324-pin BGA)
- USB-UART: FTDI FT2232HQ dual-channel (VID=0x0403, PID=0x6010), UART on Channel B
- UART pins: TXD_IN=C4, RXD_OUT=D4, CTS=D3, RTS=E5 (LVCMOS33)
- Clock: 100 MHz crystal on pin E3
- Reset: CPU_RESETN on pin C12 (active-low)
- LEDs: 16 standard LEDs (H17, K15, J13, N14, R18, V17, U17, U16, V16, T15, U14, T16, V15, U1, R2, P2)
- Pmod: JA, JB, JC, JD (8 signals each)
- Role: Hosts near-memory compute engine, executes memory-heavy NN layers locally

### 4.3 Interconnect: Jetson <-> FPGA

- Data channel: USB 2.0 via on-board FTDI FT2232HQ (~40 MB/s usable bandwidth)
- Control channel: GPIO-based sync signaling via Pmod (low-latency barrier/ownership signals)
- Key constraint: 2MB activation tensor transfer takes ~50ms. Prefetching and pipelining must hide this.

### 4.4 Development Environment

- **FPGA development (Vivado synthesis, implementation, bitstream generation):** Runs on a separate x86 workstation. The Jetson is NOT used for FPGA toolchain work.
- **FPGA Toolchain:** Vivado 2023.x+, Verilog/SystemVerilog for RTL design
- **MIG IP:** Xilinx MIG 7 Series configured for DDR2 (NOT DDR4) — MT47H64M16HR-25:H part
- **Jetson development:** Native compilation on Jetson or cross-compilation from x86 workstation
- **Workflow:** Develop and synthesize FPGA bitstream on x86 machine -> Program Nexys 4 DDR board -> Connect to Jetson -> Run inference experiments on Jetson

---

## 5. System Architecture

SplitInfer consists of four components:

### 5.1 EdgeCoh Protocol (CXL-Inspired Lightweight Coherency)

A software-directed coherency protocol inspired by CXL.mem/CXL.cache, redesigned for edge bandwidth constraints.

**Design rationale:** CXL assumes PCIe Gen5 (~64 GB/s). The Jetson-FPGA link is USB 2.0 (~40 MB/s) — a ~1600x bandwidth gap. Hardware snooping (as in CXL.cache) is infeasible. Instead, EdgeCoh exploits the fact that NN execution graphs are static and known at compile time, making transfer patterns fully predictable.

**CXL-to-EdgeCoh mapping:**

| CXL Concept | EdgeCoh Adaptation |
|---|---|
| CXL.mem HDM (Host-managed Device Memory) | FPGA DDR2 exposed as host-managed memory region. Jetson runtime maps tensor regions to FPGA DDR2 addresses |
| CXL.cache Snoop (hardware coherency) | Software-directed coherency — runtime explicitly manages ownership. No hardware snoops |
| CXL Bias modes (Host/Device Bias) | Layer Bias — executing device has exclusive ownership during layer execution. Ownership transfers at layer boundaries only |

**Protocol operations:**

- `TRANSFER_OWNERSHIP(tensor_id, target_device)` — moves tensor ownership at layer boundaries
- `PREFETCH(tensor_id, target_device)` — asynchronously pre-stages next tensor while current layer executes
- `SYNC_BARRIER()` — lightweight synchronization between devices after a pipeline stage

**Coherency model:**
- Single-writer: only one device owns a tensor at any time
- Ownership transfer is explicit and scheduled at compile time
- Double-buffered: device A computes on buffer 0 while buffer 1 is being transferred

### 5.2 Near-Memory Compute (NMC) Engine on FPGA

**Resource allocation for XC7A100T:**

| Resource | Available | DDR2 Controller | USB/Comm | NMC Engine | Margin |
|---|---|---|---|---|---|
| LUTs | 101,440 | ~15,000 | ~10,000 | ~60,000 | ~16,440 |
| DSP48E1 | 240 | 0 | 0 | ~180 | ~60 |
| BRAM | 4,860 Kb | ~500 Kb | ~200 Kb | ~3,000 Kb | ~1,160 Kb |

**Compute units:**

1. **Embedding table lookup engine**
   - Primary target workload. Purely memory-bound (zero DSP usage).
   - Large embedding tables (tens to hundreds of MB) stored in DDR2
   - FPGA reads and returns looked-up vectors — avoids transferring entire tables to Jetson
   - Minimal LUT usage: DDR2 read logic + output buffering

2. **8x8 INT8 MAC array**
   - Uses ~64 DSP48E1 slices
   - Throughput: 64 INT8 MACs/cycle x 100 MHz = ~6.4 GOPS INT8
   - Targets FC layers with large weight matrices but small input/output dimensions
   - Weight matrices stored in DDR2, tiled through the MAC array
   - Example: 4096x512 INT8 FC layer = 2MB weights in DDR2

3. **Element-wise operations**
   - ReLU, quantized add, scale — trivial logic
   - Executes on activation data already resident on FPGA side
   - Avoids round-trip to Jetson for simple post-processing

**Out of scope for XC7A100T:**
- Softmax normalization (too complex)
- GELU approximation (excessive BRAM for LUT approach)
- Large systolic arrays (insufficient DSP budget)

### 5.3 Model Partitioning Engine (Offline)

An offline profiling and optimization tool that decides the optimal GPU/FPGA split.

**Inputs:**
- ONNX model graph
- Per-layer profiling data: GPU compute time, memory footprint, activation tensor sizes
- Hardware parameters: GPU throughput, FPGA NMC throughput, USB bandwidth, DDR2 bandwidth

**Algorithm — Constrained min-latency graph partitioning:**

1. Profile each layer on Jetson GPU (measure execution time + peak memory)
2. Estimate each layer's FPGA execution time based on NMC engine specifications
3. For each candidate partition (cut point in topologically sorted layer graph):
   - Compute: GPU-side latency + FPGA-side latency + transfer overhead at cut boundaries
   - Account for prefetch pipelining (overlapped transfer + compute)
4. Constraint: FPGA-assigned layer weights must fit in DDR2 capacity (128MB)
5. Solver: dynamic programming over topologically sorted layer graph
6. Output: partition manifest + activation transfer schedule

**Output artifacts:**
- Layer assignment map (each layer -> GPU or FPGA)
- Transfer schedule (which tensors, when, which direction)
- Prefetch plan (what to pre-stage during each layer's execution)

### 5.4 SplitInfer Runtime (On Jetson)

Lightweight orchestration runtime running on the Jetson:

- Loads the partition manifest at startup
- Manages GPU execution via TensorRT (for GPU-assigned layers)
- Sends commands to FPGA via EdgeCoh protocol (for FPGA-assigned layers)
- Implements double-buffered pipelining: while GPU executes layer N, FPGA prefetches activations for layer N+1 (or vice versa)
- Handles SYNC_BARRIER synchronization between devices
- Collects runtime telemetry (latency breakdown, memory usage, prefetch hit rate)

---

## 6. Target Model Architectures

| Model | Total Size | Memory-Heavy Portion (FPGA) | Compute-Heavy Portion (GPU) | Why Selected |
|---|---|---|---|---|
| DLRM (Criteo-Kaggle) | ~4GB (embeddings) | Embedding table lookups + bottom FC | Top MLP + feature interaction | Primary target: embedding tables dominate memory |
| MobileBERT (INT8) | ~100MB | INT8 projection matrices (Q,K,V) | Attention computation, LayerNorm | Representative transformer-at-edge |
| YOLOv8-nano + MobileNetV3 | ~50MB combined | Large FC classification layers | Conv backbone + neck | Real-world multi-model edge pipeline |
| Scaled DLRM (oversized embeddings) | >8GB total, embedding subset on FPGA | Embedding table partitions (up to 128MB per pass) | Top MLP + feature interaction | Capability unlocking — demonstrates models that cannot fit on Jetson alone, with multi-pass embedding access |

---

## 7. Evaluation Plan

### 7.1 Experimental Setup

| Component | Specification |
|---|---|
| Edge GPU | NVIDIA Jetson Orin Nano Super, 8GB LPDDR5 |
| FPGA | Digilent Nexys 4 DDR, XC7A100T-1CSG324C, 128MB DDR2 |
| Interconnect | USB 2.0 (~40 MB/s) + GPIO sync |
| FPGA Development | Vivado on separate x86 workstation |
| GPU Software | JetPack 6.x, TensorRT 8.x, CUDA 12.x |
| ML Framework | ONNX Runtime (profiling), TensorRT (GPU execution) |

### 7.2 Baselines

| ID | Baseline | Description |
|---|---|---|
| B1 | Jetson-only (full precision) | Entire model on GPU, FP32/FP16. Upper bound on accuracy. |
| B2 | Jetson-only (quantized) | INT8/INT4 quantized model on GPU. State-of-the-art compression baseline. |
| B3 | Jetson CPU offload | Large layers offloaded to Arm CPU via unified memory. Naive heterogeneous baseline. |
| B4 | Jetson disk swap | Weights swapped to SD card/NVMe under memory pressure. Storage-based expansion baseline. |
| B5 | SplitInfer (no prefetch) | SplitInfer without double-buffered pipelining. Ablation baseline. |

### 7.3 Metrics

**Primary:**
- End-to-end inference latency (ms) — averaged over 1000 inferences after warmup
- Throughput (inferences/sec) — measured over 60-second sustained window
- Peak Jetson memory usage (MB) — monitored via tegrastats
- Model deployability (binary) — can the model run at all?

**Secondary:**
- Latency breakdown: GPU compute / FPGA compute / transfer / sync overhead
- FPGA resource utilization: LUT, DSP, BRAM usage + Vivado power estimate
- Accuracy preservation vs. B1 (same precision) and B2 (quantized)
- Energy consumption: Jetson (tegrastats power rail) + FPGA (Vivado power analysis)
- Prefetch hit rate: % of transfers fully hidden by pipelining

### 7.4 Experiments

**E1: Capability Unlocking (Headline Result)**
- Workload: Scaled DLRM (8-16GB embeddings)
- B1, B2, B3: OOM crash. B4: runs at terrible latency. SplitInfer: runs successfully.
- Demonstrates: "enables previously impossible edge deployments"

**E2: Performance Comparison Across Workloads**
- Workloads: DLRM, MobileBERT, YOLOv8+MobileNetV3
- All baselines vs. SplitInfer
- Report: latency, throughput, memory usage
- Expected: SplitInfer trades 10-30% latency for 40-60% memory reduction vs. B2

**E3: Partitioning Sensitivity Analysis**
- Workload: DLRM
- Vary partition cut point: sweep from "all on GPU" to "maximum FPGA offload"
- Plot: latency vs. Jetson memory usage (Pareto frontier)
- Validates: partitioning algorithm finds near-optimal operating point

**E4: Ablation Study**
- Compare: SplitInfer (full) vs. B5 (no prefetch) vs. naive round-robin partitioning
- Quantifies contribution of: (a) intelligent partitioning, (b) prefetch pipelining, (c) EdgeCoh protocol

**E5: Scalability Projection**
- Analytical extrapolation using measured compute/transfer breakdowns
- Model performance at higher bandwidths: USB 3.0 (400 MB/s), hypothetical CXL-lite (4 GB/s)
- Supports: "with better interconnect, SplitInfer's benefits scale dramatically"

---

## 8. Expected Contributions

1. **EdgeCoh** — a lightweight software-directed coherency protocol inspired by CXL.mem, designed for bandwidth-constrained edge interconnects, exploiting static NN graph predictability
2. **SplitInfer partitioning algorithm** — an NN-graph-aware partitioning framework jointly optimizing latency, memory, and inter-device bandwidth on heterogeneous edge platforms
3. **First end-to-end prototype** of CXL-inspired memory disaggregation for edge ML inference on real GPU (Jetson) + FPGA (Artix-7) hardware
4. **Comprehensive evaluation** demonstrating capability unlocking (models that couldn't previously run) and characterizing the performance/memory/energy tradeoff space

---

## 9. Risk Mitigation

| Risk | Impact | Mitigation |
|---|---|---|
| USB 2.0 bandwidth too low for acceptable latency | Latency results look bad | Focus narrative on capability unlocking (E1) + scalability projection (E5). Prefetch pipelining hides latency for models with high compute-to-transfer ratio. |
| Artix-7 NMC engine too slow vs. GPU | FPGA layers become bottleneck | Scope FPGA to memory-bound-only operations (embedding lookups) where FPGA DDR2 bandwidth (~1.3 GB/s peak) is the limit, not compute. |
| Reviewer questions "this isn't real CXL" | Paper rejected for overclaiming | Frame as "CXL-inspired" throughout. Contribution is the protocol adaptation and co-design methodology, not CXL compliance. |
| DDR2 capacity (128MB) limits embedding table size | Constrains how much can be offloaded per pass | Use multi-pass embedding access (stream embedding partitions through 128MB DDR2), hash embeddings for compression, or focus primary evaluation on models with moderate embedding sizes that fit in 128MB. For the capability unlocking experiment, demonstrate that even 128MB of FPGA-side memory extends the deployable model range. |
| Partitioning algorithm produces poor cuts | Suboptimal results | Compare against exhaustive search on small models to validate DP solver quality. |

---

## 10. Suggested Conference Targets

| Venue | Fit | Angle to Emphasize |
|---|---|---|
| **ASPLOS** | Strong | Systems + architecture co-design, real prototype |
| **MICRO** | Strong | Microarchitectural innovation (NMC engine, EdgeCoh) |
| **ISCA** | Moderate | Architecture contribution, scalability projection strengthens |
| **DAC** | Strong | Embedded systems + FPGA design, practical prototype |
| **MLSys** | Strong | ML systems optimization, partitioning algorithm |
| **SEC/EdgeSys (workshop)** | Backup | Edge computing focus, lower bar |
