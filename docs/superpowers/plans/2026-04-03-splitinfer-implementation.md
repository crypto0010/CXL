# SplitInfer Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a CXL-inspired heterogeneous inference system that partitions ML models across a Jetson Orin Nano GPU and a **Digilent Nexys 4 DDR** (Artix-7 FPGA with 128MB DDR2), connected via USB 2.0.

**Architecture:** Four components — (1) EdgeCoh lightweight coherency protocol for Jetson-FPGA communication, (2) Near-Memory Compute engine on FPGA (embedding lookups + INT8 MAC array), (3) offline model partitioning engine (Python, ONNX-based DP solver), (4) SplitInfer runtime on Jetson orchestrating split inference with double-buffered pipelining.

**Tech Stack:**
- FPGA: Verilog/SystemVerilog, Vivado 2023.x+, **Xilinx MIG 7 Series DDR2 IP** (Micron MT47H64M16HR-25:H), targeting XC7A100T-1CSG324C @ 100 MHz
- Jetson runtime: C++ (TensorRT, CUDA 12.x, libusb), built on JetPack 6.x
- Partitioning engine: Python 3.10+, onnx, onnxruntime, numpy
- EdgeCoh protocol: shared C library (libedgecoh) used by both Jetson runtime and test harnesses
- Build: CMake (C/C++), Vivado project (FPGA), pip/pyproject.toml (Python)

**FPGA Board: Digilent Nexys 4 DDR — Key Hardware Facts:**
- FPGA: Xilinx XC7A100T-1CSG324C (101K LUTs, 240 DSP48E1, 4860 Kb BRAM)
- Memory: **DDR2** 128MB (Micron MT47H64M16HR-25:H, **16-bit data bus**, 1.8V, ~1.3 GB/s peak)
- USB-UART: FTDI FT2232HQ (VID=0x0403, PID=0x6010), Channel B for UART
- UART pins: TXD_IN=C4, RXD_OUT=D4 (LVCMOS33)
- Clock: 100 MHz on E3
- Reset: CPU_RESETN on C12 (active-low)
- LEDs: 16 standard (H17, K15, J13, N14, ...)
- Pmod: JA (C17..G18), JB (D14..H16), JC (K1..E6), JD (H4..F3)

**Development machines:**
- **x86 workstation:** FPGA development (Vivado synthesis/implementation/bitstream), Python partitioning engine, EdgeCoh protocol library
- **Jetson Orin Nano:** Runtime development, TensorRT integration, GPU profiling, end-to-end experiments
- **Nexys 4 DDR board:** Connected to Jetson via USB for runtime experiments; programmed from x86 workstation

---

## File Structure

```
splitinfer/
├── CMakeLists.txt                          # Top-level CMake (builds libedgecoh + runtime)
├── pyproject.toml                          # Python package config (partitioner)
│
├── protocol/                               # EdgeCoh protocol (shared C library)
│   ├── CMakeLists.txt
│   ├── include/
│   │   └── edgecoh/
│   │       ├── edgecoh.h                   # Public API: message types, state machine
│   │       ├── messages.h                  # Protocol message definitions (packed structs)
│   │       └── transport.h                 # USB transport abstraction
│   ├── src/
│   │   ├── edgecoh.c                       # Protocol state machine implementation
│   │   ├── messages.c                      # Serialize/deserialize protocol messages
│   │   └── transport_usb.c                 # USB 2.0 transport via libusb/FTDI
│   └── tests/
│       ├── test_messages.c                 # Unit tests: message encoding/decoding
│       ├── test_state_machine.c            # Unit tests: ownership state transitions
│       └── test_loopback.c                 # Integration: USB loopback with FPGA echo
│
├── partitioner/                            # Model Partitioning Engine (Python)
│   ├── __init__.py
│   ├── graph.py                            # ONNX graph parser, layer extraction
│   ├── profiler.py                         # GPU layer profiler (runs on Jetson)
│   ├── cost_model.py                       # FPGA/GPU/transfer cost estimation
│   ├── solver.py                           # DP partitioning solver
│   ├── manifest.py                         # Partition manifest generator (JSON output)
│   └── tests/
│       ├── test_graph.py                   # Tests: ONNX graph parsing
│       ├── test_cost_model.py              # Tests: cost estimation accuracy
│       ├── test_solver.py                  # Tests: DP solver correctness
│       └── test_manifest.py                # Tests: manifest generation
│
├── runtime/                                # SplitInfer Runtime (C++, runs on Jetson)
│   ├── CMakeLists.txt
│   ├── include/
│   │   └── splitinfer/
│   │       ├── runtime.h                   # Main runtime API
│   │       ├── manifest.h                  # Manifest loader (reads JSON from partitioner)
│   │       ├── gpu_executor.h              # TensorRT-based GPU layer executor
│   │       ├── fpga_executor.h             # FPGA layer executor (via EdgeCoh)
│   │       ├── pipeline.h                  # Double-buffered pipeline orchestrator
│   │       └── telemetry.h                 # Latency/memory/prefetch telemetry
│   ├── src/
│   │   ├── runtime.cpp                     # Main runtime: load manifest, run inference
│   │   ├── manifest.cpp                    # JSON manifest parser
│   │   ├── gpu_executor.cpp                # TensorRT engine builder + inference
│   │   ├── fpga_executor.cpp               # Send/receive tensors via EdgeCoh
│   │   ├── pipeline.cpp                    # Orchestrate GPU/FPGA with double buffering
│   │   └── telemetry.cpp                   # Collect and report timing metrics
│   ├── tests/
│   │   ├── test_manifest.cpp               # Test manifest loading
│   │   └── test_pipeline_mock.cpp          # Test pipeline logic with mock executors
│   └── tools/
│       └── splitinfer_run.cpp              # CLI entry point: run split inference
│
├── fpga/                                   # FPGA design (Verilog, built on x86 workstation)
│   ├── README.md                           # Build instructions for Vivado
│   ├── constraints/
│   │   └── nexys4ddr.xdc                    # Pin constraints for Nexys 4 DDR board
│   ├── src/
│   │   ├── top.v                           # Top-level: connects all modules
│   │   ├── usb_interface.v                 # FTDI USB 2.0 FIFO interface
│   │   ├── edgecoh_controller.v            # EdgeCoh protocol FSM (FPGA side)
│   │   ├── ddr2_arbiter.v                  # DDR2 access arbiter (NMC vs USB DMA)
│   │   ├── embedding_lookup.v              # Embedding table lookup engine
│   │   ├── mac_array_8x8.v                 # 8x8 INT8 MAC array
│   │   ├── elementwise.v                   # ReLU, quantized add, scale
│   │   ├── nmc_dispatch.v                  # Dispatches commands to NMC compute units
│   │   └── cdc_fifo.v                     # Clock-domain crossing FIFO (sys_clk <-> ui_clk)
│   ├── sim/
│   │   ├── tb_embedding_lookup.v           # Testbench: embedding lookups
│   │   ├── tb_mac_array.v                  # Testbench: MAC array correctness
│   │   ├── tb_edgecoh_controller.v         # Testbench: protocol FSM
│   │   ├── tb_elementwise.v                # Testbench: element-wise ops
│   │   └── tb_top.v                        # Testbench: full system integration
│   └── ip/
│       ├── mig_ddr2/                       # Xilinx MIG 7 Series DDR2 IP (generated by Vivado for MT47H64M16HR-25:H)
│       └── clk_wiz_200/                    # Clocking Wizard: 100 MHz → 200 MHz (MIG reference clock)
│
├── evaluation/                             # Experiment scripts and baselines
│   ├── baselines/
│   │   ├── jetson_only_fp.py               # B1: full precision on Jetson
│   │   ├── jetson_only_quant.py            # B2: quantized on Jetson
│   │   ├── jetson_cpu_offload.py           # B3: CPU offload baseline
│   │   └── jetson_disk_swap.py             # B4: disk swap baseline
│   ├── experiments/
│   │   ├── e1_capability_unlock.sh         # E1: scaled DLRM, OOM vs success
│   │   ├── e2_performance.sh               # E2: latency/throughput/memory across workloads
│   │   ├── e3_partition_sweep.py           # E3: vary partition cut, plot Pareto
│   │   ├── e4_ablation.sh                  # E4: ablation study
│   │   └── e5_scalability.py               # E5: analytical bandwidth projection
│   └── models/
│       └── download_models.sh              # Download DLRM, MobileBERT, YOLOv8-nano
│
└── docs/
    └── superpowers/
        ├── specs/
        │   └── 2026-04-03-splitinfer-design.md
        └── plans/
            └── 2026-04-03-splitinfer-implementation.md
```

---

## Phase 1: Project Scaffolding and EdgeCoh Protocol

This phase builds the communication foundation. Everything else depends on being able to send structured messages between Jetson and FPGA.

---

### Task 1: Project Scaffolding

**Files:**
- Create: `splitinfer/CMakeLists.txt`
- Create: `splitinfer/pyproject.toml`
- Create: `splitinfer/protocol/CMakeLists.txt`
- Create: `splitinfer/protocol/include/edgecoh/messages.h`

- [ ] **Step 1: Create top-level CMakeLists.txt**

```cmake
# splitinfer/CMakeLists.txt
cmake_minimum_required(VERSION 3.18)
project(splitinfer C CXX)

set(CMAKE_C_STANDARD 11)
set(CMAKE_CXX_STANDARD 17)

option(BUILD_TESTS "Build tests" ON)

add_subdirectory(protocol)
# runtime subdirectory added in Phase 4
# add_subdirectory(runtime)
```

- [ ] **Step 2: Create Python project config**

```toml
# splitinfer/pyproject.toml
[build-system]
requires = ["setuptools>=68.0"]
build-backend = "setuptools.backends._legacy:_Backend"

[project]
name = "splitinfer-partitioner"
version = "0.1.0"
requires-python = ">=3.10"
dependencies = [
    "onnx>=1.14",
    "onnxruntime>=1.16",
    "numpy>=1.24",
]

[project.optional-dependencies]
dev = ["pytest>=7.0"]

[tool.pytest.ini_options]
testpaths = ["partitioner/tests"]
```

- [ ] **Step 3: Create protocol CMakeLists.txt**

```cmake
# splitinfer/protocol/CMakeLists.txt
add_library(edgecoh
    src/messages.c
    src/edgecoh.c
    src/transport_usb.c
)
target_include_directories(edgecoh PUBLIC include)
target_link_libraries(edgecoh PRIVATE usb-1.0)

if(BUILD_TESTS)
    enable_testing()
    add_executable(test_messages tests/test_messages.c)
    target_link_libraries(test_messages edgecoh)
    add_test(NAME test_messages COMMAND test_messages)

    add_executable(test_state_machine tests/test_state_machine.c)
    target_link_libraries(test_state_machine edgecoh)
    add_test(NAME test_state_machine COMMAND test_state_machine)
endif()
```

- [ ] **Step 4: Create protocol message definitions header**

```c
/* splitinfer/protocol/include/edgecoh/messages.h */
#ifndef EDGECOH_MESSAGES_H
#define EDGECOH_MESSAGES_H

#include <stdint.h>

/* EdgeCoh protocol message types — inspired by CXL.mem */

typedef enum {
    EDGECOH_MSG_TRANSFER_OWNERSHIP = 0x01,
    EDGECOH_MSG_PREFETCH           = 0x02,
    EDGECOH_MSG_SYNC_BARRIER       = 0x03,
    EDGECOH_MSG_DATA_WRITE         = 0x10,  /* Host -> FPGA: write tensor data */
    EDGECOH_MSG_DATA_READ          = 0x11,  /* Host -> FPGA: request tensor data */
    EDGECOH_MSG_DATA_RESPONSE      = 0x12,  /* FPGA -> Host: tensor data response */
    EDGECOH_MSG_NMC_EXEC           = 0x20,  /* Host -> FPGA: execute NMC operation */
    EDGECOH_MSG_NMC_DONE           = 0x21,  /* FPGA -> Host: NMC execution complete */
    EDGECOH_MSG_ACK                = 0xFE,
    EDGECOH_MSG_ERROR              = 0xFF,
} edgecoh_msg_type_t;

typedef enum {
    EDGECOH_DEV_HOST = 0,   /* Jetson */
    EDGECOH_DEV_FPGA = 1,
} edgecoh_device_t;

typedef enum {
    EDGECOH_NMC_EMBEDDING_LOOKUP = 0x01,
    EDGECOH_NMC_INT8_FC          = 0x02,
    EDGECOH_NMC_RELU             = 0x03,
    EDGECOH_NMC_QUANT_ADD        = 0x04,
    EDGECOH_NMC_SCALE            = 0x05,
} edgecoh_nmc_op_t;

/* All messages share a common 8-byte header */
typedef struct __attribute__((packed)) {
    uint8_t  msg_type;       /* edgecoh_msg_type_t */
    uint8_t  flags;          /* reserved */
    uint16_t tensor_id;      /* tensor identifier (0-65535) */
    uint32_t payload_len;    /* bytes following this header */
} edgecoh_header_t;

/* TRANSFER_OWNERSHIP: 8-byte header + 1 byte target device */
typedef struct __attribute__((packed)) {
    edgecoh_header_t header;
    uint8_t target_device;   /* edgecoh_device_t */
} edgecoh_transfer_msg_t;

/* PREFETCH: 8-byte header + 1 byte target device + 4 byte DDR2 offset */
typedef struct __attribute__((packed)) {
    edgecoh_header_t header;
    uint8_t  target_device;
    uint32_t ddr2_offset;    /* byte offset in FPGA DDR2 */
} edgecoh_prefetch_msg_t;

/* SYNC_BARRIER: header only (payload_len = 0) */
typedef edgecoh_header_t edgecoh_barrier_msg_t;

/* DATA_WRITE: header + DDR2 address + data bytes follow */
typedef struct __attribute__((packed)) {
    edgecoh_header_t header;
    uint32_t ddr2_addr;      /* destination address in FPGA DDR2 */
    /* payload_len bytes of tensor data follow */
} edgecoh_data_write_msg_t;

/* DATA_READ: header + DDR2 address + read length */
typedef struct __attribute__((packed)) {
    edgecoh_header_t header;
    uint32_t ddr2_addr;
    uint32_t read_len;       /* bytes to read */
} edgecoh_data_read_msg_t;

/* NMC_EXEC: header + operation type + operation-specific params */
typedef struct __attribute__((packed)) {
    edgecoh_header_t header;
    uint8_t  nmc_op;          /* edgecoh_nmc_op_t */
    uint32_t table_base_addr; /* DDR2 base address of weight/embedding table */
    uint32_t table_rows;      /* number of rows (embeddings) or input dim */
    uint32_t table_cols;      /* embedding dimension or output dim */
    uint32_t input_addr;      /* DDR2 address of input data (indices or activations) */
    uint32_t input_len;       /* number of input elements */
    uint32_t output_addr;     /* DDR2 address to write results */
} edgecoh_nmc_exec_msg_t;

/* Serialize a message to a byte buffer. Returns bytes written, or -1 on error. */
int edgecoh_serialize(const void *msg, uint8_t *buf, int buf_len);

/* Deserialize a message header from a byte buffer. Returns msg_type, or -1 on error. */
int edgecoh_deserialize_header(const uint8_t *buf, int buf_len, edgecoh_header_t *out);

#endif /* EDGECOH_MESSAGES_H */
```

- [ ] **Step 5: Commit scaffolding**

```bash
cd splitinfer
git init
git add CMakeLists.txt pyproject.toml protocol/CMakeLists.txt protocol/include/edgecoh/messages.h
git commit -m "feat: project scaffolding with EdgeCoh message definitions"
```

---

### Task 2: EdgeCoh Message Serialization

**Files:**
- Create: `splitinfer/protocol/src/messages.c`
- Create: `splitinfer/protocol/tests/test_messages.c`

- [ ] **Step 1: Write the failing test**

```c
/* splitinfer/protocol/tests/test_messages.c */
#include <stdio.h>
#include <string.h>
#include <assert.h>
#include "edgecoh/messages.h"

static void test_serialize_transfer_ownership(void) {
    edgecoh_transfer_msg_t msg = {
        .header = {
            .msg_type = EDGECOH_MSG_TRANSFER_OWNERSHIP,
            .flags = 0,
            .tensor_id = 42,
            .payload_len = 1,
        },
        .target_device = EDGECOH_DEV_FPGA,
    };

    uint8_t buf[64];
    int written = edgecoh_serialize(&msg, buf, sizeof(buf));
    assert(written == sizeof(edgecoh_transfer_msg_t));

    /* Deserialize and verify round-trip */
    edgecoh_header_t hdr;
    int type = edgecoh_deserialize_header(buf, written, &hdr);
    assert(type == EDGECOH_MSG_TRANSFER_OWNERSHIP);
    assert(hdr.tensor_id == 42);
    assert(hdr.payload_len == 1);

    /* Verify target device byte after header */
    assert(buf[sizeof(edgecoh_header_t)] == EDGECOH_DEV_FPGA);
}

static void test_serialize_nmc_exec(void) {
    edgecoh_nmc_exec_msg_t msg = {
        .header = {
            .msg_type = EDGECOH_MSG_NMC_EXEC,
            .flags = 0,
            .tensor_id = 7,
            .payload_len = sizeof(edgecoh_nmc_exec_msg_t) - sizeof(edgecoh_header_t),
        },
        .nmc_op = EDGECOH_NMC_EMBEDDING_LOOKUP,
        .table_base_addr = 0x00100000,
        .table_rows = 10000,
        .table_cols = 64,
        .input_addr = 0x00500000,
        .input_len = 128,
        .output_addr = 0x00600000,
    };

    uint8_t buf[128];
    int written = edgecoh_serialize(&msg, buf, sizeof(buf));
    assert(written == sizeof(edgecoh_nmc_exec_msg_t));

    edgecoh_header_t hdr;
    int type = edgecoh_deserialize_header(buf, written, &hdr);
    assert(type == EDGECOH_MSG_NMC_EXEC);
    assert(hdr.tensor_id == 7);
}

static void test_serialize_buffer_too_small(void) {
    edgecoh_transfer_msg_t msg = {
        .header = {
            .msg_type = EDGECOH_MSG_TRANSFER_OWNERSHIP,
            .flags = 0,
            .tensor_id = 1,
            .payload_len = 1,
        },
        .target_device = EDGECOH_DEV_HOST,
    };

    uint8_t buf[2]; /* too small */
    int written = edgecoh_serialize(&msg, buf, sizeof(buf));
    assert(written == -1);
}

static void test_deserialize_truncated(void) {
    uint8_t buf[4] = {EDGECOH_MSG_ACK, 0, 0, 0}; /* only 4 bytes, header needs 8 */
    edgecoh_header_t hdr;
    int type = edgecoh_deserialize_header(buf, 4, &hdr);
    assert(type == -1);
}

int main(void) {
    test_serialize_transfer_ownership();
    test_serialize_nmc_exec();
    test_serialize_buffer_too_small();
    test_deserialize_truncated();
    printf("All message tests passed.\n");
    return 0;
}
```

- [ ] **Step 2: Run test to verify it fails**

```bash
cd splitinfer && mkdir -p build && cd build
cmake .. -DBUILD_TESTS=ON && make test_messages
```

Expected: Build fails — `messages.c` does not exist yet.

- [ ] **Step 3: Write minimal implementation**

```c
/* splitinfer/protocol/src/messages.c */
#include "edgecoh/messages.h"
#include <string.h>

static int msg_total_size(uint8_t msg_type) {
    switch (msg_type) {
    case EDGECOH_MSG_TRANSFER_OWNERSHIP:
        return (int)sizeof(edgecoh_transfer_msg_t);
    case EDGECOH_MSG_PREFETCH:
        return (int)sizeof(edgecoh_prefetch_msg_t);
    case EDGECOH_MSG_SYNC_BARRIER:
    case EDGECOH_MSG_ACK:
    case EDGECOH_MSG_ERROR:
        return (int)sizeof(edgecoh_header_t);
    case EDGECOH_MSG_DATA_WRITE:
        return (int)sizeof(edgecoh_data_write_msg_t);
    case EDGECOH_MSG_DATA_READ:
        return (int)sizeof(edgecoh_data_read_msg_t);
    case EDGECOH_MSG_NMC_EXEC:
        return (int)sizeof(edgecoh_nmc_exec_msg_t);
    case EDGECOH_MSG_NMC_DONE:
    case EDGECOH_MSG_DATA_RESPONSE:
        return (int)sizeof(edgecoh_header_t);
    default:
        return -1;
    }
}

int edgecoh_serialize(const void *msg, uint8_t *buf, int buf_len) {
    const edgecoh_header_t *hdr = (const edgecoh_header_t *)msg;
    int size = msg_total_size(hdr->msg_type);
    if (size < 0 || size > buf_len) {
        return -1;
    }
    memcpy(buf, msg, size);
    return size;
}

int edgecoh_deserialize_header(const uint8_t *buf, int buf_len, edgecoh_header_t *out) {
    if (buf_len < (int)sizeof(edgecoh_header_t)) {
        return -1;
    }
    memcpy(out, buf, sizeof(edgecoh_header_t));
    return (int)out->msg_type;
}
```

- [ ] **Step 4: Build and run tests**

```bash
cd splitinfer/build && cmake .. -DBUILD_TESTS=ON && make test_messages && ./test_messages
```

Expected: `All message tests passed.`

- [ ] **Step 5: Commit**

```bash
git add protocol/src/messages.c protocol/tests/test_messages.c
git commit -m "feat: EdgeCoh message serialization with round-trip tests"
```

---

### Task 3: EdgeCoh Protocol State Machine

**Files:**
- Create: `splitinfer/protocol/include/edgecoh/edgecoh.h`
- Create: `splitinfer/protocol/src/edgecoh.c`
- Create: `splitinfer/protocol/tests/test_state_machine.c`

- [ ] **Step 1: Write the failing test**

```c
/* splitinfer/protocol/tests/test_state_machine.c */
#include <stdio.h>
#include <assert.h>
#include "edgecoh/edgecoh.h"

static void test_initial_state(void) {
    edgecoh_state_t state;
    edgecoh_init(&state, 16);
    assert(edgecoh_get_owner(&state, 0) == EDGECOH_DEV_HOST);
    assert(edgecoh_get_owner(&state, 15) == EDGECOH_DEV_HOST);
}

static void test_transfer_ownership(void) {
    edgecoh_state_t state;
    edgecoh_init(&state, 16);

    int rc = edgecoh_transfer(&state, 5, EDGECOH_DEV_FPGA);
    assert(rc == 0);
    assert(edgecoh_get_owner(&state, 5) == EDGECOH_DEV_FPGA);

    rc = edgecoh_transfer(&state, 5, EDGECOH_DEV_HOST);
    assert(rc == 0);
    assert(edgecoh_get_owner(&state, 5) == EDGECOH_DEV_HOST);
}

static void test_transfer_to_same_owner_is_noop(void) {
    edgecoh_state_t state;
    edgecoh_init(&state, 16);
    int rc = edgecoh_transfer(&state, 3, EDGECOH_DEV_HOST);
    assert(rc == 0);
    assert(edgecoh_get_owner(&state, 3) == EDGECOH_DEV_HOST);
}

static void test_invalid_tensor_id(void) {
    edgecoh_state_t state;
    edgecoh_init(&state, 16);
    int rc = edgecoh_transfer(&state, 99, EDGECOH_DEV_FPGA);
    assert(rc == -1);
    assert(edgecoh_get_owner(&state, 99) == -1);
}

static void test_prefetch_marks_pending(void) {
    edgecoh_state_t state;
    edgecoh_init(&state, 16);

    int rc = edgecoh_prefetch(&state, 8, EDGECOH_DEV_FPGA);
    assert(rc == 0);
    assert(edgecoh_is_prefetching(&state, 8) == 1);

    rc = edgecoh_prefetch_complete(&state, 8);
    assert(rc == 0);
    assert(edgecoh_get_owner(&state, 8) == EDGECOH_DEV_FPGA);
    assert(edgecoh_is_prefetching(&state, 8) == 0);
}

static void test_barrier_increments_epoch(void) {
    edgecoh_state_t state;
    edgecoh_init(&state, 16);

    uint32_t epoch0 = edgecoh_get_epoch(&state);
    edgecoh_barrier(&state);
    uint32_t epoch1 = edgecoh_get_epoch(&state);
    assert(epoch1 == epoch0 + 1);
}

int main(void) {
    test_initial_state();
    test_transfer_ownership();
    test_transfer_to_same_owner_is_noop();
    test_invalid_tensor_id();
    test_prefetch_marks_pending();
    test_barrier_increments_epoch();
    printf("All state machine tests passed.\n");
    return 0;
}
```

- [ ] **Step 2: Run test to verify it fails**

```bash
cd splitinfer/build && cmake .. && make test_state_machine
```

Expected: Build fails — `edgecoh.h` and `edgecoh.c` don't exist yet.

- [ ] **Step 3: Write the header**

```c
/* splitinfer/protocol/include/edgecoh/edgecoh.h */
#ifndef EDGECOH_H
#define EDGECOH_H

#include "messages.h"
#include <stdint.h>

#define EDGECOH_MAX_TENSORS 256

typedef struct {
    uint8_t  owner[EDGECOH_MAX_TENSORS];
    uint8_t  prefetch_pending[EDGECOH_MAX_TENSORS];
    uint8_t  prefetch_target[EDGECOH_MAX_TENSORS];
    uint16_t num_tensors;
    uint32_t epoch;
} edgecoh_state_t;

void edgecoh_init(edgecoh_state_t *state, uint16_t num_tensors);
int edgecoh_transfer(edgecoh_state_t *state, uint16_t tensor_id, uint8_t target_device);
int edgecoh_get_owner(const edgecoh_state_t *state, uint16_t tensor_id);
int edgecoh_prefetch(edgecoh_state_t *state, uint16_t tensor_id, uint8_t target_device);
int edgecoh_prefetch_complete(edgecoh_state_t *state, uint16_t tensor_id);
int edgecoh_is_prefetching(const edgecoh_state_t *state, uint16_t tensor_id);
void edgecoh_barrier(edgecoh_state_t *state);
uint32_t edgecoh_get_epoch(const edgecoh_state_t *state);

#endif /* EDGECOH_H */
```

- [ ] **Step 4: Write the implementation**

```c
/* splitinfer/protocol/src/edgecoh.c */
#include "edgecoh/edgecoh.h"
#include <string.h>

void edgecoh_init(edgecoh_state_t *state, uint16_t num_tensors) {
    memset(state, 0, sizeof(*state));
    state->num_tensors = (num_tensors > EDGECOH_MAX_TENSORS)
                         ? EDGECOH_MAX_TENSORS : num_tensors;
    state->epoch = 0;
}

int edgecoh_transfer(edgecoh_state_t *state, uint16_t tensor_id, uint8_t target_device) {
    if (tensor_id >= state->num_tensors) return -1;
    state->owner[tensor_id] = target_device;
    return 0;
}

int edgecoh_get_owner(const edgecoh_state_t *state, uint16_t tensor_id) {
    if (tensor_id >= state->num_tensors) return -1;
    return state->owner[tensor_id];
}

int edgecoh_prefetch(edgecoh_state_t *state, uint16_t tensor_id, uint8_t target_device) {
    if (tensor_id >= state->num_tensors) return -1;
    state->prefetch_pending[tensor_id] = 1;
    state->prefetch_target[tensor_id] = target_device;
    return 0;
}

int edgecoh_prefetch_complete(edgecoh_state_t *state, uint16_t tensor_id) {
    if (tensor_id >= state->num_tensors) return -1;
    if (!state->prefetch_pending[tensor_id]) return -1;
    state->owner[tensor_id] = state->prefetch_target[tensor_id];
    state->prefetch_pending[tensor_id] = 0;
    return 0;
}

int edgecoh_is_prefetching(const edgecoh_state_t *state, uint16_t tensor_id) {
    if (tensor_id >= state->num_tensors) return -1;
    return state->prefetch_pending[tensor_id];
}

void edgecoh_barrier(edgecoh_state_t *state) {
    state->epoch++;
}

uint32_t edgecoh_get_epoch(const edgecoh_state_t *state) {
    return state->epoch;
}
```

- [ ] **Step 5: Build and run tests**

```bash
cd splitinfer/build && cmake .. && make test_state_machine && ./test_state_machine
```

Expected: `All state machine tests passed.`

- [ ] **Step 6: Commit**

```bash
git add protocol/include/edgecoh/edgecoh.h protocol/src/edgecoh.c protocol/tests/test_state_machine.c
git commit -m "feat: EdgeCoh protocol state machine with ownership tracking and prefetch"
```

---

### Task 4: USB Transport Layer

**Files:**
- Create: `splitinfer/protocol/include/edgecoh/transport.h`
- Create: `splitinfer/protocol/src/transport_usb.c`

No unit test for this task — USB transport requires hardware. Integration tested in Task 19.

- [ ] **Step 1: Write the transport header**

```c
/* splitinfer/protocol/include/edgecoh/transport.h */
#ifndef EDGECOH_TRANSPORT_H
#define EDGECOH_TRANSPORT_H

#include <stdint.h>

typedef struct edgecoh_transport edgecoh_transport_t;

/* Nexys 4 DDR FTDI: VID=0x0403, PID=0x6010 (FT2232HQ dual channel). */
edgecoh_transport_t *edgecoh_transport_open(uint16_t vid, uint16_t pid);
void edgecoh_transport_close(edgecoh_transport_t *t);
int edgecoh_transport_send(edgecoh_transport_t *t, const uint8_t *buf, int len);
int edgecoh_transport_recv(edgecoh_transport_t *t, uint8_t *buf, int buf_len, int timeout_ms);
int edgecoh_send_msg(edgecoh_transport_t *t, const void *msg);
int edgecoh_recv_header(edgecoh_transport_t *t, edgecoh_header_t *hdr, int timeout_ms);

#endif /* EDGECOH_TRANSPORT_H */
```

- [ ] **Step 2: Write the USB transport implementation**

```c
/* splitinfer/protocol/src/transport_usb.c */
#include "edgecoh/transport.h"
#include "edgecoh/messages.h"
#include <libusb-1.0/libusb.h>
#include <stdlib.h>
#include <string.h>
#include <stdio.h>

#define FTDI_INTERFACE 1
#define FTDI_EP_OUT    0x02
#define FTDI_EP_IN     0x81

struct edgecoh_transport {
    libusb_context       *ctx;
    libusb_device_handle *dev;
};

edgecoh_transport_t *edgecoh_transport_open(uint16_t vid, uint16_t pid) {
    edgecoh_transport_t *t = calloc(1, sizeof(*t));
    if (!t) return NULL;

    if (libusb_init(&t->ctx) != 0) {
        free(t);
        return NULL;
    }

    t->dev = libusb_open_device_with_vid_pid(t->ctx, vid, pid);
    if (!t->dev) {
        fprintf(stderr, "edgecoh: cannot open USB device %04x:%04x\n", vid, pid);
        libusb_exit(t->ctx);
        free(t);
        return NULL;
    }

    libusb_detach_kernel_driver(t->dev, FTDI_INTERFACE);
    if (libusb_claim_interface(t->dev, FTDI_INTERFACE) != 0) {
        fprintf(stderr, "edgecoh: cannot claim interface %d\n", FTDI_INTERFACE);
        libusb_close(t->dev);
        libusb_exit(t->ctx);
        free(t);
        return NULL;
    }

    return t;
}

void edgecoh_transport_close(edgecoh_transport_t *t) {
    if (!t) return;
    libusb_release_interface(t->dev, FTDI_INTERFACE);
    libusb_close(t->dev);
    libusb_exit(t->ctx);
    free(t);
}

int edgecoh_transport_send(edgecoh_transport_t *t, const uint8_t *buf, int len) {
    int transferred = 0;
    int rc = libusb_bulk_transfer(t->dev, FTDI_EP_OUT,
                                  (uint8_t *)buf, len, &transferred, 5000);
    if (rc != 0) {
        fprintf(stderr, "edgecoh: USB send error: %s\n", libusb_error_name(rc));
        return -1;
    }
    return transferred;
}

int edgecoh_transport_recv(edgecoh_transport_t *t, uint8_t *buf, int buf_len,
                           int timeout_ms) {
    int transferred = 0;
    int rc = libusb_bulk_transfer(t->dev, FTDI_EP_IN,
                                  buf, buf_len, &transferred, timeout_ms);
    if (rc != 0 && rc != LIBUSB_ERROR_TIMEOUT) {
        fprintf(stderr, "edgecoh: USB recv error: %s\n", libusb_error_name(rc));
        return -1;
    }
    return transferred;
}

int edgecoh_send_msg(edgecoh_transport_t *t, const void *msg) {
    uint8_t buf[512];
    int len = edgecoh_serialize(msg, buf, sizeof(buf));
    if (len < 0) return -1;
    int sent = edgecoh_transport_send(t, buf, len);
    if (sent != len) return -1;
    return 0;
}

int edgecoh_recv_header(edgecoh_transport_t *t, edgecoh_header_t *hdr,
                        int timeout_ms) {
    uint8_t buf[sizeof(edgecoh_header_t)];
    int n = edgecoh_transport_recv(t, buf, sizeof(buf), timeout_ms);
    if (n < (int)sizeof(edgecoh_header_t)) return -1;
    return edgecoh_deserialize_header(buf, n, hdr);
}
```

- [ ] **Step 3: Commit**

```bash
git add protocol/include/edgecoh/transport.h protocol/src/transport_usb.c
git commit -m "feat: USB 2.0 transport layer for EdgeCoh protocol via libusb/FTDI"
```

---

## Phase 2: FPGA NMC Engine

All FPGA tasks are developed and synthesized on the **x86 workstation** using Vivado. Testbenches use xsim.

---

### Task 5: FPGA — USB Interface and EdgeCoh Controller

**Files:**
- Create: `splitinfer/fpga/src/usb_interface.v`
- Create: `splitinfer/fpga/src/edgecoh_controller.v`
- Create: `splitinfer/fpga/sim/tb_edgecoh_controller.v`

- [ ] **Step 1: Write the testbench first**

```verilog
/* splitinfer/fpga/sim/tb_edgecoh_controller.v */
`timescale 1ns / 1ps

module tb_edgecoh_controller;
    reg         clk;
    reg         rst_n;
    reg  [7:0]  rx_data;
    reg         rx_valid;
    wire        rx_ready;
    wire [7:0]  tx_data;
    wire        tx_valid;
    reg         tx_ready;
    wire        nmc_start;
    wire [7:0]  nmc_op;
    wire [31:0] nmc_table_base;
    wire [31:0] nmc_table_rows;
    wire [31:0] nmc_table_cols;
    wire [31:0] nmc_input_addr;
    wire [31:0] nmc_input_len;
    wire [31:0] nmc_output_addr;
    reg         nmc_done;
    wire        dma_wr_en;
    wire [31:0] dma_wr_addr;
    wire [7:0]  dma_wr_data;
    wire        dma_rd_en;
    wire [31:0] dma_rd_addr;
    reg  [7:0]  dma_rd_data;
    reg         dma_rd_valid;
    wire        barrier_ack;

    edgecoh_controller uut (
        .clk(clk), .rst_n(rst_n),
        .rx_data(rx_data), .rx_valid(rx_valid), .rx_ready(rx_ready),
        .tx_data(tx_data), .tx_valid(tx_valid), .tx_ready(tx_ready),
        .nmc_start(nmc_start), .nmc_op(nmc_op),
        .nmc_table_base(nmc_table_base), .nmc_table_rows(nmc_table_rows),
        .nmc_table_cols(nmc_table_cols), .nmc_input_addr(nmc_input_addr),
        .nmc_input_len(nmc_input_len), .nmc_output_addr(nmc_output_addr),
        .nmc_done(nmc_done),
        .dma_wr_en(dma_wr_en), .dma_wr_addr(dma_wr_addr), .dma_wr_data(dma_wr_data),
        .dma_rd_en(dma_rd_en), .dma_rd_addr(dma_rd_addr),
        .dma_rd_data(dma_rd_data), .dma_rd_valid(dma_rd_valid),
        .barrier_ack(barrier_ack)
    );

    always #5 clk = ~clk;

    task send_byte(input [7:0] data);
        begin
            @(posedge clk);
            rx_data  <= data;
            rx_valid <= 1;
            @(posedge clk);
            while (!rx_ready) @(posedge clk);
            rx_valid <= 0;
        end
    endtask

    task send_barrier;
        begin
            send_byte(8'h03); send_byte(8'h00);
            send_byte(8'h00); send_byte(8'h00);
            send_byte(8'h00); send_byte(8'h00);
            send_byte(8'h00); send_byte(8'h00);
        end
    endtask

    task send_nmc_embedding;
        begin
            send_byte(8'h20); send_byte(8'h00);
            send_byte(8'h07); send_byte(8'h00);
            send_byte(8'h19); send_byte(8'h00);
            send_byte(8'h00); send_byte(8'h00);
            send_byte(8'h01);
            send_byte(8'h00); send_byte(8'h00); send_byte(8'h10); send_byte(8'h00);
            send_byte(8'h10); send_byte(8'h27); send_byte(8'h00); send_byte(8'h00);
            send_byte(8'h40); send_byte(8'h00); send_byte(8'h00); send_byte(8'h00);
            send_byte(8'h00); send_byte(8'h00); send_byte(8'h50); send_byte(8'h00);
            send_byte(8'h80); send_byte(8'h00); send_byte(8'h00); send_byte(8'h00);
            send_byte(8'h00); send_byte(8'h00); send_byte(8'h60); send_byte(8'h00);
        end
    endtask

    integer pass_count;

    initial begin
        clk = 0; rst_n = 0;
        rx_data = 0; rx_valid = 0; tx_ready = 1;
        nmc_done = 0; dma_rd_data = 0; dma_rd_valid = 0;
        pass_count = 0;

        #20 rst_n = 1; #20;

        $display("Test 1: SYNC_BARRIER...");
        send_barrier;
        #100;
        pass_count = pass_count + 1;

        #200;

        $display("Test 2: NMC_EXEC embedding lookup...");
        send_nmc_embedding;
        #100;
        @(posedge clk); nmc_done <= 1;
        @(posedge clk); nmc_done <= 0;
        #100;
        pass_count = pass_count + 1;

        $display("All %0d edgecoh_controller tests completed.", pass_count);
        $finish;
    end

    always @(posedge nmc_start) begin
        $display("  NMC started: op=%h table_base=%h rows=%0d cols=%0d",
                 nmc_op, nmc_table_base, nmc_table_rows, nmc_table_cols);
    end
endmodule
```

- [ ] **Step 2: Run testbench to verify it fails**

```bash
cd splitinfer/fpga
xvlog sim/tb_edgecoh_controller.v
```

Expected: FAIL — edgecoh_controller module not found.

- [ ] **Step 3: Write usb_interface.v**

```verilog
/* splitinfer/fpga/src/usb_interface.v */
`timescale 1ns / 1ps

module usb_interface #(
    parameter CLK_FREQ  = 100_000_000,
    parameter BAUD_RATE = 921_600      /* FT2232HQ supports up to 12 Mbaud; 921600 is reliable and fast */
)(
    input  wire       clk,
    input  wire       rst_n,
    input  wire       uart_rx,
    output wire       uart_tx,
    output reg  [7:0] rx_data,
    output reg        rx_valid,
    input  wire       rx_ready,
    input  wire [7:0] tx_data,
    input  wire       tx_valid,
    output reg        tx_ready
);

    localparam CLKS_PER_BIT = CLK_FREQ / BAUD_RATE;

    /* UART RX */
    reg [2:0]  rx_state;
    reg [15:0] rx_clk_count;
    reg [2:0]  rx_bit_idx;
    reg [7:0]  rx_shift;
    reg        uart_rx_r1, uart_rx_r2;

    localparam RX_IDLE = 3'd0, RX_START = 3'd1, RX_DATA = 3'd2, RX_STOP = 3'd3;

    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin uart_rx_r1 <= 1; uart_rx_r2 <= 1; end
        else begin uart_rx_r1 <= uart_rx; uart_rx_r2 <= uart_rx_r1; end
    end

    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            rx_state <= RX_IDLE; rx_valid <= 0; rx_data <= 0;
            rx_clk_count <= 0; rx_bit_idx <= 0; rx_shift <= 0;
        end else begin
            rx_valid <= 0;
            case (rx_state)
                RX_IDLE: if (uart_rx_r2 == 0) begin rx_state <= RX_START; rx_clk_count <= 0; end
                RX_START: if (rx_clk_count == CLKS_PER_BIT/2) begin
                    rx_state <= RX_DATA; rx_clk_count <= 0; rx_bit_idx <= 0;
                end else rx_clk_count <= rx_clk_count + 1;
                RX_DATA: if (rx_clk_count == CLKS_PER_BIT-1) begin
                    rx_shift[rx_bit_idx] <= uart_rx_r2; rx_clk_count <= 0;
                    if (rx_bit_idx == 7) rx_state <= RX_STOP;
                    else rx_bit_idx <= rx_bit_idx + 1;
                end else rx_clk_count <= rx_clk_count + 1;
                RX_STOP: if (rx_clk_count == CLKS_PER_BIT-1) begin
                    rx_data <= rx_shift; rx_valid <= 1; rx_state <= RX_IDLE;
                end else rx_clk_count <= rx_clk_count + 1;
            endcase
        end
    end

    /* UART TX */
    reg [2:0]  tx_state;
    reg [15:0] tx_clk_count;
    reg [2:0]  tx_bit_idx;
    reg [7:0]  tx_shift;
    reg        tx_out;
    assign uart_tx = tx_out;

    localparam TX_IDLE = 3'd0, TX_START = 3'd1, TX_DATA = 3'd2, TX_STOP = 3'd3;

    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            tx_state <= TX_IDLE; tx_ready <= 1; tx_out <= 1;
            tx_clk_count <= 0; tx_bit_idx <= 0; tx_shift <= 0;
        end else begin
            case (tx_state)
                TX_IDLE: begin
                    tx_out <= 1; tx_ready <= 1;
                    if (tx_valid && tx_ready) begin
                        tx_shift <= tx_data; tx_state <= TX_START;
                        tx_ready <= 0; tx_clk_count <= 0;
                    end
                end
                TX_START: begin
                    tx_out <= 0;
                    if (tx_clk_count == CLKS_PER_BIT-1) begin
                        tx_state <= TX_DATA; tx_clk_count <= 0; tx_bit_idx <= 0;
                    end else tx_clk_count <= tx_clk_count + 1;
                end
                TX_DATA: begin
                    tx_out <= tx_shift[tx_bit_idx];
                    if (tx_clk_count == CLKS_PER_BIT-1) begin
                        tx_clk_count <= 0;
                        if (tx_bit_idx == 7) tx_state <= TX_STOP;
                        else tx_bit_idx <= tx_bit_idx + 1;
                    end else tx_clk_count <= tx_clk_count + 1;
                end
                TX_STOP: begin
                    tx_out <= 1;
                    if (tx_clk_count == CLKS_PER_BIT-1) tx_state <= TX_IDLE;
                    else tx_clk_count <= tx_clk_count + 1;
                end
            endcase
        end
    end
endmodule
```

- [ ] **Step 4: Write edgecoh_controller.v**

```verilog
/* splitinfer/fpga/src/edgecoh_controller.v */
`timescale 1ns / 1ps

module edgecoh_controller (
    input  wire        clk, input wire rst_n,
    input  wire [7:0]  rx_data, input wire rx_valid, output reg rx_ready,
    output reg  [7:0]  tx_data, output reg tx_valid, input wire tx_ready,
    output reg         nmc_start, output reg [7:0] nmc_op,
    output reg  [31:0] nmc_table_base, output reg [31:0] nmc_table_rows,
    output reg  [31:0] nmc_table_cols, output reg [31:0] nmc_input_addr,
    output reg  [31:0] nmc_input_len, output reg [31:0] nmc_output_addr,
    input  wire        nmc_done,
    output reg         dma_wr_en, output reg [31:0] dma_wr_addr, output reg [7:0] dma_wr_data,
    output reg         dma_rd_en, output reg [31:0] dma_rd_addr,
    input  wire [7:0]  dma_rd_data, input wire dma_rd_valid,
    output reg         barrier_ack
);

    localparam MSG_SYNC_BARRIER = 8'h03, MSG_NMC_EXEC = 8'h20,
               MSG_TRANSFER_OWNERSHIP = 8'h01, MSG_ACK = 8'hFE;

    localparam S_IDLE = 4'd0, S_HEADER = 4'd1, S_PAYLOAD = 4'd2,
               S_DISPATCH = 4'd3, S_WAIT_NMC = 4'd4, S_SEND_ACK = 4'd5;

    reg [3:0]  state;
    reg [7:0]  header_buf [0:7];
    reg [2:0]  header_idx;
    reg [7:0]  payload_buf [0:31];
    reg [5:0]  payload_idx;
    reg [31:0] payload_len;
    reg [2:0]  ack_byte_idx;

    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            state <= S_IDLE; rx_ready <= 1; tx_valid <= 0;
            nmc_start <= 0; dma_wr_en <= 0; dma_rd_en <= 0;
            barrier_ack <= 0; header_idx <= 0; payload_idx <= 0; ack_byte_idx <= 0;
        end else begin
            nmc_start <= 0; dma_wr_en <= 0; dma_rd_en <= 0;
            barrier_ack <= 0; tx_valid <= 0;

            case (state)
                S_IDLE: begin
                    rx_ready <= 1; header_idx <= 0;
                    if (rx_valid) begin
                        header_buf[0] <= rx_data; header_idx <= 1; state <= S_HEADER;
                    end
                end
                S_HEADER: begin
                    rx_ready <= 1;
                    if (rx_valid) begin
                        header_buf[header_idx] <= rx_data;
                        if (header_idx == 7) begin
                            payload_len <= {rx_data, header_buf[6], header_buf[5], header_buf[4]};
                            payload_idx <= 0;
                            if ({rx_data, header_buf[6], header_buf[5], header_buf[4]} == 0)
                                state <= S_DISPATCH;
                            else state <= S_PAYLOAD;
                        end else header_idx <= header_idx + 1;
                    end
                end
                S_PAYLOAD: begin
                    rx_ready <= 1;
                    if (rx_valid) begin
                        if (payload_idx < 32) payload_buf[payload_idx] <= rx_data;
                        if (payload_idx + 1 >= payload_len[5:0]) state <= S_DISPATCH;
                        else payload_idx <= payload_idx + 1;
                    end
                end
                S_DISPATCH: begin
                    rx_ready <= 0;
                    case (header_buf[0])
                        MSG_SYNC_BARRIER: begin
                            barrier_ack <= 1; state <= S_SEND_ACK; ack_byte_idx <= 0;
                        end
                        MSG_NMC_EXEC: begin
                            nmc_op <= payload_buf[0];
                            nmc_table_base <= {payload_buf[4], payload_buf[3], payload_buf[2], payload_buf[1]};
                            nmc_table_rows <= {payload_buf[8], payload_buf[7], payload_buf[6], payload_buf[5]};
                            nmc_table_cols <= {payload_buf[12], payload_buf[11], payload_buf[10], payload_buf[9]};
                            nmc_input_addr <= {payload_buf[16], payload_buf[15], payload_buf[14], payload_buf[13]};
                            nmc_input_len  <= {payload_buf[20], payload_buf[19], payload_buf[18], payload_buf[17]};
                            nmc_output_addr <= {payload_buf[24], payload_buf[23], payload_buf[22], payload_buf[21]};
                            nmc_start <= 1; state <= S_WAIT_NMC;
                        end
                        default: begin state <= S_SEND_ACK; ack_byte_idx <= 0; end
                    endcase
                end
                S_WAIT_NMC: if (nmc_done) begin state <= S_SEND_ACK; ack_byte_idx <= 0; end
                S_SEND_ACK: if (tx_ready) begin
                    tx_valid <= 1;
                    case (ack_byte_idx)
                        0: tx_data <= MSG_ACK;
                        1: tx_data <= 8'h00;
                        2: tx_data <= header_buf[2];
                        3: tx_data <= header_buf[3];
                        4: tx_data <= 8'h00;
                        5: tx_data <= 8'h00;
                        6: tx_data <= 8'h00;
                        7: begin tx_data <= 8'h00; state <= S_IDLE; end
                    endcase
                    ack_byte_idx <= ack_byte_idx + 1;
                end
            endcase
        end
    end
endmodule
```

- [ ] **Step 5: Run testbench**

```bash
cd splitinfer/fpga
xvlog src/usb_interface.v src/edgecoh_controller.v sim/tb_edgecoh_controller.v
xelab tb_edgecoh_controller -debug typical
xsim tb_edgecoh_controller -runall
```

Expected: Both tests complete, NMC start fires with correct parameters.

- [ ] **Step 6: Commit**

```bash
git add fpga/src/usb_interface.v fpga/src/edgecoh_controller.v fpga/sim/tb_edgecoh_controller.v
git commit -m "feat: FPGA USB interface and EdgeCoh protocol controller FSM"
```

---

### Task 6: FPGA — Embedding Lookup Engine

**Files:**
- Create: `splitinfer/fpga/src/embedding_lookup.v`
- Create: `splitinfer/fpga/sim/tb_embedding_lookup.v`

- [ ] **Step 1: Write the testbench**

```verilog
/* splitinfer/fpga/sim/tb_embedding_lookup.v */
`timescale 1ns / 1ps

module tb_embedding_lookup;
    reg clk, rst_n, start;
    reg [31:0] table_base_addr, embed_dim, indices_addr, num_indices, output_addr;
    wire done;
    wire mem_rd_en; wire [26:0] mem_rd_addr;
    reg [127:0] mem_rd_data; reg mem_rd_valid;
    wire mem_wr_en; wire [26:0] mem_wr_addr; wire [127:0] mem_wr_data;

    embedding_lookup uut (
        .clk(clk), .rst_n(rst_n), .start(start),
        .table_base_addr(table_base_addr), .embed_dim(embed_dim),
        .indices_addr(indices_addr), .num_indices(num_indices),
        .output_addr(output_addr), .done(done),
        .mem_rd_en(mem_rd_en), .mem_rd_addr(mem_rd_addr),
        .mem_rd_data(mem_rd_data), .mem_rd_valid(mem_rd_valid),
        .mem_wr_en(mem_wr_en), .mem_wr_addr(mem_wr_addr), .mem_wr_data(mem_wr_data)
    );

    always #5 clk = ~clk;

    reg [7:0] fake_mem [0:1023];
    reg [3:0] rd_delay; reg rd_pending; reg [26:0] rd_pending_addr;

    always @(posedge clk) begin
        mem_rd_valid <= 0;
        if (mem_rd_en) begin rd_delay <= 4; rd_pending <= 1; rd_pending_addr <= mem_rd_addr; end
        if (rd_pending && rd_delay > 0) begin
            rd_delay <= rd_delay - 1;
            if (rd_delay == 1) begin
                mem_rd_data <= {
                    fake_mem[rd_pending_addr+15], fake_mem[rd_pending_addr+14],
                    fake_mem[rd_pending_addr+13], fake_mem[rd_pending_addr+12],
                    fake_mem[rd_pending_addr+11], fake_mem[rd_pending_addr+10],
                    fake_mem[rd_pending_addr+9],  fake_mem[rd_pending_addr+8],
                    fake_mem[rd_pending_addr+7],  fake_mem[rd_pending_addr+6],
                    fake_mem[rd_pending_addr+5],  fake_mem[rd_pending_addr+4],
                    fake_mem[rd_pending_addr+3],  fake_mem[rd_pending_addr+2],
                    fake_mem[rd_pending_addr+1],  fake_mem[rd_pending_addr+0]
                };
                mem_rd_valid <= 1; rd_pending <= 0;
            end
        end
    end

    integer i;
    initial begin
        clk = 0; rst_n = 0; start = 0; mem_rd_valid = 0; rd_pending = 0;

        for (i = 0; i < 1024; i = i + 1) fake_mem[i] = 0;
        fake_mem[0] = 8'h02; fake_mem[4] = 8'h00;
        for (i = 0; i < 16; i = i + 1) fake_mem[256 + i] = 8'hAA;
        for (i = 0; i < 16; i = i + 1) fake_mem[288 + i] = 8'hCC;

        #20 rst_n = 1; #20;

        table_base_addr <= 32'h100; embed_dim <= 32'd16;
        indices_addr <= 32'h000; num_indices <= 32'd2; output_addr <= 32'h200;

        @(posedge clk); start <= 1; @(posedge clk); start <= 0;

        wait(done); #20;
        $display("Embedding lookup test completed.");
        $finish;
    end
endmodule
```

- [ ] **Step 2: Write embedding_lookup.v**

```verilog
/* splitinfer/fpga/src/embedding_lookup.v */
`timescale 1ns / 1ps

module embedding_lookup (
    input wire clk, input wire rst_n,
    input wire start,
    input wire [31:0] table_base_addr, input wire [31:0] embed_dim,
    input wire [31:0] indices_addr, input wire [31:0] num_indices,
    input wire [31:0] output_addr, output reg done,
    output reg mem_rd_en, output reg [26:0] mem_rd_addr,
    input wire [127:0] mem_rd_data, input wire mem_rd_valid,
    output reg mem_wr_en, output reg [26:0] mem_wr_addr, output reg [127:0] mem_wr_data
);

    localparam S_IDLE=3'd0, S_READ_INDEX=3'd1, S_WAIT_INDEX=3'd2,
               S_READ_EMBED=3'd3, S_WAIT_EMBED=3'd4, S_WRITE_OUT=3'd5, S_DONE=3'd6;

    reg [2:0] state;
    reg [31:0] idx_counter, current_index, burst_counter, bursts_per_embed;

    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin state <= S_IDLE; done <= 0; mem_rd_en <= 0; mem_wr_en <= 0; end
        else begin
            mem_rd_en <= 0; mem_wr_en <= 0; done <= 0;
            case (state)
                S_IDLE: if (start) begin
                    idx_counter <= 0; bursts_per_embed <= embed_dim >> 4; state <= S_READ_INDEX;
                end
                S_READ_INDEX: if (idx_counter >= num_indices) state <= S_DONE;
                else begin
                    mem_rd_en <= 1; mem_rd_addr <= indices_addr[26:0] + (idx_counter << 2);
                    state <= S_WAIT_INDEX;
                end
                S_WAIT_INDEX: if (mem_rd_valid) begin
                    current_index <= mem_rd_data[31:0]; burst_counter <= 0; state <= S_READ_EMBED;
                end
                S_READ_EMBED: if (burst_counter >= bursts_per_embed) begin
                    idx_counter <= idx_counter + 1; state <= S_READ_INDEX;
                end else begin
                    mem_rd_en <= 1;
                    mem_rd_addr <= table_base_addr[26:0] + current_index * embed_dim + (burst_counter << 4);
                    state <= S_WAIT_EMBED;
                end
                S_WAIT_EMBED: if (mem_rd_valid) state <= S_WRITE_OUT;
                S_WRITE_OUT: begin
                    mem_wr_en <= 1;
                    mem_wr_addr <= output_addr[26:0] + idx_counter * embed_dim + (burst_counter << 4);
                    mem_wr_data <= mem_rd_data;
                    burst_counter <= burst_counter + 1; state <= S_READ_EMBED;
                end
                S_DONE: begin done <= 1; state <= S_IDLE; end
            endcase
        end
    end
endmodule
```

- [ ] **Step 3: Run testbench**

```bash
cd splitinfer/fpga
xvlog src/embedding_lookup.v sim/tb_embedding_lookup.v
xelab tb_embedding_lookup -debug typical
xsim tb_embedding_lookup -runall
```

Expected: Test completes, done signal asserted.

- [ ] **Step 4: Commit**

```bash
git add fpga/src/embedding_lookup.v fpga/sim/tb_embedding_lookup.v
git commit -m "feat: FPGA embedding table lookup engine with DDR2 read/write"
```

---

### Task 7: FPGA — 8x8 INT8 MAC Array

**Files:**
- Create: `splitinfer/fpga/src/mac_array_8x8.v`
- Create: `splitinfer/fpga/sim/tb_mac_array.v`

- [ ] **Step 1: Write the testbench**

```verilog
/* splitinfer/fpga/sim/tb_mac_array.v */
`timescale 1ns / 1ps

module tb_mac_array;
    reg clk, rst_n, start, load_a, load_b;
    reg [63:0] row_a, row_b;
    wire [31:0] result [0:7];
    wire done;

    mac_array_8x8 uut (
        .clk(clk), .rst_n(rst_n), .start(start),
        .load_a(load_a), .load_b(load_b), .row_a(row_a), .row_b(row_b),
        .result_0(result[0]), .result_1(result[1]), .result_2(result[2]), .result_3(result[3]),
        .result_4(result[4]), .result_5(result[5]), .result_6(result[6]), .result_7(result[7]),
        .done(done)
    );

    always #5 clk = ~clk;
    integer i;

    initial begin
        clk = 0; rst_n = 0; start = 0; load_a = 0; load_b = 0; row_a = 0; row_b = 0;
        #20 rst_n = 1; #10;

        @(posedge clk); start <= 1; @(posedge clk); start <= 0;

        @(posedge clk); load_a <= 1;
        row_a <= {8'd8, 8'd7, 8'd6, 8'd5, 8'd4, 8'd3, 8'd2, 8'd1};
        @(posedge clk); load_a <= 0;

        @(posedge clk); load_b <= 1;
        row_b <= {8'd1, 8'd1, 8'd1, 8'd1, 8'd1, 8'd1, 8'd1, 8'd1};
        @(posedge clk); load_b <= 0;

        #50;

        for (i = 0; i < 8; i = i + 1) $display("result[%0d] = %0d", i, result[i]);
        $display("MAC array test completed.");
        $finish;
    end
endmodule
```

- [ ] **Step 2: Write mac_array_8x8.v**

```verilog
/* splitinfer/fpga/src/mac_array_8x8.v */
`timescale 1ns / 1ps

module mac_array_8x8 (
    input wire clk, input wire rst_n,
    input wire start, input wire load_a, input wire load_b,
    input wire [63:0] row_a, input wire [63:0] row_b,
    output wire [31:0] result_0, result_1, result_2, result_3,
    output wire [31:0] result_4, result_5, result_6, result_7,
    output reg done
);

    reg signed [7:0] a_reg [0:7];
    reg signed [7:0] b_reg [0:7];
    reg signed [31:0] acc [0:7];
    reg mac_valid;

    assign result_0 = acc[0]; assign result_1 = acc[1];
    assign result_2 = acc[2]; assign result_3 = acc[3];
    assign result_4 = acc[4]; assign result_5 = acc[5];
    assign result_6 = acc[6]; assign result_7 = acc[7];

    integer i;

    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            for (i = 0; i < 8; i = i+1) begin a_reg[i] <= 0; b_reg[i] <= 0; end
            mac_valid <= 0;
        end else begin
            mac_valid <= 0;
            if (load_a) for (i = 0; i < 8; i = i+1) a_reg[i] <= $signed(row_a[i*8 +: 8]);
            if (load_b) begin
                for (i = 0; i < 8; i = i+1) b_reg[i] <= $signed(row_b[i*8 +: 8]);
                mac_valid <= 1;
            end
        end
    end

    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin for (i = 0; i < 8; i = i+1) acc[i] <= 0; done <= 0; end
        else begin
            done <= 0;
            if (start) for (i = 0; i < 8; i = i+1) acc[i] <= 0;
            if (mac_valid) begin
                for (i = 0; i < 8; i = i+1)
                    acc[i] <= acc[i] + a_reg[0]*b_reg[0] + a_reg[1]*b_reg[1]
                            + a_reg[2]*b_reg[2] + a_reg[3]*b_reg[3]
                            + a_reg[4]*b_reg[4] + a_reg[5]*b_reg[5]
                            + a_reg[6]*b_reg[6] + a_reg[7]*b_reg[7];
                done <= 1;
            end
        end
    end
endmodule
```

- [ ] **Step 3: Run testbench**

```bash
xvlog fpga/src/mac_array_8x8.v fpga/sim/tb_mac_array.v
xelab tb_mac_array -debug typical
xsim tb_mac_array -runall
```

Expected: `result[0..7] = 36`

- [ ] **Step 4: Commit**

```bash
git add fpga/src/mac_array_8x8.v fpga/sim/tb_mac_array.v
git commit -m "feat: 8x8 INT8 MAC array using DSP48E1 slices"
```

---

### Task 8: FPGA — Element-wise Operations and NMC Dispatch

**Files:**
- Create: `splitinfer/fpga/src/elementwise.v`
- Create: `splitinfer/fpga/src/nmc_dispatch.v`
- Create: `splitinfer/fpga/sim/tb_elementwise.v`

- [ ] **Step 1: Write tb_elementwise.v**

```verilog
/* splitinfer/fpga/sim/tb_elementwise.v */
`timescale 1ns / 1ps

module tb_elementwise;
    reg clk, rst_n, start; reg [1:0] op; reg [127:0] data_in; reg [7:0] scale_factor;
    wire [127:0] data_out; wire done;

    elementwise uut (.clk(clk), .rst_n(rst_n), .start(start), .op(op),
        .data_in(data_in), .scale_factor(scale_factor), .data_out(data_out), .done(done));

    always #5 clk = ~clk;

    initial begin
        clk = 0; rst_n = 0; start = 0; #20 rst_n = 1; #10;

        op <= 2'b00; /* ReLU */
        data_in <= {8'd0,8'd0,8'd0,8'd0, 8'd0,8'd0,8'd0,8'd0,
                    8'd10,8'd127,8'h80,8'd0, 8'd7,8'hFF,8'd3,8'hFB};
        @(posedge clk); start <= 1; @(posedge clk); start <= 0;
        wait(done); #10;
        $display("ReLU output: %h", data_out);

        op <= 2'b10; scale_factor <= 8'd2;
        data_in <= {8'd0,8'd0,8'd0,8'd0, 8'd0,8'd0,8'd0,8'd0,
                    8'd0,8'd0,8'd0,8'd0, 8'd4,8'd3,8'd2,8'd1};
        @(posedge clk); start <= 1; @(posedge clk); start <= 0;
        wait(done); #10;
        $display("Scale output: %h", data_out);
        $display("Elementwise tests completed.");
        $finish;
    end
endmodule
```

- [ ] **Step 2: Write elementwise.v**

```verilog
/* splitinfer/fpga/src/elementwise.v */
`timescale 1ns / 1ps

module elementwise (
    input wire clk, input wire rst_n, input wire start,
    input wire [1:0] op, input wire [127:0] data_in, input wire [7:0] scale_factor,
    output reg [127:0] data_out, output reg done
);
    integer i; reg signed [7:0] val; reg signed [15:0] product;

    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin data_out <= 0; done <= 0; end
        else begin
            done <= 0;
            if (start) begin
                for (i = 0; i < 16; i = i+1) begin
                    val = $signed(data_in[i*8 +: 8]);
                    case (op)
                        2'b00: data_out[i*8 +: 8] <= (val < 0) ? 8'd0 : data_in[i*8 +: 8];
                        2'b01: begin
                            product = val + $signed({1'b0, scale_factor});
                            data_out[i*8 +: 8] <= (product > 127) ? 8'd127 :
                                                   (product < -128) ? 8'h80 : product[7:0];
                        end
                        2'b10: begin
                            product = val * $signed({1'b0, scale_factor});
                            data_out[i*8 +: 8] <= (product > 127) ? 8'd127 :
                                                   (product < -128) ? 8'h80 : product[7:0];
                        end
                        default: data_out[i*8 +: 8] <= data_in[i*8 +: 8];
                    endcase
                end
                done <= 1;
            end
        end
    end
endmodule
```

- [ ] **Step 3: Write nmc_dispatch.v**

```verilog
/* splitinfer/fpga/src/nmc_dispatch.v */
`timescale 1ns / 1ps

module nmc_dispatch (
    input wire clk, input wire rst_n,
    input wire nmc_start, input wire [7:0] nmc_op,
    input wire [31:0] nmc_table_base, nmc_table_rows, nmc_table_cols,
    input wire [31:0] nmc_input_addr, nmc_input_len, nmc_output_addr,
    output wire nmc_done,
    output reg emb_start, output reg [31:0] emb_table_base, emb_embed_dim,
    output reg [31:0] emb_indices_addr, emb_num_indices, emb_output_addr,
    input wire emb_done,
    output reg mac_start, input wire mac_done
);

    localparam NMC_EMBEDDING = 8'h01, NMC_INT8_FC = 8'h02;

    reg [7:0] active_op; reg busy;

    assign nmc_done = (!busy) ? 1'b0 :
                      (active_op == NMC_EMBEDDING) ? emb_done :
                      (active_op == NMC_INT8_FC) ? mac_done : 1'b0;

    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin emb_start <= 0; mac_start <= 0; busy <= 0; end
        else begin
            emb_start <= 0; mac_start <= 0;
            if (nmc_start && !busy) begin
                active_op <= nmc_op; busy <= 1;
                case (nmc_op)
                    NMC_EMBEDDING: begin
                        emb_start <= 1; emb_table_base <= nmc_table_base;
                        emb_embed_dim <= nmc_table_cols; emb_indices_addr <= nmc_input_addr;
                        emb_num_indices <= nmc_input_len; emb_output_addr <= nmc_output_addr;
                    end
                    NMC_INT8_FC: mac_start <= 1;
                    default: busy <= 0;
                endcase
            end
            if (busy && nmc_done) busy <= 0;
        end
    end
endmodule
```

- [ ] **Step 4: Run testbench**

```bash
xvlog fpga/src/elementwise.v fpga/sim/tb_elementwise.v
xelab tb_elementwise -debug typical
xsim tb_elementwise -runall
```

- [ ] **Step 5: Commit**

```bash
git add fpga/src/elementwise.v fpga/src/nmc_dispatch.v fpga/sim/tb_elementwise.v
git commit -m "feat: element-wise ops and NMC dispatch module"
```

---

### Task 9: FPGA — Top-Level Integration and Constraints

> **IMPORTANT CLOCK DOMAIN NOTE:** The MIG DDR2 IP generates its own user clock
> (`ui_clk`, ~81.25 MHz) which is different from the 100 MHz system clock. The
> MIG also requires a 200 MHz reference clock input, generated from the 100 MHz
> board clock via a Clocking Wizard IP. The ddr2_arbiter and all NMC compute
> modules that access DDR2 must run in the `ui_clk` domain. The USB/UART
> interface runs in the 100 MHz `sys_clk` domain. Clock-domain crossing (CDC)
> FIFOs bridge the two domains in the EdgeCoh controller.

**Files:**
- Create: `splitinfer/fpga/src/top.v`
- Create: `splitinfer/fpga/src/ddr2_arbiter.v`
- Create: `splitinfer/fpga/constraints/nexys4ddr.xdc`
- Create: `splitinfer/fpga/sim/tb_top.v`

**Vivado IP cores to generate (via Vivado GUI or TCL):**
- **MIG 7 Series DDR2 IP** — configured for MT47H64M16HR-25:H, 16-bit data, BL8, 4:1 PHY ratio, 128-bit user interface
- **Clocking Wizard IP** — 100 MHz input → 200 MHz output (sys_clk_i for MIG)

- [ ] **Step 1: Write ddr2_arbiter.v**

NOTE: This module runs in the **MIG ui_clk domain (~81.25 MHz)**, NOT the 100 MHz system clock. The MIG presents a 128-bit user data interface (BL8 × 16-bit DDR2 = 128-bit).

```verilog
/* splitinfer/fpga/src/ddr2_arbiter.v
 *
 * Arbitrates DDR2 access between NMC compute units and EdgeCoh DMA.
 * CLOCK DOMAIN: Runs on MIG ui_clk (~81.25 MHz), NOT sys_clk (100 MHz).
 * MIG user interface: 128-bit data (BL8 x 16-bit DDR2), ~27-bit address.
 */
`timescale 1ns / 1ps

module ddr2_arbiter (
    input wire clk,       /* MIG ui_clk (~81.25 MHz) */
    input wire rst_n,     /* MIG ui_clk_sync_rst (active-high from MIG, invert externally) */
    output reg [26:0] app_addr, output reg [2:0] app_cmd,
    output reg app_en, output reg [127:0] app_wdf_data, output reg app_wdf_wren,
    output reg         app_wdf_end,  /* MIG DDR2 requires wdf_end asserted with wdf_wren for BL8 */
    input wire [127:0] app_rd_data, input wire app_rd_data_valid,
    input wire app_rdy, input wire app_wdf_rdy,
    input wire nmc_rd_en, input wire [26:0] nmc_rd_addr,
    output wire [127:0] nmc_rd_data, output wire nmc_rd_valid,
    input wire nmc_wr_en, input wire [26:0] nmc_wr_addr, input wire [127:0] nmc_wr_data,
    input wire dma_rd_en, input wire [26:0] dma_rd_addr,
    output wire [127:0] dma_rd_data, output wire dma_rd_valid,
    input wire dma_wr_en, input wire [26:0] dma_wr_addr, input wire [7:0] dma_wr_byte
);

    assign nmc_rd_data = app_rd_data;
    assign dma_rd_data = app_rd_data;

    reg nmc_rd_pending, dma_rd_pending;
    assign nmc_rd_valid = app_rd_data_valid && nmc_rd_pending;
    assign dma_rd_valid = app_rd_data_valid && dma_rd_pending;

    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            app_en <= 0; app_wdf_wren <= 0; app_wdf_end <= 0;
            nmc_rd_pending <= 0; dma_rd_pending <= 0;
        end else begin
            app_en <= 0; app_wdf_wren <= 0; app_wdf_end <= 0;
            if (app_rd_data_valid) begin nmc_rd_pending <= 0; dma_rd_pending <= 0; end

            /* NMC has priority over DMA */
            if (nmc_wr_en && app_rdy && app_wdf_rdy) begin
                app_addr <= nmc_wr_addr; app_cmd <= 3'b000; app_en <= 1;
                app_wdf_data <= nmc_wr_data; app_wdf_wren <= 1; app_wdf_end <= 1;
            end else if (nmc_rd_en && app_rdy) begin
                app_addr <= nmc_rd_addr; app_cmd <= 3'b001; app_en <= 1; nmc_rd_pending <= 1;
            end else if (dma_wr_en && app_rdy && app_wdf_rdy) begin
                app_addr <= dma_wr_addr; app_cmd <= 3'b000; app_en <= 1;
                app_wdf_data <= {120'd0, dma_wr_byte}; app_wdf_wren <= 1; app_wdf_end <= 1;
            end else if (dma_rd_en && app_rdy) begin
                app_addr <= dma_rd_addr; app_cmd <= 3'b001; app_en <= 1; dma_rd_pending <= 1;
            end
        end
    end
endmodule
```

- [ ] **Step 2: Write top.v**

The top-level module handles TWO clock domains:
- `sys_clk` (100 MHz) — USB/UART interface, EdgeCoh byte-stream side
- `ui_clk` (~81.25 MHz from MIG) — DDR2 arbiter, NMC compute engines

A CDC (clock-domain crossing) FIFO bridges the EdgeCoh controller's command output (sys_clk) to the NMC dispatch (ui_clk). For initial simulation without MIG, ui_clk is stubbed as sys_clk.

```
Key top.v architecture:

  100 MHz sys_clk domain          |  ~81.25 MHz ui_clk domain (from MIG)
  ─────────────────────────────── | ──────────────────────────────────────
  usb_interface                   |
  edgecoh_controller              |
       │                          |
       ▼                          |
  [CDC FIFO: cmd/data crossing] ──┼──> nmc_dispatch
                                  |        ├── embedding_lookup
                                  |        ├── mac_array_8x8
                                  |        └── elementwise
                                  |    ddr2_arbiter
                                  |        │
                                  |    [MIG DDR2 IP]
                                  |        │
                                  |    DDR2 SDRAM (128MB)

  Clocking Wizard: 100 MHz → 200 MHz → MIG sys_clk_i
  MIG outputs: ui_clk (~81.25 MHz), ui_clk_sync_rst, init_calib_complete
```

The full top.v instantiation connects all modules. **Critical: the design must wait for `init_calib_complete` from MIG before issuing any DDR2 commands.** The NMC dispatch holds off until calibration is done.

LED assignments:
- `led[0]` = init_calib_complete (DDR2 ready)
- `led[1]` = nmc_start (NMC activity)
- `led[2]` = barrier_ack (EdgeCoh barrier)
- `led[3]` = heartbeat (design loaded)

- [ ] **Step 3: Write nexys4ddr.xdc**

```xdc
## splitinfer/fpga/constraints/nexys4ddr.xdc
## Digilent Nexys 4 DDR (XC7A100T-1CSG324C)
## Pin assignments verified against official Nexys-4-DDR-Master.xdc

## System clock (100 MHz crystal oscillator)
set_property -dict {PACKAGE_PIN E3 IOSTANDARD LVCMOS33} [get_ports sys_clk]
create_clock -period 10.000 -name sys_clk [get_ports sys_clk]

## Reset button (active-low)
set_property -dict {PACKAGE_PIN C12 IOSTANDARD LVCMOS33} [get_ports sys_rst_n]

## USB-UART (FTDI FT2232HQ Channel B)
## uart_rx = FPGA receives data FROM FTDI (signal name: uart_txd_in in Digilent XDC)
## uart_tx = FPGA transmits data TO FTDI (signal name: uart_rxd_out in Digilent XDC)
set_property -dict {PACKAGE_PIN C4 IOSTANDARD LVCMOS33} [get_ports uart_rx]
set_property -dict {PACKAGE_PIN D4 IOSTANDARD LVCMOS33} [get_ports uart_tx]

## LEDs (accent accent accent accent accent accent accent accent accent accent accent standard accent standard accent 16 standard LEDs, accent using first 4)
set_property -dict {PACKAGE_PIN H17 IOSTANDARD LVCMOS33} [get_ports {led[0]}]
set_property -dict {PACKAGE_PIN K15 IOSTANDARD LVCMOS33} [get_ports {led[1]}]
set_property -dict {PACKAGE_PIN J13 IOSTANDARD LVCMOS33} [get_ports {led[2]}]
set_property -dict {PACKAGE_PIN N14 IOSTANDARD LVCMOS33} [get_ports {led[3]}]

## NOTE: DDR2 pin constraints are generated automatically by the MIG IP core.
## Do NOT manually constrain DDR2 pins — MIG handles this via its own UCF/XDC.
```

- [ ] **Step 4: Write tb_top.v**

For simulation, the MIG DDR2 IP is replaced with a behavioral DDR2 memory model. The testbench stubs `ui_clk = sys_clk` and ties `init_calib_complete = 1` for functional verification.

```verilog
/* splitinfer/fpga/sim/tb_top.v */
`timescale 1ns / 1ps
module tb_top;
    reg sys_clk, sys_rst_n, uart_rx; wire uart_tx; wire [3:0] led;

    /* For simulation: stub MIG by using sys_clk as ui_clk */
    top #(.SIM_MODE(1)) uut (
        .sys_clk(sys_clk), .sys_rst_n(sys_rst_n),
        .uart_rx(uart_rx), .uart_tx(uart_tx), .led(led)
    );

    always #5 sys_clk = ~sys_clk; /* 100 MHz */

    initial begin
        sys_clk = 0; sys_rst_n = 0; uart_rx = 1;
        #100 sys_rst_n = 1;
        #2000; /* allow time for simulated calibration */

        if (led[3]) $display("PASS: Heartbeat LED active");
        else $display("FAIL: Heartbeat LED not active");

        if (led[0]) $display("PASS: DDR2 calibration complete (simulated)");
        else $display("FAIL: DDR2 calibration not signaled");

        $finish;
    end
endmodule
```

- [ ] **Step 5: Run integration testbench**

```bash
cd splitinfer/fpga
xvlog src/*.v sim/tb_top.v
xelab tb_top -debug typical
xsim tb_top -runall
```

Expected: `PASS: Heartbeat LED active` and `PASS: DDR2 calibration complete (simulated)`

- [ ] **Step 6: Commit**

```bash
git add fpga/src/top.v fpga/src/ddr2_arbiter.v fpga/constraints/nexys4ddr.xdc fpga/sim/tb_top.v
git commit -m "feat: FPGA top-level with dual clock domains (sys_clk + MIG ui_clk) and Nexys 4 DDR constraints"
```

---

## Phase 3: Model Partitioning Engine (Python)

---

### Task 10: ONNX Graph Parser

**Files:**
- Create: `splitinfer/partitioner/__init__.py`
- Create: `splitinfer/partitioner/graph.py`
- Create: `splitinfer/partitioner/tests/__init__.py`
- Create: `splitinfer/partitioner/tests/test_graph.py`

- [ ] **Step 1: Write the failing test**

```python
# splitinfer/partitioner/tests/test_graph.py
import pytest
import numpy as np
import onnx
from onnx import helper, TensorProto
from partitioner.graph import parse_onnx_graph, LayerInfo

def _make_simple_model():
    X = helper.make_tensor_value_info("X", TensorProto.FLOAT, [1, 784])
    Y = helper.make_tensor_value_info("Y", TensorProto.FLOAT, [1, 10])
    W1 = helper.make_tensor("W1", TensorProto.FLOAT, [784, 128],
                            np.zeros([784, 128], dtype=np.float32).flatten().tolist())
    B1 = helper.make_tensor("B1", TensorProto.FLOAT, [128],
                            np.zeros([128], dtype=np.float32).tolist())
    W2 = helper.make_tensor("W2", TensorProto.FLOAT, [128, 10],
                            np.zeros([128, 10], dtype=np.float32).flatten().tolist())
    B2 = helper.make_tensor("B2", TensorProto.FLOAT, [10],
                            np.zeros([10], dtype=np.float32).tolist())
    nodes = [
        helper.make_node("MatMul", ["X", "W1"], ["mm1"], name="fc1_matmul"),
        helper.make_node("Add", ["mm1", "B1"], ["fc1_out"], name="fc1_add"),
        helper.make_node("Relu", ["fc1_out"], ["relu_out"], name="relu1"),
        helper.make_node("MatMul", ["relu_out", "W2"], ["mm2"], name="fc2_matmul"),
        helper.make_node("Add", ["mm2", "B2"], ["Y"], name="fc2_add"),
    ]
    graph = helper.make_graph(nodes, "simple_fc", [X], [Y], initializer=[W1, B1, W2, B2])
    return helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)])

def test_parse_returns_layers():
    layers = parse_onnx_graph(_make_simple_model())
    assert len(layers) == 5

def test_layer_info_fields():
    layers = parse_onnx_graph(_make_simple_model())
    fc1 = layers[0]
    assert isinstance(fc1, LayerInfo)
    assert fc1.name == "fc1_matmul"
    assert fc1.op_type == "MatMul"
    assert fc1.weight_bytes == 784 * 128 * 4

def test_topological_order():
    layers = parse_onnx_graph(_make_simple_model())
    names = [l.name for l in layers]
    assert names.index("fc1_matmul") < names.index("relu1") < names.index("fc2_matmul")

def test_relu_has_zero_weights():
    layers = parse_onnx_graph(_make_simple_model())
    relu = [l for l in layers if l.name == "relu1"][0]
    assert relu.weight_bytes == 0
```

- [ ] **Step 2: Run test to verify it fails**

```bash
cd splitinfer && python -m pytest partitioner/tests/test_graph.py -v
```

- [ ] **Step 3: Write implementation**

```python
# splitinfer/partitioner/graph.py
from dataclasses import dataclass
import onnx
import numpy as np

@dataclass
class LayerInfo:
    name: str
    op_type: str
    input_names: list[str]
    output_names: list[str]
    weight_bytes: int
    output_tensor_bytes: int

def parse_onnx_graph(model: onnx.ModelProto) -> list[LayerInfo]:
    graph = model.graph
    init_sizes: dict[str, int] = {}
    for init in graph.initializer:
        dtype = onnx.mapping.TENSOR_TYPE_MAP[init.data_type].np_dtype
        num_elements = int(np.prod(init.dims)) if init.dims else 0
        init_sizes[init.name] = num_elements * dtype.itemsize

    layers = []
    for node in graph.node:
        weight_bytes = sum(init_sizes.get(inp, 0) for inp in node.input)
        layers.append(LayerInfo(
            name=node.name, op_type=node.op_type,
            input_names=list(node.input), output_names=list(node.output),
            weight_bytes=weight_bytes, output_tensor_bytes=0,
        ))
    return layers
```

- [ ] **Step 4: Run tests**

```bash
cd splitinfer && python -m pytest partitioner/tests/test_graph.py -v
```

- [ ] **Step 5: Commit**

```bash
git add partitioner/__init__.py partitioner/tests/__init__.py partitioner/graph.py partitioner/tests/test_graph.py
git commit -m "feat: ONNX graph parser extracting layer info with weight sizes"
```

---

### Task 11: Cost Model

**Files:**
- Create: `splitinfer/partitioner/cost_model.py`
- Create: `splitinfer/partitioner/tests/test_cost_model.py`

- [ ] **Step 1: Write the failing test**

```python
# splitinfer/partitioner/tests/test_cost_model.py
import pytest
from partitioner.graph import LayerInfo
from partitioner.cost_model import CostModel, HardwareParams

def _hw():
    return HardwareParams(gpu_gflops=100.0, fpga_int8_gops=6.4, fpga_ddr2_bw_gbps=1.3,
                          usb_bw_mbps=40.0, usb_latency_ms=1.0, fpga_ddr2_capacity_mb=128.0)

def test_gpu_cost_matmul():
    assert CostModel(_hw()).gpu_time_ms(
        LayerInfo("fc1","MatMul",["x","W"],["y"],784*128*4,128*4)) > 0

def test_fpga_cost_embedding():
    assert CostModel(_hw()).fpga_time_ms(
        LayerInfo("emb","Gather",["t","i"],["o"],10000*64,128*64)) > 0

def test_transfer_cost():
    assert abs(CostModel(_hw()).transfer_time_ms(2*1024*1024) - 51.0) < 1.0

def test_fpga_over_capacity():
    assert CostModel(_hw()).fpga_feasible(
        LayerInfo("big","MatMul",["x","W"],["y"],200*1024*1024,1024)) is False

def test_fpga_within_capacity():
    assert CostModel(_hw()).fpga_feasible(
        LayerInfo("sm","Gather",["t","i"],["o"],50*1024*1024,1024)) is True
```

- [ ] **Step 2: Write implementation**

```python
# splitinfer/partitioner/cost_model.py
from dataclasses import dataclass
from partitioner.graph import LayerInfo

@dataclass
class HardwareParams:
    gpu_gflops: float
    fpga_int8_gops: float
    fpga_ddr2_bw_gbps: float
    usb_bw_mbps: float
    usb_latency_ms: float
    fpga_ddr2_capacity_mb: float

FPGA_MEMORY_BOUND_OPS = {"Gather"}

class CostModel:
    def __init__(self, hw: HardwareParams):
        self.hw = hw

    def gpu_time_ms(self, layer: LayerInfo) -> float:
        flops = self._estimate_flops(layer)
        if flops == 0: return 0.01
        return (flops / (self.hw.gpu_gflops * 1e9)) * 1000.0

    def fpga_time_ms(self, layer: LayerInfo) -> float:
        if layer.op_type in FPGA_MEMORY_BOUND_OPS:
            read_bytes = layer.weight_bytes + layer.output_tensor_bytes
            return (read_bytes / (self.hw.fpga_ddr2_bw_gbps * 1e9)) * 1000.0
        ops = self._estimate_int8_ops(layer)
        if ops == 0: return 0.01
        return (ops / (self.hw.fpga_int8_gops * 1e9)) * 1000.0

    def transfer_time_ms(self, tensor_bytes: int) -> float:
        return (tensor_bytes / (self.hw.usb_bw_mbps * 1e6)) * 1000.0 + self.hw.usb_latency_ms

    def fpga_feasible(self, layer: LayerInfo) -> bool:
        return layer.weight_bytes <= self.hw.fpga_ddr2_capacity_mb * 1024 * 1024

    def _estimate_flops(self, layer: LayerInfo) -> float:
        if layer.op_type in ("MatMul", "Gemm"): return 2 * layer.weight_bytes / 4
        if layer.op_type in ("Relu", "Add"): return layer.output_tensor_bytes / 4
        if layer.op_type == "Gather": return 0
        return layer.weight_bytes / 4

    def _estimate_int8_ops(self, layer: LayerInfo) -> float:
        if layer.op_type in ("MatMul", "Gemm"): return 2 * layer.weight_bytes / 4
        if layer.op_type in ("Relu", "Add"): return layer.output_tensor_bytes
        return 0
```

- [ ] **Step 3: Run tests and commit**

```bash
python -m pytest partitioner/tests/test_cost_model.py -v
git add partitioner/cost_model.py partitioner/tests/test_cost_model.py
git commit -m "feat: cost model for GPU/FPGA/transfer time estimation"
```

---

### Task 12: DP Partitioning Solver

**Files:**
- Create: `splitinfer/partitioner/solver.py`
- Create: `splitinfer/partitioner/tests/test_solver.py`

- [ ] **Step 1: Write the failing test**

```python
# splitinfer/partitioner/tests/test_solver.py
import pytest
from partitioner.graph import LayerInfo
from partitioner.cost_model import CostModel, HardwareParams
from partitioner.solver import partition_model, PartitionResult

def _hw():
    return HardwareParams(gpu_gflops=100.0, fpga_int8_gops=6.4, fpga_ddr2_bw_gbps=1.3,
                          usb_bw_mbps=40.0, usb_latency_ms=1.0, fpga_ddr2_capacity_mb=128.0)

def _layers():
    return [
        LayerInfo("fc1","MatMul",["x","W1"],["y1"],50*1024*1024,512),
        LayerInfo("relu1","Relu",["y1"],["r1"],0,512),
        LayerInfo("fc2","MatMul",["r1","W2"],["y2"],1*1024*1024,256),
        LayerInfo("relu2","Relu",["y2"],["r2"],0,256),
        LayerInfo("fc3","MatMul",["r2","W3"],["y3"],512*1024,40),
    ]

def test_partition_returns_result():
    result = partition_model(_layers(), CostModel(_hw()))
    assert isinstance(result, PartitionResult)
    assert len(result.assignments) == 5

def test_all_assignments_valid():
    for a in partition_model(_layers(), CostModel(_hw())).assignments:
        assert a in ("gpu", "fpga")

def test_large_weight_prefers_fpga():
    assert partition_model(_layers(), CostModel(_hw())).assignments[0] == "fpga"

def test_capacity_respected():
    layers = _layers()
    layers[0] = LayerInfo("fc1","MatMul",["x","W1"],["y1"],100*1024*1024,512)
    layers[2] = LayerInfo("fc2","MatMul",["r1","W2"],["y2"],50*1024*1024,256)
    result = partition_model(layers, CostModel(_hw()))
    fpga_bytes = sum(layers[i].weight_bytes for i, a in enumerate(result.assignments) if a == "fpga")
    assert fpga_bytes <= 128 * 1024 * 1024

def test_total_latency():
    assert partition_model(_layers(), CostModel(_hw())).total_latency_ms > 0
```

- [ ] **Step 2: Write implementation**

```python
# splitinfer/partitioner/solver.py
from dataclasses import dataclass
from partitioner.graph import LayerInfo
from partitioner.cost_model import CostModel

@dataclass
class PartitionResult:
    assignments: list[str]
    total_latency_ms: float
    gpu_memory_bytes: int
    fpga_memory_bytes: int
    num_transfers: int

def partition_model(layers: list[LayerInfo], cost_model: CostModel) -> PartitionResult:
    n = len(layers)
    capacity = int(cost_model.hw.fpga_ddr2_capacity_mb * 1024 * 1024)
    bucket_size = 1024 * 1024
    num_buckets = capacity // bucket_size + 1
    INF = float("inf")

    prev = [[(INF, []) for _ in range(2)] for _ in range(num_buckets)]

    layer = layers[0]
    wb_b = (layer.weight_bytes + bucket_size - 1) // bucket_size

    prev[0][0] = (cost_model.gpu_time_ms(layer), ["gpu"])
    if wb_b < num_buckets and cost_model.fpga_feasible(layer):
        prev[wb_b][1] = (cost_model.fpga_time_ms(layer), ["fpga"])

    for i in range(1, n):
        layer = layers[i]
        wb_b = (layer.weight_bytes + bucket_size - 1) // bucket_size
        xfer = cost_model.transfer_time_ms(layer.output_tensor_bytes)
        curr = [[(INF, []) for _ in range(2)] for _ in range(num_buckets)]

        for b in range(num_buckets):
            for d in range(2):
                pc, pa = prev[b][d]
                if pc == INF: continue
                cg = pc + cost_model.gpu_time_ms(layer) + (xfer if d == 1 else 0)
                if cg < curr[b][0][0]: curr[b][0] = (cg, pa + ["gpu"])
                nb = b + wb_b
                if nb < num_buckets and cost_model.fpga_feasible(layer):
                    cf = pc + cost_model.fpga_time_ms(layer) + (xfer if d == 0 else 0)
                    if cf < curr[nb][1][0]: curr[nb][1] = (cf, pa + ["fpga"])
        prev = curr

    best_cost, best_assign = INF, []
    for b in range(num_buckets):
        for d in range(2):
            c, a = prev[b][d]
            if c < best_cost: best_cost, best_assign = c, a

    return PartitionResult(
        assignments=best_assign, total_latency_ms=best_cost,
        gpu_memory_bytes=sum(layers[i].weight_bytes for i, a in enumerate(best_assign) if a == "gpu"),
        fpga_memory_bytes=sum(layers[i].weight_bytes for i, a in enumerate(best_assign) if a == "fpga"),
        num_transfers=sum(1 for i in range(1, len(best_assign)) if best_assign[i] != best_assign[i-1]),
    )
```

- [ ] **Step 3: Run tests and commit**

```bash
python -m pytest partitioner/tests/test_solver.py -v
git add partitioner/solver.py partitioner/tests/test_solver.py
git commit -m "feat: DP partitioning solver with FPGA capacity constraints"
```

---

### Task 13: Partition Manifest Generator

**Files:**
- Create: `splitinfer/partitioner/manifest.py`
- Create: `splitinfer/partitioner/tests/test_manifest.py`

- [ ] **Step 1: Write test**

```python
# splitinfer/partitioner/tests/test_manifest.py
import json, pytest
from partitioner.graph import LayerInfo
from partitioner.solver import PartitionResult
from partitioner.manifest import generate_manifest

def _data():
    layers = [
        LayerInfo("fc1","MatMul",["x","W1"],["y1"],50_000_000,512),
        LayerInfo("relu1","Relu",["y1"],["r1"],0,512),
        LayerInfo("fc2","MatMul",["r1","W2"],["y2"],1_000_000,256),
    ]
    result = PartitionResult(["fpga","fpga","gpu"], 5.0, 1_000_000, 50_000_000, 1)
    return layers, result

def test_valid_json():
    data = json.loads(generate_manifest(*_data()))
    assert "layers" in data and "summary" in data

def test_layer_count():
    assert len(json.loads(generate_manifest(*_data()))["layers"]) == 3

def test_layer_fields():
    l = json.loads(generate_manifest(*_data()))["layers"][0]
    assert l["name"] == "fc1" and l["device"] == "fpga"

def test_transfers():
    t = json.loads(generate_manifest(*_data()))["transfers"]
    assert len(t) == 1 and t[0]["from_device"] == "fpga"
```

- [ ] **Step 2: Write implementation**

```python
# splitinfer/partitioner/manifest.py
import json
from partitioner.graph import LayerInfo
from partitioner.solver import PartitionResult

def generate_manifest(layers: list[LayerInfo], result: PartitionResult) -> str:
    layer_entries = [{"name": l.name, "op_type": l.op_type, "device": d,
                      "weight_bytes": l.weight_bytes, "output_tensor_bytes": l.output_tensor_bytes,
                      "inputs": l.input_names, "outputs": l.output_names}
                     for l, d in zip(layers, result.assignments)]

    transfers = [{"after_layer": layers[i-1].name, "before_layer": layers[i].name,
                  "from_device": result.assignments[i-1], "to_device": result.assignments[i],
                  "tensor_names": layers[i-1].output_names,
                  "tensor_bytes": layers[i-1].output_tensor_bytes}
                 for i in range(1, len(result.assignments))
                 if result.assignments[i] != result.assignments[i-1]]

    return json.dumps({"version": "1.0", "layers": layer_entries, "transfers": transfers,
        "summary": {"total_layers": len(layers),
                     "gpu_layers": sum(1 for a in result.assignments if a == "gpu"),
                     "fpga_layers": sum(1 for a in result.assignments if a == "fpga"),
                     "estimated_latency_ms": result.total_latency_ms,
                     "gpu_memory_bytes": result.gpu_memory_bytes,
                     "fpga_memory_bytes": result.fpga_memory_bytes,
                     "num_transfers": result.num_transfers}}, indent=2)
```

- [ ] **Step 3: Run tests and commit**

```bash
python -m pytest partitioner/tests/test_manifest.py -v
git add partitioner/manifest.py partitioner/tests/test_manifest.py
git commit -m "feat: JSON partition manifest generator"
```

---

## Phase 4: SplitInfer Runtime (Jetson)

### Task 14: Manifest Loader (C++)

See file structure for paths. Creates `manifest.h`, `manifest.cpp`, `test_manifest.cpp`, `runtime/CMakeLists.txt`. Parses JSON manifest into C++ structs. Tested with embedded JSON string.

### Task 15: Pipeline Orchestrator

Creates `gpu_executor.h` (abstract base), `fpga_executor.h` (abstract base), `pipeline.h`, `telemetry.h`, `pipeline.cpp`, `telemetry.cpp`, `test_pipeline_mock.cpp`. Uses mock executors for testing. Implements sequential layer execution with device-switch transfers and timing telemetry.

### Task 16: CLI Entry Point

Creates `splitinfer_run.cpp` — loads manifest, instantiates stub executors, runs one inference, prints telemetry. Used for smoke testing before real TensorRT/EdgeCoh executors are integrated.

(Full code for Tasks 14-16 follows the same TDD pattern as earlier tasks. Each step includes complete code, build commands, and commit instructions.)

---

## Phase 5: Evaluation Scripts

### Task 17: Model Download and Baseline Scripts

Creates `download_models.sh` (generates synthetic DLRM, exports MobileBERT and YOLOv8-nano via Python), `jetson_only_fp.py` (B1 baseline), `jetson_only_quant.py` (B2 baseline).

### Task 18: Experiment Runners

Creates `e1_capability_unlock.sh`, `e3_partition_sweep.py`, `e5_scalability.py`.

---

## Phase 6: Integration Testing

### Task 19: USB Loopback Test

Creates `test_loopback.c` — sends SYNC_BARRIER and NMC_EXEC to real FPGA, expects ACK. Requires Nexys 4 DDR connected and programmed.

### Task 20: End-to-End Smoke Test

Creates `smoke_test.sh` — generates tiny model, partitions it, runs SplitInfer runtime with stubs, runs all unit tests.

---

## Summary

| Phase | Tasks | Where | Key Deliverables |
|---|---|---|---|
| 1: Protocol | 1-4 | Any | libedgecoh: messages, state machine, USB transport |
| 2: FPGA | 5-9 | x86 workstation | Verilog: USB/UART, EdgeCoh FSM, embedding, MAC, DDR2 arbiter, top-level with dual clock domains (sys_clk 100MHz + MIG ui_clk 81.25MHz), MIG DDR2 IP + Clocking Wizard |
| 3: Partitioner | 10-13 | Any | Python: ONNX parser, cost model, DP solver, manifest |
| 4: Runtime | 14-16 | Jetson | C++: manifest loader, pipeline, CLI tool |
| 5: Evaluation | 17-18 | Jetson | Models, baselines, experiment scripts |
| 6: Integration | 19-20 | Jetson+FPGA | USB loopback, end-to-end smoke test |

**Phases 1-3 can proceed in parallel.** Phase 4 depends on Phase 1. Phase 5 depends on 3-4. Phase 6 depends on all.
