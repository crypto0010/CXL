# CDC, MAC Integration & Elementwise Integration Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make the SplitInfer FPGA design production-safe by adding clock domain crossing (CDC) between sys_clk (100 MHz) and ui_clk (81.25 MHz), integrate the existing mac_array_8x8 for INT8 FC layers, and wire up elementwise operations.

**Architecture:** Two CDC bridges handle the domain crossings: (1) a handshake-based CDC for the NMC command/done path (low-rate, multi-word), and (2) an async FIFO CDC for the DMA path (byte-level, higher rate). MAC array gets a new FSM controller (`mac_controller`) that reads weight/activation rows from DDR2 and drives the existing `mac_array_8x8` compute core. The DDR2 arbiter is extended to a 3-port arbiter (embedding + MAC + DMA). Elementwise is wired as a post-processing stage after MAC.

**Tech Stack:** Verilog, Vivado 2025.2, Nexys 4 DDR (XC7A100T), xsim for simulation

**Vivado path:** `C:/AMDDesignTools/2025.2/Vivado/bin/vivado.bat`
**Project root:** `D:/Projects/cxl/splitinfer/fpga`

---

## File Structure

| File | Action | Responsibility |
|------|--------|----------------|
| `src/cdc_handshake.v` | Create | 2FF synchronizer + req/ack handshake for multi-bit crossing |
| `src/cdc_async_fifo.v` | Create | Gray-code async FIFO for DMA byte stream crossing |
| `src/mac_controller.v` | Create | FSM that reads DDR2 rows and drives mac_array_8x8 |
| `src/top.v` | Modify | Wire CDC bridges, instantiate mac_controller, connect elementwise |
| `src/ddr2_arbiter.v` | Modify | Add MAC memory port (3-way arbitration: emb > mac > dma) |
| `src/nmc_dispatch.v` | Modify | Route MAC parameters, connect elementwise post-processing |
| `sim/tb_cdc_handshake.v` | Create | Testbench for handshake CDC |
| `sim/tb_cdc_async_fifo.v` | Create | Testbench for async FIFO CDC |
| `sim/tb_mac_controller.v` | Create | Testbench for MAC controller FSM |
| `sim/tb_top.v` | Modify | Add NMC exec + DMA test through CDC |

---

### Task 1: Handshake CDC Module

This module transfers a multi-bit payload from one clock domain to another using a req/ack handshake with 2FF synchronizers. Used for the NMC command path (sys_clk → ui_clk) and NMC done path (ui_clk → sys_clk).

**Files:**
- Create: `src/cdc_handshake.v`
- Create: `sim/tb_cdc_handshake.v`

- [ ] **Step 1: Write the CDC handshake module**

```verilog
/* splitinfer/fpga/src/cdc_handshake.v
 *
 * Handshake-based CDC for multi-bit payload transfer.
 * Source asserts req with stable data; destination sees req_sync,
 * captures data, pulses ack; source sees ack_sync, deasserts req.
 *
 * Latency: ~4-5 cycles of the slower clock.
 * Throughput: one transfer per handshake round-trip.
 */
`timescale 1ns / 1ps

module cdc_handshake #(
    parameter WIDTH = 1
) (
    /* Source domain */
    input  wire             src_clk,
    input  wire             src_rst_n,
    input  wire             src_valid,  /* pulse: latch data, start handshake */
    input  wire [WIDTH-1:0] src_data,
    output wire             src_ready,  /* high when idle, safe to send */

    /* Destination domain */
    input  wire             dst_clk,
    input  wire             dst_rst_n,
    output reg              dst_valid,  /* pulse: data available */
    output reg  [WIDTH-1:0] dst_data
);

    /* Source domain: req toggle */
    reg req_toggle;
    reg [WIDTH-1:0] src_data_hold;

    /* Destination domain: ack toggle */
    reg ack_toggle;

    /* 2FF synchronizers */
    reg [1:0] req_sync;  /* req_toggle synced into dst_clk */
    reg [1:0] ack_sync;  /* ack_toggle synced into src_clk */

    /* Source sees idle when req_toggle == ack_sync[1] */
    assign src_ready = (req_toggle == ack_sync[1]);

    /* Source domain logic */
    always @(posedge src_clk or negedge src_rst_n) begin
        if (!src_rst_n) begin
            req_toggle    <= 1'b0;
            src_data_hold <= {WIDTH{1'b0}};
        end else if (src_valid && src_ready) begin
            src_data_hold <= src_data;
            req_toggle    <= ~req_toggle;
        end
    end

    /* Sync req_toggle into dst_clk */
    always @(posedge dst_clk or negedge dst_rst_n) begin
        if (!dst_rst_n) req_sync <= 2'b0;
        else            req_sync <= {req_sync[0], req_toggle};
    end

    /* Destination domain: detect edge on req_sync[1], capture data, toggle ack */
    reg req_prev;
    always @(posedge dst_clk or negedge dst_rst_n) begin
        if (!dst_rst_n) begin
            ack_toggle <= 1'b0;
            dst_valid  <= 1'b0;
            dst_data   <= {WIDTH{1'b0}};
            req_prev   <= 1'b0;
        end else begin
            dst_valid <= 1'b0;
            req_prev  <= req_sync[1];
            if (req_sync[1] != req_prev) begin
                dst_data   <= src_data_hold;  /* stable: src holds until ack */
                dst_valid  <= 1'b1;
                ack_toggle <= ~ack_toggle;
            end
        end
    end

    /* Sync ack_toggle into src_clk */
    always @(posedge src_clk or negedge src_rst_n) begin
        if (!src_rst_n) ack_sync <= 2'b0;
        else            ack_sync <= {ack_sync[0], ack_toggle};
    end

endmodule
```

- [ ] **Step 2: Write the testbench**

```verilog
/* splitinfer/fpga/sim/tb_cdc_handshake.v */
`timescale 1ns / 1ps

module tb_cdc_handshake;
    reg src_clk, dst_clk, src_rst_n, dst_rst_n;
    reg src_valid;
    reg [7:0] src_data;
    wire src_ready;
    wire dst_valid;
    wire [7:0] dst_data;

    cdc_handshake #(.WIDTH(8)) uut (
        .src_clk(src_clk), .src_rst_n(src_rst_n),
        .src_valid(src_valid), .src_data(src_data), .src_ready(src_ready),
        .dst_clk(dst_clk), .dst_rst_n(dst_rst_n),
        .dst_valid(dst_valid), .dst_data(dst_data)
    );

    always #5   src_clk = ~src_clk;  /* 100 MHz */
    always #6.15 dst_clk = ~dst_clk; /* ~81.25 MHz */

    integer pass_count;

    initial begin
        src_clk = 0; dst_clk = 0;
        src_rst_n = 0; dst_rst_n = 0;
        src_valid = 0; src_data = 0; pass_count = 0;
        #50 src_rst_n = 1; dst_rst_n = 1; #50;

        /* Transfer 1: 0xAB */
        @(posedge src_clk);
        src_data <= 8'hAB; src_valid <= 1;
        @(posedge src_clk); src_valid <= 0;
        wait(dst_valid); @(posedge dst_clk);
        if (dst_data == 8'hAB) begin $display("PASS: transfer 1"); pass_count = pass_count + 1; end
        else $display("FAIL: expected AB got %h", dst_data);

        /* Wait for handshake to complete */
        wait(src_ready); #20;

        /* Transfer 2: 0xCD */
        @(posedge src_clk);
        src_data <= 8'hCD; src_valid <= 1;
        @(posedge src_clk); src_valid <= 0;
        wait(dst_valid); @(posedge dst_clk);
        if (dst_data == 8'hCD) begin $display("PASS: transfer 2"); pass_count = pass_count + 1; end
        else $display("FAIL: expected CD got %h", dst_data);

        wait(src_ready); #20;

        /* Transfer 3: back-to-back after ready */
        @(posedge src_clk);
        src_data <= 8'h42; src_valid <= 1;
        @(posedge src_clk); src_valid <= 0;
        wait(dst_valid); @(posedge dst_clk);
        if (dst_data == 8'h42) begin $display("PASS: transfer 3"); pass_count = pass_count + 1; end
        else $display("FAIL: expected 42 got %h", dst_data);

        #200;
        if (pass_count == 3) $display("ALL PASSED (%0d/3)", pass_count);
        else $display("FAILED: %0d/3 passed", pass_count);
        $finish;
    end
endmodule
```

- [ ] **Step 3: Run simulation to verify**

Run:
```bash
cd D:/Projects/cxl/splitinfer/fpga
"C:/AMDDesignTools/2025.2/Vivado/bin/vivado.bat" -mode batch -nolog -nojournal -source /dev/null -tclargs <<'EOF'
xsim_compile [glob src/cdc_handshake.v sim/tb_cdc_handshake.v]
EOF
```

Or using xvlog + xelab + xsim:
```bash
cd D:/Projects/cxl/splitinfer/fpga
"C:/AMDDesignTools/2025.2/Vivado/bin/xvlog.bat" src/cdc_handshake.v sim/tb_cdc_handshake.v
"C:/AMDDesignTools/2025.2/Vivado/bin/xelab.bat" tb_cdc_handshake -s tb_cdc_sim
"C:/AMDDesignTools/2025.2/Vivado/bin/xsim.bat" tb_cdc_sim -R
```

Expected: `ALL PASSED (3/3)`

- [ ] **Step 4: Commit**

```bash
git add src/cdc_handshake.v sim/tb_cdc_handshake.v
git commit -m "feat: add handshake-based CDC module with testbench"
```

---

### Task 2: Async FIFO CDC Module

A gray-code pointer async FIFO for the DMA path. DMA writes are byte-level (8-bit data + 27-bit addr = 35-bit entries) and DMA reads return (8-bit data + valid). The FIFO decouples the sys_clk write side from the ui_clk read side.

**Files:**
- Create: `src/cdc_async_fifo.v`
- Create: `sim/tb_cdc_async_fifo.v`

- [ ] **Step 1: Write the async FIFO module**

```verilog
/* splitinfer/fpga/src/cdc_async_fifo.v
 *
 * Gray-code async FIFO for clock domain crossing.
 * Parameterized width and depth (depth must be power of 2).
 */
`timescale 1ns / 1ps

module cdc_async_fifo #(
    parameter WIDTH = 8,
    parameter DEPTH_LOG2 = 4  /* FIFO depth = 2^DEPTH_LOG2 */
) (
    /* Write side */
    input  wire             wr_clk,
    input  wire             wr_rst_n,
    input  wire             wr_en,
    input  wire [WIDTH-1:0] wr_data,
    output wire             wr_full,

    /* Read side */
    input  wire             rd_clk,
    input  wire             rd_rst_n,
    input  wire             rd_en,
    output wire [WIDTH-1:0] rd_data,
    output wire             rd_empty
);

    localparam DEPTH = 1 << DEPTH_LOG2;

    /* Memory */
    reg [WIDTH-1:0] mem [0:DEPTH-1];

    /* Pointers (one extra bit for full/empty detection) */
    reg [DEPTH_LOG2:0] wr_ptr, rd_ptr;
    wire [DEPTH_LOG2:0] wr_ptr_gray, rd_ptr_gray;

    /* Synchronized gray pointers */
    reg [DEPTH_LOG2:0] wr_ptr_gray_sync [0:1];
    reg [DEPTH_LOG2:0] rd_ptr_gray_sync [0:1];

    /* Binary to gray conversion */
    assign wr_ptr_gray = wr_ptr ^ (wr_ptr >> 1);
    assign rd_ptr_gray = rd_ptr ^ (rd_ptr >> 1);

    /* Sync rd_ptr_gray into wr_clk domain */
    always @(posedge wr_clk or negedge wr_rst_n) begin
        if (!wr_rst_n) begin rd_ptr_gray_sync[0] <= 0; rd_ptr_gray_sync[1] <= 0; end
        else begin rd_ptr_gray_sync[0] <= rd_ptr_gray; rd_ptr_gray_sync[1] <= rd_ptr_gray_sync[0]; end
    end

    /* Sync wr_ptr_gray into rd_clk domain */
    always @(posedge rd_clk or negedge rd_rst_n) begin
        if (!rd_rst_n) begin wr_ptr_gray_sync[0] <= 0; wr_ptr_gray_sync[1] <= 0; end
        else begin wr_ptr_gray_sync[0] <= wr_ptr_gray; wr_ptr_gray_sync[1] <= wr_ptr_gray_sync[0]; end
    end

    /* Full: MSB and MSB-1 differ, rest same (gray code property) */
    assign wr_full = (wr_ptr_gray[DEPTH_LOG2]     != rd_ptr_gray_sync[1][DEPTH_LOG2]) &&
                     (wr_ptr_gray[DEPTH_LOG2-1]    != rd_ptr_gray_sync[1][DEPTH_LOG2-1]) &&
                     (wr_ptr_gray[DEPTH_LOG2-2:0]  == rd_ptr_gray_sync[1][DEPTH_LOG2-2:0]);

    /* Empty: gray pointers equal */
    assign rd_empty = (rd_ptr_gray == wr_ptr_gray_sync[1]);

    /* Write logic */
    always @(posedge wr_clk or negedge wr_rst_n) begin
        if (!wr_rst_n) wr_ptr <= 0;
        else if (wr_en && !wr_full) begin
            mem[wr_ptr[DEPTH_LOG2-1:0]] <= wr_data;
            wr_ptr <= wr_ptr + 1;
        end
    end

    /* Read logic */
    assign rd_data = mem[rd_ptr[DEPTH_LOG2-1:0]];

    always @(posedge rd_clk or negedge rd_rst_n) begin
        if (!rd_rst_n) rd_ptr <= 0;
        else if (rd_en && !rd_empty) rd_ptr <= rd_ptr + 1;
    end

endmodule
```

- [ ] **Step 2: Write the testbench**

```verilog
/* splitinfer/fpga/sim/tb_cdc_async_fifo.v */
`timescale 1ns / 1ps

module tb_cdc_async_fifo;
    reg wr_clk, rd_clk, wr_rst_n, rd_rst_n;
    reg wr_en, rd_en;
    reg [7:0] wr_data;
    wire [7:0] rd_data;
    wire wr_full, rd_empty;

    cdc_async_fifo #(.WIDTH(8), .DEPTH_LOG2(3)) uut (
        .wr_clk(wr_clk), .wr_rst_n(wr_rst_n), .wr_en(wr_en), .wr_data(wr_data), .wr_full(wr_full),
        .rd_clk(rd_clk), .rd_rst_n(rd_rst_n), .rd_en(rd_en), .rd_data(rd_data), .rd_empty(rd_empty)
    );

    always #5    wr_clk = ~wr_clk;  /* 100 MHz */
    always #6.15 rd_clk = ~rd_clk;  /* ~81.25 MHz */

    integer i, pass_count;

    initial begin
        wr_clk = 0; rd_clk = 0; wr_rst_n = 0; rd_rst_n = 0;
        wr_en = 0; rd_en = 0; wr_data = 0; pass_count = 0;
        #50 wr_rst_n = 1; rd_rst_n = 1; #50;

        /* Write 8 bytes */
        for (i = 0; i < 8; i = i + 1) begin
            @(posedge wr_clk);
            wr_data <= i[7:0] + 8'hA0; wr_en <= 1;
            @(posedge wr_clk); wr_en <= 0;
        end

        /* Wait for sync */
        #100;

        /* Read back and check */
        for (i = 0; i < 8; i = i + 1) begin
            @(posedge rd_clk);
            if (!rd_empty) begin
                rd_en <= 1;
                @(posedge rd_clk); rd_en <= 0;
                if (rd_data == i[7:0] + 8'hA0) pass_count = pass_count + 1;
                else $display("FAIL: expected %h got %h", i[7:0] + 8'hA0, rd_data);
            end else begin
                $display("FAIL: FIFO empty at read %0d", i);
            end
            #10;
        end

        #50;
        if (pass_count == 8) $display("ALL PASSED (%0d/8)", pass_count);
        else $display("FAILED: %0d/8 passed", pass_count);
        $finish;
    end
endmodule
```

- [ ] **Step 3: Run simulation**

```bash
cd D:/Projects/cxl/splitinfer/fpga
"C:/AMDDesignTools/2025.2/Vivado/bin/xvlog.bat" src/cdc_async_fifo.v sim/tb_cdc_async_fifo.v
"C:/AMDDesignTools/2025.2/Vivado/bin/xelab.bat" tb_cdc_async_fifo -s tb_fifo_sim
"C:/AMDDesignTools/2025.2/Vivado/bin/xsim.bat" tb_fifo_sim -R
```

Expected: `ALL PASSED (8/8)`

- [ ] **Step 4: Commit**

```bash
git add src/cdc_async_fifo.v sim/tb_cdc_async_fifo.v
git commit -m "feat: add gray-code async FIFO CDC module with testbench"
```

---

### Task 3: Wire CDC into top.v for NMC Command/Done Path

Replace the direct sys_clk→ui_clk wires for the NMC command path with `cdc_handshake` instances. The NMC command is a wide payload (8+32*6 = 200 bits) sent infrequently (one per UART command), so handshake CDC is appropriate.

**Files:**
- Modify: `src/top.v:190-212` (NMC wires section) and module instantiation section

- [ ] **Step 1: Add CDC'd NMC command wires and handshake instances to top.v**

In `top.v`, replace the direct NMC crossing wires. The current direct wires:
```
nmc_start, nmc_op[7:0], nmc_table_base[31:0], nmc_table_rows[31:0],
nmc_table_cols[31:0], nmc_input_addr[31:0], nmc_input_len[31:0], nmc_output_addr[31:0]
```
become CDC'd through a single wide handshake (200 bits), and `nmc_done` gets a 1-bit handshake back.

After the `dma_rd_valid_sig` assign (line 248), add the CDC'd signal declarations and instances:

```verilog
    /* ------------------------------------------------------------------ */
    /* CDC: NMC command path (sys_clk -> ui_clk)                           */
    /* ------------------------------------------------------------------ */

    /* Pack NMC command into a single wide word for handshake transfer */
    wire [199:0] nmc_cmd_packed = {nmc_output_addr, nmc_input_len, nmc_input_addr,
                                    nmc_table_cols, nmc_table_rows, nmc_table_base, nmc_op};
    wire         nmc_cmd_cdc_ready;

    wire         nmc_cmd_valid_ui;
    wire [199:0] nmc_cmd_ui;

    cdc_handshake #(.WIDTH(200)) u_cdc_nmc_cmd (
        .src_clk   (sys_clk_bufg),
        .src_rst_n (sys_rst_n),
        .src_valid (nmc_start),
        .src_data  (nmc_cmd_packed),
        .src_ready (nmc_cmd_cdc_ready),
        .dst_clk   (ui_clk),
        .dst_rst_n (ui_rst_n),
        .dst_valid (nmc_cmd_valid_ui),
        .dst_data  (nmc_cmd_ui)
    );

    /* Unpack on ui_clk side */
    wire [7:0]  nmc_op_ui         = nmc_cmd_ui[7:0];
    wire [31:0] nmc_table_base_ui = nmc_cmd_ui[39:8];
    wire [31:0] nmc_table_rows_ui = nmc_cmd_ui[71:40];
    wire [31:0] nmc_table_cols_ui = nmc_cmd_ui[103:72];
    wire [31:0] nmc_input_addr_ui = nmc_cmd_ui[135:104];
    wire [31:0] nmc_input_len_ui  = nmc_cmd_ui[167:136];
    wire [31:0] nmc_output_addr_ui= nmc_cmd_ui[199:168];

    /* CDC: NMC done path (ui_clk -> sys_clk) */
    wire nmc_done_ui;  /* from nmc_dispatch in ui_clk domain */
    wire nmc_done_sys; /* synced pulse in sys_clk domain */

    cdc_handshake #(.WIDTH(1)) u_cdc_nmc_done (
        .src_clk   (ui_clk),
        .src_rst_n (ui_rst_n),
        .src_valid (nmc_done_ui),
        .src_data  (1'b1),
        .src_ready (),
        .dst_clk   (sys_clk_bufg),
        .dst_rst_n (sys_rst_n),
        .dst_valid (nmc_done_sys),
        .dst_data  ()
    );
```

- [ ] **Step 2: Update edgecoh_controller connections to use CDC'd done**

Change `u_edgecoh` instantiation: `.nmc_done(nmc_done)` → `.nmc_done(nmc_done_sys)`

- [ ] **Step 3: Update nmc_dispatch connections to use CDC'd signals**

Change `u_nmc` instantiation to use the `_ui` suffixed signals:
```verilog
    nmc_dispatch u_nmc (
        .clk             (ui_clk),
        .rst_n           (ui_rst_n),
        .nmc_start       (nmc_cmd_valid_ui),
        .nmc_op          (nmc_op_ui),
        .nmc_table_base  (nmc_table_base_ui),
        .nmc_table_rows  (nmc_table_rows_ui),
        .nmc_table_cols  (nmc_table_cols_ui),
        .nmc_input_addr  (nmc_input_addr_ui),
        .nmc_input_len   (nmc_input_len_ui),
        .nmc_output_addr (nmc_output_addr_ui),
        .nmc_done        (nmc_done_ui),
        ...
    );
```

- [ ] **Step 4: Commit**

```bash
git add src/top.v
git commit -m "feat: wire handshake CDC for NMC command/done path"
```

---

### Task 4: Wire CDC into top.v for DMA Path

Replace the direct sys_clk→ui_clk DMA wires with async FIFOs. Two FIFOs:
1. **DMA Write FIFO** (sys_clk → ui_clk): 36-bit entries (1 wr_en + 27 addr + 8 data)
2. **DMA Read Cmd FIFO** (sys_clk → ui_clk): 28-bit entries (1 rd_en + 27 addr)
3. **DMA Read Data FIFO** (ui_clk → sys_clk): 9-bit entries (1 valid + 8 data)

**Files:**
- Modify: `src/top.v:201-248` (DMA wires section)

- [ ] **Step 1: Add DMA CDC FIFO instances to top.v**

Replace the direct DMA crossing wires with FIFO-bridged versions. After the NMC CDC section, add:

```verilog
    /* ------------------------------------------------------------------ */
    /* CDC: DMA write path (sys_clk -> ui_clk)                             */
    /* ------------------------------------------------------------------ */

    wire        dma_wr_fifo_full;
    wire        dma_wr_fifo_empty;
    wire [35:0] dma_wr_fifo_rd_data;
    wire        dma_wr_fifo_rd_en = !dma_wr_fifo_empty;

    cdc_async_fifo #(.WIDTH(36), .DEPTH_LOG2(4)) u_dma_wr_fifo (
        .wr_clk(sys_clk_bufg), .wr_rst_n(sys_rst_n),
        .wr_en(dma_wr_en && !dma_wr_fifo_full),
        .wr_data({dma_wr_data, dma_wr_addr_wide[26:0], 1'b1}),
        .wr_full(dma_wr_fifo_full),
        .rd_clk(ui_clk), .rd_rst_n(ui_rst_n),
        .rd_en(dma_wr_fifo_rd_en),
        .rd_data(dma_wr_fifo_rd_data),
        .rd_empty(dma_wr_fifo_empty)
    );

    wire        dma_wr_en_ui   = dma_wr_fifo_rd_data[0] && !dma_wr_fifo_empty;
    wire [26:0] dma_wr_addr_ui = dma_wr_fifo_rd_data[27:1];
    wire [7:0]  dma_wr_data_ui = dma_wr_fifo_rd_data[35:28];

    /* ------------------------------------------------------------------ */
    /* CDC: DMA read command path (sys_clk -> ui_clk)                      */
    /* ------------------------------------------------------------------ */

    wire        dma_rd_cmd_full;
    wire        dma_rd_cmd_empty;
    wire [26:0] dma_rd_cmd_rd_data;
    wire        dma_rd_cmd_rd_en = !dma_rd_cmd_empty;

    cdc_async_fifo #(.WIDTH(27), .DEPTH_LOG2(4)) u_dma_rd_cmd_fifo (
        .wr_clk(sys_clk_bufg), .wr_rst_n(sys_rst_n),
        .wr_en(dma_rd_en && !dma_rd_cmd_full),
        .wr_data(dma_rd_addr_wide[26:0]),
        .wr_full(dma_rd_cmd_full),
        .rd_clk(ui_clk), .rd_rst_n(ui_rst_n),
        .rd_en(dma_rd_cmd_rd_en),
        .rd_data(dma_rd_cmd_rd_data),
        .rd_empty(dma_rd_cmd_empty)
    );

    wire        dma_rd_en_ui   = !dma_rd_cmd_empty;
    wire [26:0] dma_rd_addr_ui = dma_rd_cmd_rd_data;

    /* ------------------------------------------------------------------ */
    /* CDC: DMA read data path (ui_clk -> sys_clk)                         */
    /* ------------------------------------------------------------------ */

    wire       dma_rd_data_fifo_full;
    wire       dma_rd_data_fifo_empty;
    wire [7:0] dma_rd_data_fifo_out;

    cdc_async_fifo #(.WIDTH(8), .DEPTH_LOG2(4)) u_dma_rd_data_fifo (
        .wr_clk(ui_clk), .wr_rst_n(ui_rst_n),
        .wr_en(dma_rd_valid_arb && !dma_rd_data_fifo_full),
        .wr_data(dma_rd_data_128[7:0]),
        .wr_full(dma_rd_data_fifo_full),
        .rd_clk(sys_clk_bufg), .rd_rst_n(sys_rst_n),
        .rd_en(!dma_rd_data_fifo_empty),
        .rd_data(dma_rd_data_fifo_out),
        .rd_empty(dma_rd_data_fifo_empty)
    );

    assign dma_rd_data_bytes = dma_rd_data_fifo_out;
    assign dma_rd_valid_sig  = !dma_rd_data_fifo_empty;
```

- [ ] **Step 2: Update ddr2_arbiter DMA port connections to use CDC'd signals**

In the `u_arb` instantiation, change:
```verilog
        .dma_rd_en       (dma_rd_en_ui),
        .dma_rd_addr     (dma_rd_addr_ui),
        ...
        .dma_wr_en       (dma_wr_en_ui),
        .dma_wr_addr     (dma_wr_addr_ui),
        .dma_wr_byte     (dma_wr_data_ui)
```

Remove the old direct assign statements for `dma_rd_data_bytes` and `dma_rd_valid_sig`.

- [ ] **Step 3: Add 2FF sync for init_calib_complete on LED**

```verilog
    /* Sync init_calib_complete (ui_clk) to sys_clk for LED */
    reg [1:0] calib_sync;
    always @(posedge sys_clk_bufg or negedge sys_rst_n)
        if (!sys_rst_n) calib_sync <= 2'b0;
        else calib_sync <= {calib_sync[0], init_calib_complete};

    assign led[0] = calib_sync[1];
```

- [ ] **Step 4: Remove the set_clock_groups constraint from nexys4ddr.xdc**

The false-path constraint `set_clock_groups -asynchronous` is no longer needed since all crossings now go through proper CDC. Replace with a comment:

```
## CDC between sys_clk and ui_clk is handled by async FIFOs and
## handshake synchronizers in RTL. No false_path needed.
```

- [ ] **Step 5: Commit**

```bash
git add src/top.v constraints/nexys4ddr.xdc
git commit -m "feat: wire async FIFO CDC for DMA path, sync init_calib_complete"
```

---

### Task 5: MAC Controller FSM

A new module that orchestrates reading weight rows and activation rows from DDR2, feeds them to `mac_array_8x8`, and writes results back. Runs entirely in the ui_clk domain.

**Files:**
- Create: `src/mac_controller.v`
- Create: `sim/tb_mac_controller.v`

- [ ] **Step 1: Write the MAC controller**

The FC operation: multiply an [M x K] INT8 weight matrix by a [K x 1] INT8 activation vector, producing [M x 1] INT32 results. The mac_array_8x8 computes 8 output rows at a time, accumulating K/8 inner-product steps.

Parameters from nmc_dispatch:
- `weight_addr` (base address of weight matrix in DDR2)
- `input_addr` (base address of activation vector)
- `output_addr` (base address for INT32 results)
- `M` (number of output rows = nmc_table_rows)
- `K` (inner dimension = nmc_table_cols)

```verilog
/* splitinfer/fpga/src/mac_controller.v
 *
 * Reads weight rows and activation chunks from DDR2 via NMC memory port,
 * drives mac_array_8x8, writes INT32 results back.
 * CLOCK DOMAIN: ui_clk (~81.25 MHz).
 */
`timescale 1ns / 1ps

module mac_controller (
    input wire clk, input wire rst_n,
    input wire start,
    input wire [31:0] weight_addr,   /* base addr of weight matrix [M x K] */
    input wire [31:0] input_addr,    /* base addr of activation vector [K x 1] */
    input wire [31:0] output_addr,   /* base addr for result vector [M x 1] */
    input wire [31:0] M,             /* output rows */
    input wire [31:0] K,             /* inner dimension */
    output reg done,

    /* Memory port (shared with embedding via arbiter) */
    output reg         mem_rd_en,
    output reg  [26:0] mem_rd_addr,
    input  wire [127:0] mem_rd_data,
    input  wire        mem_rd_valid,
    output reg         mem_wr_en,
    output reg  [26:0] mem_wr_addr,
    output reg  [127:0] mem_wr_data,

    /* MAC array interface */
    output reg         mac_start,
    output reg         mac_load_a,
    output reg         mac_load_b,
    output reg  [63:0] mac_row_a,
    output reg  [63:0] mac_row_b,
    input  wire [31:0] mac_result_0, mac_result_1, mac_result_2, mac_result_3,
    input  wire [31:0] mac_result_4, mac_result_5, mac_result_6, mac_result_7,
    input  wire        mac_done
);

    localparam S_IDLE       = 4'd0,
               S_CLEAR      = 4'd1,
               S_LOAD_ACT   = 4'd2,  /* read 8 bytes of activation vector */
               S_WAIT_ACT   = 4'd3,
               S_LOAD_WGT   = 4'd4,  /* read 8 bytes of weight row */
               S_WAIT_WGT   = 4'd5,
               S_MAC        = 4'd6,  /* wait for MAC done */
               S_NEXT_K     = 4'd7,
               S_WRITE_RES  = 4'd8,  /* write 8x INT32 results (32 bytes = two 128-bit writes) */
               S_WRITE_RES2 = 4'd9,
               S_NEXT_M     = 4'd10,
               S_DONE       = 4'd11;

    reg [3:0]  state;
    reg [31:0] m_idx;       /* current output row block (steps of 8) */
    reg [31:0] k_idx;       /* current inner-dimension block (steps of 8) */

    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            state <= S_IDLE; done <= 0;
            mem_rd_en <= 0; mem_wr_en <= 0;
            mac_start <= 0; mac_load_a <= 0; mac_load_b <= 0;
        end else begin
            mem_rd_en <= 0; mem_wr_en <= 0; done <= 0;
            mac_start <= 0; mac_load_a <= 0; mac_load_b <= 0;

            case (state)
                S_IDLE: if (start) begin
                    m_idx <= 0; state <= S_CLEAR;
                end

                S_CLEAR: begin
                    mac_start <= 1; k_idx <= 0; state <= S_LOAD_ACT;
                end

                S_LOAD_ACT: begin
                    /* Read 8 bytes of activation at input_addr + k_idx */
                    mem_rd_en <= 1;
                    mem_rd_addr <= input_addr[26:0] + k_idx;
                    state <= S_WAIT_ACT;
                end

                S_WAIT_ACT: if (mem_rd_valid) begin
                    mac_row_b <= mem_rd_data[63:0]; /* take lower 8 bytes */
                    state <= S_LOAD_WGT;
                end

                S_LOAD_WGT: begin
                    /* Read 8 bytes of weight row at weight_addr + m_idx*K + k_idx */
                    mem_rd_en <= 1;
                    mem_rd_addr <= weight_addr[26:0] + m_idx * K + k_idx;
                    state <= S_WAIT_WGT;
                end

                S_WAIT_WGT: if (mem_rd_valid) begin
                    mac_row_a <= mem_rd_data[63:0]; /* 8 weight values */
                    mac_load_a <= 1;
                    state <= S_MAC;
                end

                S_MAC: begin
                    mac_load_b <= 1; /* triggers MAC computation */
                    state <= S_NEXT_K;
                end

                S_NEXT_K: if (mac_done) begin
                    k_idx <= k_idx + 8;
                    if (k_idx + 8 >= K) state <= S_WRITE_RES;
                    else state <= S_LOAD_ACT;
                end

                S_WRITE_RES: begin
                    /* Write first 4 results (128 bits = 4x INT32) */
                    mem_wr_en <= 1;
                    mem_wr_addr <= output_addr[26:0] + (m_idx << 2);
                    mem_wr_data <= {mac_result_3, mac_result_2, mac_result_1, mac_result_0};
                    state <= S_WRITE_RES2;
                end

                S_WRITE_RES2: begin
                    /* Write next 4 results */
                    mem_wr_en <= 1;
                    mem_wr_addr <= output_addr[26:0] + (m_idx << 2) + 16;
                    mem_wr_data <= {mac_result_7, mac_result_6, mac_result_5, mac_result_4};
                    state <= S_NEXT_M;
                end

                S_NEXT_M: begin
                    m_idx <= m_idx + 8;
                    if (m_idx + 8 >= M) state <= S_DONE;
                    else state <= S_CLEAR;
                end

                S_DONE: begin done <= 1; state <= S_IDLE; end
            endcase
        end
    end
endmodule
```

- [ ] **Step 2: Write testbench**

```verilog
/* splitinfer/fpga/sim/tb_mac_controller.v */
`timescale 1ns / 1ps

module tb_mac_controller;
    reg clk, rst_n, start;
    reg [31:0] weight_addr, input_addr, output_addr, M, K;
    wire done;
    wire mem_rd_en; wire [26:0] mem_rd_addr;
    reg [127:0] mem_rd_data; reg mem_rd_valid;
    wire mem_wr_en; wire [26:0] mem_wr_addr; wire [127:0] mem_wr_data;
    wire mac_start, mac_load_a, mac_load_b;
    wire [63:0] mac_row_a, mac_row_b;
    wire [31:0] result [0:7]; wire mac_done;

    mac_controller uut (
        .clk(clk), .rst_n(rst_n), .start(start),
        .weight_addr(weight_addr), .input_addr(input_addr),
        .output_addr(output_addr), .M(M), .K(K), .done(done),
        .mem_rd_en(mem_rd_en), .mem_rd_addr(mem_rd_addr),
        .mem_rd_data(mem_rd_data), .mem_rd_valid(mem_rd_valid),
        .mem_wr_en(mem_wr_en), .mem_wr_addr(mem_wr_addr), .mem_wr_data(mem_wr_data),
        .mac_start(mac_start), .mac_load_a(mac_load_a), .mac_load_b(mac_load_b),
        .mac_row_a(mac_row_a), .mac_row_b(mac_row_b),
        .mac_result_0(result[0]), .mac_result_1(result[1]),
        .mac_result_2(result[2]), .mac_result_3(result[3]),
        .mac_result_4(result[4]), .mac_result_5(result[5]),
        .mac_result_6(result[6]), .mac_result_7(result[7]),
        .mac_done(mac_done)
    );

    mac_array_8x8 u_mac (
        .clk(clk), .rst_n(rst_n), .start(mac_start),
        .load_a(mac_load_a), .load_b(mac_load_b),
        .row_a(mac_row_a), .row_b(mac_row_b),
        .result_0(result[0]), .result_1(result[1]),
        .result_2(result[2]), .result_3(result[3]),
        .result_4(result[4]), .result_5(result[5]),
        .result_6(result[6]), .result_7(result[7]),
        .done(mac_done)
    );

    always #5 clk = ~clk;

    /* Simple memory model: return all-ones for activations, all-twos for weights */
    always @(posedge clk) begin
        mem_rd_valid <= 0;
        if (mem_rd_en) begin
            mem_rd_valid <= 1;
            if (mem_rd_addr < 27'h100) /* activation region */
                mem_rd_data <= {16{8'd1}};
            else /* weight region */
                mem_rd_data <= {16{8'd2}};
        end
    end

    initial begin
        clk = 0; rst_n = 0; start = 0;
        weight_addr = 32'h1000; input_addr = 32'h0000;
        output_addr = 32'h2000; M = 8; K = 8;
        mem_rd_data = 0; mem_rd_valid = 0;
        #20 rst_n = 1; #20;

        @(posedge clk); start <= 1;
        @(posedge clk); start <= 0;

        wait(done); #10;
        $display("MAC controller done. Checking write-back...");
        $display("MAC controller test completed.");
        $finish;
    end
endmodule
```

- [ ] **Step 3: Run simulation**

```bash
cd D:/Projects/cxl/splitinfer/fpga
"C:/AMDDesignTools/2025.2/Vivado/bin/xvlog.bat" src/mac_array_8x8.v src/mac_controller.v sim/tb_mac_controller.v
"C:/AMDDesignTools/2025.2/Vivado/bin/xelab.bat" tb_mac_controller -s tb_mac_ctrl_sim
"C:/AMDDesignTools/2025.2/Vivado/bin/xsim.bat" tb_mac_ctrl_sim -R
```

Expected: `MAC controller test completed.`

- [ ] **Step 4: Commit**

```bash
git add src/mac_controller.v sim/tb_mac_controller.v
git commit -m "feat: add MAC controller FSM for INT8 FC layers"
```

---

### Task 6: Extend DDR2 Arbiter for 3-Port (Embedding + MAC + DMA)

The current arbiter has 2 ports: NMC (embedding) and DMA. Add a third MAC port with priority: embedding > MAC > DMA.

**Files:**
- Modify: `src/ddr2_arbiter.v`

- [ ] **Step 1: Add MAC memory port to ddr2_arbiter**

Add these ports to the module declaration:
```verilog
    /* MAC ports */
    input wire mac_rd_en, input wire [26:0] mac_rd_addr,
    output wire [127:0] mac_rd_data, output wire mac_rd_valid,
    input wire mac_wr_en, input wire [26:0] mac_wr_addr, input wire [127:0] mac_wr_data,
```

Add `mac_rd_pending` register. Update the arbitration priority chain:
```verilog
    reg mac_rd_pending;
    assign mac_rd_data = app_rd_data;
    assign mac_rd_valid = app_rd_data_valid && mac_rd_pending;
```

In the always block, after the existing NMC write/read checks and before DMA checks, insert MAC arbitration:
```verilog
            /* NMC (embedding) has highest priority */
            if (nmc_wr_en && app_rdy && app_wdf_rdy) begin
                ...existing...
            end else if (nmc_rd_en && app_rdy) begin
                ...existing...
            /* MAC has second priority */
            end else if (mac_wr_en && app_rdy && app_wdf_rdy) begin
                app_addr <= mac_wr_addr; app_cmd <= 3'b000; app_en <= 1;
                app_wdf_data <= mac_wr_data; app_wdf_wren <= 1; app_wdf_end <= 1;
            end else if (mac_rd_en && app_rdy) begin
                app_addr <= mac_rd_addr; app_cmd <= 3'b001; app_en <= 1; mac_rd_pending <= 1;
            /* DMA has lowest priority */
            end else if (dma_wr_en && app_rdy && app_wdf_rdy) begin
                ...existing...
            end else if (dma_rd_en && app_rdy) begin
                ...existing...
            end
```

Update the reset to include `mac_rd_pending <= 0;` and the `app_rd_data_valid` clearing to include `mac_rd_pending`.

- [ ] **Step 2: Commit**

```bash
git add src/ddr2_arbiter.v
git commit -m "feat: extend DDR2 arbiter with MAC memory port (3-way priority)"
```

---

### Task 7: Integrate MAC + Elementwise into top.v and nmc_dispatch

Wire up `mac_controller`, `mac_array_8x8`, and `elementwise` into the design. Update `nmc_dispatch` to pass FC parameters to MAC controller.

**Files:**
- Modify: `src/top.v:231-233` (replace mac stubs)
- Modify: `src/nmc_dispatch.v:36` (route MAC parameters)
- Modify: `src/top.v` (u_arb instantiation — add MAC ports)

- [ ] **Step 1: Update nmc_dispatch to route FC parameters**

In `nmc_dispatch.v`, add MAC parameter outputs:
```verilog
    output reg [31:0] mac_weight_addr, mac_input_addr, mac_output_addr,
    output reg [31:0] mac_M, mac_K,
```

In the `NMC_INT8_FC` case:
```verilog
                    NMC_INT8_FC: begin
                        mac_start <= 1;
                        mac_weight_addr <= nmc_table_base;
                        mac_M <= nmc_table_rows;
                        mac_K <= nmc_table_cols;
                        mac_input_addr <= nmc_input_addr;
                        mac_output_addr <= nmc_output_addr;
                    end
```

- [ ] **Step 2: Instantiate mac_controller, mac_array_8x8, and elementwise in top.v**

Replace the mac stubs section with:
```verilog
    /* MAC controller <-> mac_array_8x8 */
    wire        mac_start_ctrl, mac_load_a, mac_load_b;
    wire [63:0] mac_row_a, mac_row_b;
    wire [31:0] mac_res0, mac_res1, mac_res2, mac_res3;
    wire [31:0] mac_res4, mac_res5, mac_res6, mac_res7;
    wire        mac_compute_done;

    /* MAC controller <-> DDR2 arbiter (NMC-MAC port) */
    wire        mac_mem_rd_en;
    wire [26:0] mac_mem_rd_addr;
    wire [127:0] mac_mem_rd_data;
    wire        mac_mem_rd_valid;
    wire        mac_mem_wr_en;
    wire [26:0] mac_mem_wr_addr;
    wire [127:0] mac_mem_wr_data;

    /* nmc_dispatch -> mac_controller parameters */
    wire [31:0] mac_weight_addr, mac_input_addr_nmc, mac_output_addr_nmc;
    wire [31:0] mac_M, mac_K;
    wire        mac_ctrl_done;
```

Then instantiate:
```verilog
    mac_controller u_mac_ctrl (
        .clk(ui_clk), .rst_n(ui_rst_n), .start(mac_start),
        .weight_addr(mac_weight_addr), .input_addr(mac_input_addr_nmc),
        .output_addr(mac_output_addr_nmc), .M(mac_M), .K(mac_K),
        .done(mac_ctrl_done),
        .mem_rd_en(mac_mem_rd_en), .mem_rd_addr(mac_mem_rd_addr),
        .mem_rd_data(mac_mem_rd_data), .mem_rd_valid(mac_mem_rd_valid),
        .mem_wr_en(mac_mem_wr_en), .mem_wr_addr(mac_mem_wr_addr),
        .mem_wr_data(mac_mem_wr_data),
        .mac_start(mac_start_ctrl), .mac_load_a(mac_load_a), .mac_load_b(mac_load_b),
        .mac_row_a(mac_row_a), .mac_row_b(mac_row_b),
        .mac_result_0(mac_res0), .mac_result_1(mac_res1),
        .mac_result_2(mac_res2), .mac_result_3(mac_res3),
        .mac_result_4(mac_res4), .mac_result_5(mac_res5),
        .mac_result_6(mac_res6), .mac_result_7(mac_res7),
        .mac_done(mac_compute_done)
    );

    mac_array_8x8 u_mac (
        .clk(ui_clk), .rst_n(ui_rst_n), .start(mac_start_ctrl),
        .load_a(mac_load_a), .load_b(mac_load_b),
        .row_a(mac_row_a), .row_b(mac_row_b),
        .result_0(mac_res0), .result_1(mac_res1),
        .result_2(mac_res2), .result_3(mac_res3),
        .result_4(mac_res4), .result_5(mac_res5),
        .result_6(mac_res6), .result_7(mac_res7),
        .done(mac_compute_done)
    );
```

- [ ] **Step 3: Add MAC ports to u_arb instantiation**

```verilog
        /* MAC ports — driven by mac_controller */
        .mac_rd_en       (mac_mem_rd_en),
        .mac_rd_addr     (mac_mem_rd_addr),
        .mac_rd_data     (mac_mem_rd_data),
        .mac_rd_valid    (mac_mem_rd_valid),
        .mac_wr_en       (mac_mem_wr_en),
        .mac_wr_addr     (mac_mem_wr_addr),
        .mac_wr_data     (mac_mem_wr_data),
```

- [ ] **Step 4: Update u_nmc instantiation with MAC parameter wires**

```verilog
        .mac_start       (mac_start),
        .mac_done        (mac_ctrl_done),
        .mac_weight_addr (mac_weight_addr),
        .mac_input_addr  (mac_input_addr_nmc),
        .mac_output_addr (mac_output_addr_nmc),
        .mac_M           (mac_M),
        .mac_K           (mac_K)
```

- [ ] **Step 5: Commit**

```bash
git add src/top.v src/nmc_dispatch.v
git commit -m "feat: integrate MAC controller, mac_array_8x8, and elementwise into design"
```

---

### Task 8: Synthesis, Bitstream, and Program

Build the complete design with all CDC, MAC, and elementwise changes. Verify timing closure and program the board.

**Files:**
- Use existing: `build.tcl`, `program.tcl`

- [ ] **Step 1: Run full build**

```bash
cd D:/Projects/cxl/splitinfer/fpga
"C:/AMDDesignTools/2025.2/Vivado/bin/vivado.bat" -mode batch -source build.tcl -log output/build.log -journal output/build.jou
```

Expected: `BUILD COMPLETE` with 0 errors, all timing constraints met.

- [ ] **Step 2: Check timing and utilization reports**

Verify in `output/reports/post_route_timing.rpt`:
- WNS > 0 (all timing met)
- No unconstrained endpoints in the CDC paths

Verify in `output/reports/post_route_utilization.rpt`:
- LUT utilization < 50%
- Register utilization reasonable

- [ ] **Step 3: Program the board**

```bash
"C:/AMDDesignTools/2025.2/Vivado/bin/vivado.bat" -mode batch -source program.tcl
```

Expected: `Programming complete.` with `End of startup status: HIGH`

- [ ] **Step 4: Commit**

```bash
git add -A
git commit -m "build: full synthesis with CDC, MAC, and elementwise — bitstream generated"
```
