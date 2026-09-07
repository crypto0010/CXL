/* EPILOGUE golden: acc INT32[M] + bias INT32[M], * mult >>> shift, relu, sat8.
 * M = 44 (partial final group of 16) — exercises the zero-padded flush. */
`timescale 1ns / 1ps
module tb_epilogue_golden;
    localparam M = 44, NW = (M + 3) / 4;
    localparam A_BASE = 27'h1000, B_BASE = 27'h2000, O_BASE = 27'h3000;
`ifndef EPI_MULT
`define EPI_MULT 16'd1187
`define EPI_SHIFT 5'd14
`define EPI_RELU 1'b1
`endif
    localparam [15:0] MULT = `EPI_MULT; localparam [4:0] SHIFT = `EPI_SHIFT; localparam RELU = `EPI_RELU;
    reg clk = 0, rst_n = 0; always #6 clk = ~clk;
    wire [26:0] app_addr; wire [2:0] app_cmd; wire app_en; wire [127:0] app_wdf_data;
    wire [15:0] app_wdf_mask; wire app_wdf_wren, app_wdf_end; wire [127:0] app_rd_data;
    wire app_rd_data_valid, app_rdy, app_wdf_rdy;
    reg dma_wr_en = 0; reg [26:0] dma_wr_addr = 0; reg [7:0] dma_wr_byte = 0; wire dma_wr_ready;
    reg dma_rd_en = 0; reg [26:0] dma_rd_addr = 0; wire dma_rd_ready; wire [7:0] dma_rd_byte; wire dma_rd_valid;
    wire e_rd_en; wire [26:0] e_rd_addr; wire [127:0] e_rd_data; wire e_rd_valid;
    wire e_wr_en; wire [26:0] e_wr_addr; wire [127:0] e_wr_data;
    reg start = 0; wire done;
    ddr2_ui_model #(.BYTES(1<<16)) mem (.clk(clk), .rst_n(rst_n), .app_addr(app_addr), .app_cmd(app_cmd),
        .app_en(app_en), .app_wdf_data(app_wdf_data), .app_wdf_mask(app_wdf_mask), .app_wdf_wren(app_wdf_wren),
        .app_wdf_end(app_wdf_end), .app_rd_data(app_rd_data), .app_rd_data_valid(app_rd_data_valid),
        .app_rdy(app_rdy), .app_wdf_rdy(app_wdf_rdy));
    ddr2_arbiter arb (.clk(clk), .rst_n(rst_n), .app_addr(app_addr), .app_cmd(app_cmd), .app_en(app_en),
        .app_wdf_data(app_wdf_data), .app_wdf_mask(app_wdf_mask), .app_wdf_wren(app_wdf_wren), .app_wdf_end(app_wdf_end),
        .app_rd_data(app_rd_data), .app_rd_data_valid(app_rd_data_valid), .app_rdy(app_rdy), .app_wdf_rdy(app_wdf_rdy),
        .nmc_rd_en(e_rd_en), .nmc_rd_addr(e_rd_addr), .nmc_rd_data(e_rd_data), .nmc_rd_valid(e_rd_valid),
        .nmc_wr_en(e_wr_en), .nmc_wr_addr(e_wr_addr), .nmc_wr_data(e_wr_data),
        .mac_rd_en(1'b0), .mac_rd_addr(27'd0), .mac_wr_en(1'b0), .mac_wr_addr(27'd0), .mac_wr_data(128'd0),
        .dma_wr_en(dma_wr_en), .dma_wr_addr(dma_wr_addr), .dma_wr_byte(dma_wr_byte), .dma_wr_ready(dma_wr_ready),
        .dma_rd_en(dma_rd_en), .dma_rd_addr(dma_rd_addr), .dma_rd_ready(dma_rd_ready),
        .dma_rd_byte(dma_rd_byte), .dma_rd_valid(dma_rd_valid));
    eltwise_controller elt (.clk(clk), .rst_n(rst_n), .start(start), .input_addr({5'd0, A_BASE}), .addr2({5'd0, B_BASE}),
        .output_addr({5'd0, O_BASE}), .num_words(NW), .op(3'd4), .scale({RELU, 2'b00, SHIFT}), .mult(MULT), .done(done),
        .mem_rd_en(e_rd_en), .mem_rd_addr(e_rd_addr), .mem_rd_data(e_rd_data), .mem_rd_valid(e_rd_valid),
        .mem_wr_en(e_wr_en), .mem_wr_addr(e_wr_addr), .mem_wr_data(e_wr_data));
    reg signed [31:0] A [0:M-1]; reg signed [31:0] B [0:M-1]; reg signed [7:0] Yref [0:M-1];
    reg signed [63:0] t; integer m, j, errs = 0; reg [7:0] got;
    task wr_byte(input [26:0] a, input [7:0] d); begin
        @(negedge clk); dma_wr_addr = a; dma_wr_byte = d; dma_wr_en = 1;
        @(posedge clk); while (!dma_wr_ready) @(posedge clk);
        @(negedge clk); dma_wr_en = 0;
    end endtask
    task rd_byte(input [26:0] a, output [7:0] d); begin
        @(negedge clk); dma_rd_addr = a; dma_rd_en = 1;
        @(posedge clk); while (!dma_rd_ready) @(posedge clk);
        @(negedge clk); dma_rd_en = 0;
        @(posedge clk); while (!dma_rd_valid) @(posedge clk);
        d = dma_rd_byte;
    end endtask
    initial begin
        for (m = 0; m < M; m = m + 1) begin
            A[m] = $signed($urandom) >>> 6; B[m] = $signed($urandom) >>> 12;
            t = (A[m] + B[m]); t = t * $signed(MULT); t = t >>> SHIFT;
            if (RELU && t < 0) t = 0;
            Yref[m] = (t > 127) ? 127 : (t < -128) ? -128 : t[7:0];
        end
        repeat (4) @(negedge clk); rst_n = 1; repeat (4) @(negedge clk);
        for (m = 0; m < M; m = m + 1) for (j = 0; j < 4; j = j + 1) begin
            wr_byte(A_BASE + m*4 + j, A[m][j*8 +: 8]); wr_byte(B_BASE + m*4 + j, B[m][j*8 +: 8]); end
        repeat (200) @(negedge clk);
        $display("=== Epilogue golden: M=%0d mult=%0d shift=%0d relu=%0d ===", M, MULT, SHIFT, RELU);
        @(negedge clk); start = 1; @(negedge clk); start = 0; while (!done) @(posedge clk);
        repeat (200) @(negedge clk);
        for (m = 0; m < M; m = m + 1) begin rd_byte(O_BASE + m, got);
            if ($signed(got) !== Yref[m]) begin errs = errs + 1; if (errs < 8) $display("  m=%0d got %0d want %0d (acc %0d bias %0d)", m, $signed(got), Yref[m], A[m], B[m]); end end
        if (errs == 0) $display("=== TEST PASSED ==="); else $display("=== TEST FAILED: %0d errors ===", errs);
        $finish;
    end
endmodule
