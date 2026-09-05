/* Random INT8 matrix-vector product through DMA -> mac_controller -> DMA,
 * compared against a reference computed in the testbench.  M=24, K=48. */
`timescale 1ns / 1ps
module tb_mac_golden;
    localparam M = 24, K = 48;
    localparam W_BASE = 27'h1000, X_BASE = 27'h3000, Y_BASE = 27'h4000;
    reg clk = 0, rst_n = 0; always #6 clk = ~clk;
    wire [26:0] app_addr; wire [2:0] app_cmd; wire app_en; wire [127:0] app_wdf_data;
    wire [15:0] app_wdf_mask; wire app_wdf_wren, app_wdf_end; wire [127:0] app_rd_data;
    wire app_rd_data_valid, app_rdy, app_wdf_rdy;
    reg dma_wr_en = 0; reg [26:0] dma_wr_addr = 0; reg [7:0] dma_wr_byte = 0; wire dma_wr_ready;
    reg dma_rd_en = 0; reg [26:0] dma_rd_addr = 0; wire dma_rd_ready; wire [7:0] dma_rd_byte; wire dma_rd_valid;
    wire mac_rd_en; wire [26:0] mac_rd_addr; wire [127:0] mac_rd_data; wire mac_rd_valid;
    wire mac_wr_en; wire [26:0] mac_wr_addr; wire [127:0] mac_wr_data;
    reg start = 0; wire done;
    wire mac_start, mac_load_b; wire [511:0] a_rows; wire [63:0] bvec;
    wire [31:0] r0,r1,r2,r3,r4,r5,r6,r7; wire mac_done;

    ddr2_ui_model #(.BYTES(1<<16)) mem (.clk(clk), .rst_n(rst_n), .app_addr(app_addr), .app_cmd(app_cmd),
        .app_en(app_en), .app_wdf_data(app_wdf_data), .app_wdf_mask(app_wdf_mask), .app_wdf_wren(app_wdf_wren),
        .app_wdf_end(app_wdf_end), .app_rd_data(app_rd_data), .app_rd_data_valid(app_rd_data_valid),
        .app_rdy(app_rdy), .app_wdf_rdy(app_wdf_rdy));
    ddr2_arbiter arb (.clk(clk), .rst_n(rst_n), .app_addr(app_addr), .app_cmd(app_cmd), .app_en(app_en),
        .app_wdf_data(app_wdf_data), .app_wdf_mask(app_wdf_mask), .app_wdf_wren(app_wdf_wren), .app_wdf_end(app_wdf_end),
        .app_rd_data(app_rd_data), .app_rd_data_valid(app_rd_data_valid), .app_rdy(app_rdy), .app_wdf_rdy(app_wdf_rdy),
        .nmc_rd_en(1'b0), .nmc_rd_addr(27'd0), .nmc_wr_en(1'b0), .nmc_wr_addr(27'd0), .nmc_wr_data(128'd0),
        .mac_rd_en(mac_rd_en), .mac_rd_addr(mac_rd_addr), .mac_rd_data(mac_rd_data), .mac_rd_valid(mac_rd_valid),
        .mac_wr_en(mac_wr_en), .mac_wr_addr(mac_wr_addr), .mac_wr_data(mac_wr_data),
        .dma_wr_en(dma_wr_en), .dma_wr_addr(dma_wr_addr), .dma_wr_byte(dma_wr_byte), .dma_wr_ready(dma_wr_ready),
        .dma_rd_en(dma_rd_en), .dma_rd_addr(dma_rd_addr), .dma_rd_ready(dma_rd_ready),
        .dma_rd_byte(dma_rd_byte), .dma_rd_valid(dma_rd_valid));
    mac_controller ctrl (.clk(clk), .rst_n(rst_n), .start(start), .weight_addr({5'd0, W_BASE}),
        .input_addr({5'd0, X_BASE}), .output_addr({5'd0, Y_BASE}), .M(M), .K(K), .done(done),
        .mem_rd_en(mac_rd_en), .mem_rd_addr(mac_rd_addr), .mem_rd_data(mac_rd_data), .mem_rd_valid(mac_rd_valid),
        .mem_wr_en(mac_wr_en), .mem_wr_addr(mac_wr_addr), .mem_wr_data(mac_wr_data),
        .mac_start(mac_start), .mac_load_b(mac_load_b), .mac_a_rows(a_rows), .mac_b(bvec),
        .mac_result_0(r0), .mac_result_1(r1), .mac_result_2(r2), .mac_result_3(r3),
        .mac_result_4(r4), .mac_result_5(r5), .mac_result_6(r6), .mac_result_7(r7), .mac_done(mac_done));
    mac_array_8x8 arr (.clk(clk), .rst_n(rst_n), .start(mac_start), .load_b(mac_load_b), .a_rows(a_rows), .b(bvec),
        .result_0(r0), .result_1(r1), .result_2(r2), .result_3(r3), .result_4(r4), .result_5(r5), .result_6(r6), .result_7(r7),
        .done(mac_done));

    reg signed [7:0] W [0:M*K-1]; reg signed [7:0] X [0:K-1]; reg signed [31:0] Yref [0:M-1];
    integer m, k, errs = 0, cyc0, cyc1; reg [7:0] got; reg [31:0] y;
    task wr_byte(input [26:0] a, input [7:0] d); begin
        @(negedge clk); dma_wr_addr = a; dma_wr_byte = d; dma_wr_en = 1;
        @(posedge clk); while (!dma_wr_ready) @(posedge clk);   /* transfer at this edge */
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
        for (m = 0; m < M*K; m = m + 1) W[m] = $urandom;
        for (k = 0; k < K; k = k + 1) X[k] = $urandom;
        for (m = 0; m < M; m = m + 1) begin Yref[m] = 0; for (k = 0; k < K; k = k + 1) Yref[m] = Yref[m] + W[m*K+k] * X[k]; end
        repeat (4) @(negedge clk); rst_n = 1; repeat (4) @(negedge clk);
        for (m = 0; m < M*K; m = m + 1) wr_byte(W_BASE + m, W[m]);
        for (k = 0; k < K; k = k + 1) wr_byte(X_BASE + k, X[k]);
        repeat (200) @(negedge clk);
        $display("=== MAC golden: M=%0d K=%0d ===", M, K);
        start = 1; cyc0 = $time; @(negedge clk); start = 0;
        while (!done) @(negedge clk); cyc1 = $time;
        repeat (200) @(negedge clk);
        for (m = 0; m < M; m = m + 1) begin
            y = 0; for (k = 0; k < 4; k = k + 1) begin rd_byte(Y_BASE + m*4 + k, got); y[k*8 +: 8] = got; end
            if ($signed(y) !== Yref[m]) begin errs = errs + 1; if (errs < 10) $display("  row %0d: got %0d want %0d", m, $signed(y), Yref[m]); end
        end
        $display("  compute cycles: %0d  (%0d MACs -> %0.3f MAC/cycle)", (cyc1-cyc0)/12, M*K, (M*K*12.0)/(cyc1-cyc0));
        if (errs == 0) $display("=== TEST PASSED ==="); else $display("=== TEST FAILED: %0d/%0d rows wrong ===", errs, M);
        $finish;
    end
endmodule
