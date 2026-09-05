/* Embedding gather through DMA -> embedding_lookup -> DMA with a reference.
 * Table 32 rows x 32 bytes (embed_dim=32 -> 2 bursts/row), 12 indices. */
`timescale 1ns / 1ps
module tb_embedding_golden;
    localparam ROWS = 32, DIM = 32, NIDX = 12;
    localparam T_BASE = 27'h2000, I_BASE = 27'h5000, O_BASE = 27'h6000;
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
    embedding_lookup emb (.clk(clk), .rst_n(rst_n), .start(start), .table_base_addr({5'd0, T_BASE}), .embed_dim(DIM),
        .indices_addr({5'd0, I_BASE}), .num_indices(NIDX), .output_addr({5'd0, O_BASE}), .done(done),
        .mem_rd_en(e_rd_en), .mem_rd_addr(e_rd_addr), .mem_rd_data(e_rd_data), .mem_rd_valid(e_rd_valid),
        .mem_wr_en(e_wr_en), .mem_wr_addr(e_wr_addr), .mem_wr_data(e_wr_data));
    reg [7:0] T [0:ROWS*DIM-1]; reg [31:0] IDX [0:NIDX-1];
    integer i, j, errs = 0; reg [7:0] got;
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
        for (i = 0; i < ROWS*DIM; i = i + 1) T[i] = $urandom;
        for (i = 0; i < NIDX; i = i + 1) IDX[i] = $urandom % ROWS;
        repeat (4) @(negedge clk); rst_n = 1; repeat (4) @(negedge clk);
        for (i = 0; i < ROWS*DIM; i = i + 1) wr_byte(T_BASE + i, T[i]);
        for (i = 0; i < NIDX; i = i + 1) for (j = 0; j < 4; j = j + 1) wr_byte(I_BASE + i*4 + j, IDX[i][j*8 +: 8]);
        repeat (200) @(negedge clk);
        $display("=== Embedding golden: %0d rows x %0d, %0d indices ===", ROWS, DIM, NIDX);
        start = 1; @(negedge clk); start = 0; while (!done) @(negedge clk);
        repeat (200) @(negedge clk);
        for (i = 0; i < NIDX; i = i + 1) for (j = 0; j < DIM; j = j + 1) begin
            rd_byte(O_BASE + i*DIM + j, got);
            if (got !== T[IDX[i]*DIM + j]) begin errs = errs + 1; if (errs < 8) $display("  idx %0d byte %0d: got %02x want %02x", i, j, got, T[IDX[i]*DIM+j]); end
        end
        if (errs == 0) $display("=== TEST PASSED ==="); else $display("=== TEST FAILED: %0d errors ===", errs);
        $finish;
    end
endmodule
