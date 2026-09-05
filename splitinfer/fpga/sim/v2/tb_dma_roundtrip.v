/* Write N bytes through the DMA byte port at an unaligned start address,
 * read them back byte-by-byte, and compare.  Then verify neighbouring bytes
 * were NOT clobbered (the v1 defect).  Exercises coalescing (bytes every
 * cycle) and the UART-rate path (bytes 200 cycles apart). */
`timescale 1ns / 1ps
module tb_dma_roundtrip;
    reg clk = 0, rst_n = 0; always #6 clk = ~clk;
    wire [26:0] app_addr; wire [2:0] app_cmd; wire app_en; wire [127:0] app_wdf_data;
    wire [15:0] app_wdf_mask; wire app_wdf_wren, app_wdf_end; wire [127:0] app_rd_data;
    wire app_rd_data_valid, app_rdy, app_wdf_rdy;
    reg dma_wr_en = 0; reg [26:0] dma_wr_addr = 0; reg [7:0] dma_wr_byte = 0; wire dma_wr_ready;
    reg dma_rd_en = 0; reg [26:0] dma_rd_addr = 0; wire dma_rd_ready; wire [7:0] dma_rd_byte; wire dma_rd_valid;

    ddr2_ui_model #(.BYTES(1<<16)) mem (.clk(clk), .rst_n(rst_n), .app_addr(app_addr), .app_cmd(app_cmd),
        .app_en(app_en), .app_wdf_data(app_wdf_data), .app_wdf_mask(app_wdf_mask), .app_wdf_wren(app_wdf_wren),
        .app_wdf_end(app_wdf_end), .app_rd_data(app_rd_data), .app_rd_data_valid(app_rd_data_valid),
        .app_rdy(app_rdy), .app_wdf_rdy(app_wdf_rdy));
    ddr2_arbiter arb (.clk(clk), .rst_n(rst_n), .app_addr(app_addr), .app_cmd(app_cmd), .app_en(app_en),
        .app_wdf_data(app_wdf_data), .app_wdf_mask(app_wdf_mask), .app_wdf_wren(app_wdf_wren), .app_wdf_end(app_wdf_end),
        .app_rd_data(app_rd_data), .app_rd_data_valid(app_rd_data_valid), .app_rdy(app_rdy), .app_wdf_rdy(app_wdf_rdy),
        .nmc_rd_en(1'b0), .nmc_rd_addr(27'd0), .nmc_wr_en(1'b0), .nmc_wr_addr(27'd0), .nmc_wr_data(128'd0),
        .mac_rd_en(1'b0), .mac_rd_addr(27'd0), .mac_wr_en(1'b0), .mac_wr_addr(27'd0), .mac_wr_data(128'd0),
        .dma_wr_en(dma_wr_en), .dma_wr_addr(dma_wr_addr), .dma_wr_byte(dma_wr_byte), .dma_wr_ready(dma_wr_ready),
        .dma_rd_en(dma_rd_en), .dma_rd_addr(dma_rd_addr), .dma_rd_ready(dma_rd_ready),
        .dma_rd_byte(dma_rd_byte), .dma_rd_valid(dma_rd_valid));

    integer errs = 0, n, gap;
    reg [7:0] got;
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
    task run_case(input [26:0] base, input integer len, input integer igap); integer k; begin
        // guard bytes on both sides
        for (k = 1; k <= 20; k = k + 1) begin wr_byte(base - k, 8'hA5); wr_byte(base + len - 1 + k, 8'h5A); repeat(40) @(negedge clk); end
        repeat (100) @(negedge clk);
        for (k = 0; k < len; k = k + 1) begin wr_byte(base + k, (k * 7 + 3) & 8'hFF); repeat (igap) @(negedge clk); end
        repeat (100) @(negedge clk);
        for (k = 0; k < len; k = k + 1) begin rd_byte(base + k, got);
            if (got !== ((k * 7 + 3) & 8'hFF)) begin errs = errs + 1; if (errs < 8) $display("  MISMATCH @%0d: got %02x want %02x", k, got, (k*7+3)&8'hFF); end end
        for (k = 1; k <= 20; k = k + 1) begin rd_byte(base - k, got); if (got !== 8'hA5) begin errs = errs + 1; $display("  GUARD-LO clobbered @-%0d = %02x", k, got); end
                                         rd_byte(base + len - 1 + k, got); if (got !== 8'hA5 && got !== 8'h5A) begin errs = errs + 1; $display("  GUARD-HI clobbered @+%0d = %02x", k, got); end end
        $display("  case base=%0d len=%0d gap=%0d -> errs so far %0d", base, len, igap, errs);
    end endtask

    initial begin
        repeat (4) @(negedge clk); rst_n = 1; repeat (4) @(negedge clk);
        $display("=== DMA byte round-trip ===");
        run_case(27'd1003, 300, 0);     // unaligned, back-to-back (coalescing)
        run_case(27'd4096, 64, 0);      // aligned
        run_case(27'd8193, 40, 200);    // UART-like spacing (idle flush per byte)
        if (errs == 0) $display("=== TEST PASSED ==="); else $display("=== TEST FAILED: %0d errors ===", errs);
        $finish;
    end
endmodule
