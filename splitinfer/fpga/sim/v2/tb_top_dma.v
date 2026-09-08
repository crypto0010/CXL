/* Full-chip: UART bytes into top -> controller -> CDC FIFOs -> arbiter ->
 * behavioural MIG -> back out over UART.  Exercises exactly the path the
 * board failed on (DATA_READ returned a header and no payload). */
`timescale 1ns / 1ps
module tb_top_dma;
    localparam BAUD = 2_000_000;             // sim-only; overrides top's 115200
    localparam BIT_NS = 1_000_000_000 / BAUD;
    reg sys_clk = 0; always #5 sys_clk = ~sys_clk;
    reg sys_rst_n = 0; reg uart_rx = 1; wire uart_tx; wire [3:0] led;
    wire [15:0] ddr2_dq; wire [1:0] ddr2_dqs_p, ddr2_dqs_n; wire [12:0] ddr2_addr; wire [2:0] ddr2_ba;
    wire ddr2_ras_n, ddr2_cas_n, ddr2_we_n; wire [0:0] ddr2_ck_p, ddr2_ck_n, ddr2_cke, ddr2_cs_n, ddr2_odt; wire [1:0] ddr2_dm;
    top dut (.sys_clk(sys_clk), .sys_rst_n(sys_rst_n), .uart_rx(uart_rx), .uart_tx(uart_tx), .led(led),
        .ddr2_dq(ddr2_dq), .ddr2_dqs_p(ddr2_dqs_p), .ddr2_dqs_n(ddr2_dqs_n), .ddr2_addr(ddr2_addr), .ddr2_ba(ddr2_ba),
        .ddr2_ras_n(ddr2_ras_n), .ddr2_cas_n(ddr2_cas_n), .ddr2_we_n(ddr2_we_n), .ddr2_ck_p(ddr2_ck_p), .ddr2_ck_n(ddr2_ck_n),
        .ddr2_cke(ddr2_cke), .ddr2_cs_n(ddr2_cs_n), .ddr2_dm(ddr2_dm), .ddr2_odt(ddr2_odt));
    defparam dut.u_usb.BAUD_RATE = BAUD;

    /* UART driver */
    task send_byte(input [7:0] b); integer i; begin
        uart_rx = 0; #BIT_NS; for (i = 0; i < 8; i = i + 1) begin uart_rx = b[i]; #BIT_NS; end uart_rx = 1; #BIT_NS;
    end endtask
    /* UART receiver -> queue */
    reg [7:0] rxq [0:4095]; integer rxn = 0, rxr = 0;
    initial forever begin
        @(negedge uart_tx); #(BIT_NS / 2); if (uart_tx == 0) begin : rb integer i; reg [7:0] b;
            for (i = 0; i < 8; i = i + 1) begin #BIT_NS; b[i] = uart_tx; end #BIT_NS; rxq[rxn] = b; rxn = rxn + 1; end
    end
    task expect_bytes(input integer n, input integer timeout_us, output integer ok); integer t; begin
        t = 0; while (rxn - rxr < n && t < timeout_us) begin #1000; t = t + 1; end ok = (rxn - rxr >= n);
    end endtask
    task hdr(input [7:0] t, input [31:0] plen); begin
        send_byte(t); send_byte(0); send_byte(0); send_byte(0);
        send_byte(plen[7:0]); send_byte(plen[15:8]); send_byte(plen[23:16]); send_byte(plen[31:24]);
    end endtask
    task u32(input [31:0] v); begin send_byte(v[7:0]); send_byte(v[15:8]); send_byte(v[23:16]); send_byte(v[31:24]); end endtask

    integer ok, i, errs = 0; reg [7:0] pat [0:63];
    task show_state; begin
        $display("    ctrl.state=%0d dma_rd_en=%b rd_requested=%0d rd_sent=%0d have=%b | cmdfifo empty=%b full=%b | arb: dr_busy=%b dr_req=%b rd_busy=%b owner=%0d issuing=%b wa_valid=%b flush=%b rd_ready=%b | datafifo empty=%b",
            dut.u_edgecoh.state, dut.u_edgecoh.dma_rd_en, dut.u_edgecoh.dma_rd_requested, dut.u_edgecoh.dma_rd_sent, dut.u_edgecoh.dma_rd_have,
            dut.u_dma_rd_cmd_fifo.rd_empty, dut.u_dma_rd_cmd_fifo.wr_full,
            dut.u_arb.dr_busy, dut.u_arb.dr_req, dut.u_arb.rd_busy, dut.u_arb.rd_owner, dut.u_arb.issuing, dut.u_arb.wa_valid, dut.u_arb.wa_flush_req, dut.u_arb.dma_rd_ready,
            dut.u_dma_rd_data_fifo.rd_empty);
    end endtask

    initial begin
        #200 sys_rst_n = 1;
        wait (led[0] == 1); #2000;
        $display("=== top DMA test (calib done at %0t) ===", $time);
        /* 1. barrier */
        hdr(8'h03, 0); expect_bytes(8, 500, ok);
        $display("[barrier] ok=%0d type=%02x", ok, rxq[rxr]); if (!ok || rxq[rxr] != 8'hFE) errs = errs + 1; rxr = rxn;
        /* 2. write 64 bytes @0x10000 */
        for (i = 0; i < 64; i = i + 1) pat[i] = (i * 7 + 3);
        hdr(8'h10, 68); u32(32'h10000); for (i = 0; i < 64; i = i + 1) send_byte(pat[i]);
        expect_bytes(8, 500, ok); $display("[write] ok=%0d type=%02x", ok, rxq[rxr]); if (!ok || rxq[rxr] != 8'hFE) errs = errs + 1; rxr = rxn;
        /* 3. read back */
        hdr(8'h11, 8); u32(32'h10000); u32(64);
        expect_bytes(8, 500, ok); $display("[read hdr] ok=%0d type=%02x len=%0d", ok, rxq[rxr], {rxq[rxr+7], rxq[rxr+6], rxq[rxr+5], rxq[rxr+4]});
        if (!ok || rxq[rxr] != 8'h12) errs = errs + 1; rxr = rxr + 8;
        expect_bytes(64, 2000, ok); $display("[read payload] ok=%0d got %0d bytes", ok, rxn - rxr);
        if (!ok) begin errs = errs + 1; show_state; end
        else begin for (i = 0; i < 64; i = i + 1) if (rxq[rxr + i] !== pat[i]) begin errs = errs + 1; if (errs < 6) $display("  byte %0d: got %02x want %02x", i, rxq[rxr+i], pat[i]); end end
        rxr = rxn;
        /* 4. embedding gather: 4 rows x 32 B table @0x20000, idx [2,0] @0x21000, out @0x22000 */
        hdr(8'h10, 4 + 128); u32(32'h20000); for (i = 0; i < 128; i = i + 1) send_byte(i);
        expect_bytes(8, 500, ok); rxr = rxn;
        hdr(8'h10, 4 + 8); u32(32'h21000); u32(2); u32(0);
        expect_bytes(8, 500, ok); rxr = rxn;
        hdr(8'h20, 25); send_byte(8'h01); u32(32'h20000); u32(4); u32(32); u32(32'h21000); u32(2); u32(32'h22000);
        expect_bytes(8, 2000, ok); $display("[nmc gather] ok=%0d type=%02x", ok, rxq[rxr]); if (!ok || rxq[rxr] != 8'hFE) errs = errs + 1; rxr = rxn;
        hdr(8'h11, 8); u32(32'h22000); u32(64);
        expect_bytes(72, 3000, ok); $display("[gather readback] ok=%0d", ok);
        if (ok) begin for (i = 0; i < 32; i = i + 1) begin if (rxq[rxr+8+i] !== 64+i) errs = errs + 1; if (rxq[rxr+8+32+i] !== i) errs = errs + 1; end end else begin errs = errs + 1; show_state; end
        if (errs == 0) $display("=== TEST PASSED ==="); else $display("=== TEST FAILED: %0d errors ===", errs);
        $finish;
    end
    initial begin #60_000_000; $display("=== TIMEOUT ==="); show_state; $finish; end
endmodule
