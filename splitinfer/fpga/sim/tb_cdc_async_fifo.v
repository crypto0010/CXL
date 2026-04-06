/* splitinfer/fpga/sim/tb_cdc_async_fifo.v
 *
 * Testbench for cdc_async_fifo: dual-clock write/read with PASS/FAIL checks.
 */
`timescale 1ns / 1ps

module tb_cdc_async_fifo;

    localparam WIDTH      = 8;
    localparam DEPTH_LOG2 = 3;
    localparam NUM_ITEMS  = 8;

    /* --- Clocks: 100 MHz wr, ~81.25 MHz rd --- */
    reg wr_clk = 0, rd_clk = 0;
    always #5.0  wr_clk = ~wr_clk;   /* 10 ns period */
    always #6.15 rd_clk = ~rd_clk;   /* 12.3 ns period */

    reg wr_rst_n = 0, rd_rst_n = 0;
    reg             wr_en;
    reg [WIDTH-1:0] wr_data;
    wire            wr_full;
    reg             rd_en;
    wire [WIDTH-1:0] rd_data;
    wire             rd_empty;

    cdc_async_fifo #(.WIDTH(WIDTH), .DEPTH_LOG2(DEPTH_LOG2)) uut (
        .wr_clk   (wr_clk),
        .wr_rst_n (wr_rst_n),
        .wr_en    (wr_en),
        .wr_data  (wr_data),
        .wr_full  (wr_full),
        .rd_clk   (rd_clk),
        .rd_rst_n (rd_rst_n),
        .rd_en    (rd_en),
        .rd_data  (rd_data),
        .rd_empty (rd_empty)
    );

    integer i, pass_cnt;
    reg [WIDTH-1:0] expected;

    /* Watchdog */
    initial begin
        #20000;
        $display("TIMEOUT");
        $finish;
    end

    initial begin
        wr_en = 0; rd_en = 0; wr_data = 0;
        pass_cnt = 0;

        /* Release resets */
        #50;
        @(posedge wr_clk); #1; wr_rst_n = 1;
        @(posedge rd_clk); #1; rd_rst_n = 1;
        #40;

        /* Write 8 bytes: 0xA0 .. 0xA7.
         * Drive wr_en and wr_data on negedge so they are
         * stable when the DUT samples on posedge. */
        for (i = 0; i < NUM_ITEMS; i = i + 1) begin
            @(negedge wr_clk);
            wr_en   = 1;
            wr_data = 8'hA0 + i;
        end
        @(negedge wr_clk);
        wr_en = 0;

        /* Wait for synchronizer latency */
        #300;

        /* Read back and verify.
         * Wait for rd_empty to deassert, sample rd_data, then pulse
         * rd_en for one cycle to advance the read pointer. */
        for (i = 0; i < NUM_ITEMS; i = i + 1) begin
            @(posedge rd_clk);
            while (rd_empty) @(posedge rd_clk);
            /* rd_data is combinational on rd_bin, valid now */
            expected = 8'hA0 + i;
            #1;
            if (rd_data === expected) begin
                $display("PASS: read[%0d] = 0x%02X", i, rd_data);
                pass_cnt = pass_cnt + 1;
            end else begin
                $display("FAIL: read[%0d] = 0x%02X, expected 0x%02X",
                         i, rd_data, expected);
            end
            /* Pulse rd_en to advance pointer */
            @(negedge rd_clk);
            rd_en = 1;
            @(negedge rd_clk);
            rd_en = 0;
        end

        #50;
        if (pass_cnt == NUM_ITEMS)
            $display("ALL PASSED (%0d/%0d)", pass_cnt, NUM_ITEMS);
        else
            $display("SOME FAILED (%0d/%0d passed)", pass_cnt, NUM_ITEMS);

        $finish;
    end

endmodule
