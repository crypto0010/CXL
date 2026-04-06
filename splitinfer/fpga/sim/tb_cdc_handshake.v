/* splitinfer/fpga/sim/tb_cdc_handshake.v
 *
 * Testbench for cdc_handshake: transfers 3 values across async clocks.
 */
`timescale 1ns / 1ps

module tb_cdc_handshake;

    parameter WIDTH = 8;

    reg              src_clk, dst_clk;
    reg              src_rst_n, dst_rst_n;
    reg              src_valid;
    reg  [WIDTH-1:0] src_data;
    wire             src_ready;
    wire             dst_valid;
    wire [WIDTH-1:0] dst_data;

    cdc_handshake #(.WIDTH(WIDTH)) uut (
        .src_clk(src_clk), .src_rst_n(src_rst_n),
        .src_valid(src_valid), .src_data(src_data), .src_ready(src_ready),
        .dst_clk(dst_clk), .dst_rst_n(dst_rst_n),
        .dst_valid(dst_valid), .dst_data(dst_data)
    );

    /* 100 MHz src_clk */
    initial src_clk = 0;
    always #5 src_clk = ~src_clk;

    /* ~81.25 MHz dst_clk (period 12.3 ns) */
    initial dst_clk = 0;
    always #6.15 dst_clk = ~dst_clk;

    integer pass_cnt;
    integer fail_cnt;

    task send_and_check;
        input [WIDTH-1:0] val;
        input [WIDTH-1:0] exp;
        begin
            /* Wait for src_ready */
            @(posedge src_clk);
            while (!src_ready) @(posedge src_clk);
            src_data  <= val;
            src_valid <= 1'b1;
            @(posedge src_clk);
            src_valid <= 1'b0;
            /* Wait for dst_valid pulse */
            @(posedge dst_clk);
            while (!dst_valid) @(posedge dst_clk);
            if (dst_data === exp) begin
                $display("PASS: received 0x%02X (expected 0x%02X)", dst_data, exp);
                pass_cnt = pass_cnt + 1;
            end else begin
                $display("FAIL: received 0x%02X (expected 0x%02X)", dst_data, exp);
                fail_cnt = fail_cnt + 1;
            end
        end
    endtask

    initial begin
        pass_cnt  = 0;
        fail_cnt  = 0;
        src_rst_n = 0;
        dst_rst_n = 0;
        src_valid = 0;
        src_data  = 0;
        #50;
        src_rst_n = 1;
        dst_rst_n = 1;
        #50;

        send_and_check(8'hAB, 8'hAB);
        send_and_check(8'hCD, 8'hCD);
        send_and_check(8'h42, 8'h42);

        #100;
        if (fail_cnt == 0)
            $display("ALL PASSED (%0d/%0d)", pass_cnt, pass_cnt);
        else
            $display("FAILED: %0d passed, %0d failed", pass_cnt, fail_cnt);
        $finish;
    end

endmodule
