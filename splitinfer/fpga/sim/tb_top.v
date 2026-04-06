/* splitinfer/fpga/sim/tb_top.v */
`timescale 1ns / 1ps
module tb_top;
    reg sys_clk, sys_rst_n, uart_rx; wire uart_tx; wire [3:0] led;

    /* DDR2 stubs — unused in SIM_MODE but required as top-level ports */
    wire [15:0] ddr2_dq;
    wire [1:0]  ddr2_dqs_p, ddr2_dqs_n;
    wire [12:0] ddr2_addr;
    wire [2:0]  ddr2_ba;
    wire        ddr2_ras_n, ddr2_cas_n, ddr2_we_n;
    wire [0:0]  ddr2_ck_p, ddr2_ck_n, ddr2_cke, ddr2_cs_n, ddr2_odt;
    wire [1:0]  ddr2_dm;

    top #(.SIM_MODE(1)) uut (
        .sys_clk(sys_clk), .sys_rst_n(sys_rst_n),
        .uart_rx(uart_rx), .uart_tx(uart_tx), .led(led),
        .ddr2_dq(ddr2_dq), .ddr2_dqs_p(ddr2_dqs_p), .ddr2_dqs_n(ddr2_dqs_n),
        .ddr2_addr(ddr2_addr), .ddr2_ba(ddr2_ba),
        .ddr2_ras_n(ddr2_ras_n), .ddr2_cas_n(ddr2_cas_n), .ddr2_we_n(ddr2_we_n),
        .ddr2_ck_p(ddr2_ck_p), .ddr2_ck_n(ddr2_ck_n),
        .ddr2_cke(ddr2_cke), .ddr2_cs_n(ddr2_cs_n),
        .ddr2_dm(ddr2_dm), .ddr2_odt(ddr2_odt)
    );

    always #5 sys_clk = ~sys_clk; /* 100 MHz */

    integer pass_count;

    initial begin
        sys_clk = 0; sys_rst_n = 0; uart_rx = 1; pass_count = 0;
        #100 sys_rst_n = 1;
        #2000;

        if (led[3]) begin $display("PASS: Heartbeat LED active"); pass_count = pass_count + 1; end
        else $display("FAIL: Heartbeat LED not active");

        if (led[0]) begin $display("PASS: DDR2 calibration complete (simulated)"); pass_count = pass_count + 1; end
        else $display("FAIL: DDR2 calibration not signaled");

        $display("%0d/2 tests passed.", pass_count);
        $finish;
    end
endmodule
