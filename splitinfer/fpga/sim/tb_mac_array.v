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
