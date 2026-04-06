/* splitinfer/fpga/sim/tb_elementwise.v */
`timescale 1ns / 1ps

module tb_elementwise;
    reg clk, rst_n, start; reg [1:0] op; reg [127:0] data_in; reg [7:0] scale_factor;
    wire [127:0] data_out; wire done;

    elementwise uut (.clk(clk), .rst_n(rst_n), .start(start), .op(op),
        .data_in(data_in), .scale_factor(scale_factor), .data_out(data_out), .done(done));

    always #5 clk = ~clk;

    initial begin
        clk = 0; rst_n = 0; start = 0; #20 rst_n = 1; #10;

        op <= 2'b00; /* ReLU */
        data_in <= {8'd0,8'd0,8'd0,8'd0, 8'd0,8'd0,8'd0,8'd0,
                    8'd10,8'd127,8'h80,8'd0, 8'd7,8'hFF,8'd3,8'hFB};
        @(posedge clk); start <= 1; @(posedge clk); start <= 0;
        wait(done); #10;
        $display("ReLU output: %h", data_out);

        op <= 2'b10; scale_factor <= 8'd2;
        data_in <= {8'd0,8'd0,8'd0,8'd0, 8'd0,8'd0,8'd0,8'd0,
                    8'd0,8'd0,8'd0,8'd0, 8'd4,8'd3,8'd2,8'd1};
        @(posedge clk); start <= 1; @(posedge clk); start <= 0;
        wait(done); #10;
        $display("Scale output: %h", data_out);
        $display("Elementwise tests completed.");
        $finish;
    end
endmodule
