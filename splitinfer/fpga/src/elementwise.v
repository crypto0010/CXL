/* splitinfer/fpga/src/elementwise.v */
`timescale 1ns / 1ps

module elementwise (
    input wire clk, input wire rst_n, input wire start,
    input wire [1:0] op, input wire [127:0] data_in, input wire [7:0] scale_factor,
    output reg [127:0] data_out, output reg done
);
    integer i; reg signed [7:0] val; reg signed [15:0] product;

    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin data_out <= 0; done <= 0; end
        else begin
            done <= 0;
            if (start) begin
                for (i = 0; i < 16; i = i+1) begin
                    val = $signed(data_in[i*8 +: 8]);
                    case (op)
                        2'b00: data_out[i*8 +: 8] <= (val < 0) ? 8'd0 : data_in[i*8 +: 8];
                        2'b01: begin
                            product = val + $signed({1'b0, scale_factor});
                            data_out[i*8 +: 8] <= (product > 127) ? 8'd127 :
                                                   (product < -128) ? 8'h80 : product[7:0];
                        end
                        2'b10: begin
                            product = val * $signed({1'b0, scale_factor});
                            data_out[i*8 +: 8] <= (product > 127) ? 8'd127 :
                                                   (product < -128) ? 8'h80 : product[7:0];
                        end
                        default: data_out[i*8 +: 8] <= data_in[i*8 +: 8];
                    endcase
                end
                done <= 1;
            end
        end
    end
endmodule
