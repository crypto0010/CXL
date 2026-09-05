/* splitinfer/fpga/src/elementwise.v  (v2)
 *
 * Single-cycle elementwise unit on 128-bit words.
 *   op 0  RELU8    : 16 x INT8  relu
 *   op 1  ADDS8    : 16 x INT8  + scalar, saturating
 *   op 2  MULS8    : 16 x INT8  * scalar, saturating
 *   op 3  ADDV8    : 16 x INT8  + 16 x INT8 (data_b), saturating
 *   op 4  EPILOGUE : 4 x INT32 acc (data_in) + 4 x INT32 bias (data_b),
 *                    * mult (INT16) >>> shift, optional ReLU, saturate to
 *                    INT8.  Emits 4 bytes in data_out[31:0].  This is the
 *                    standard per-tensor integer requantisation, so the
 *                    host reference can reproduce it bit-exactly.
 */
`timescale 1ns / 1ps

module elementwise (
    input  wire         clk, input wire rst_n, input wire start,
    input  wire [2:0]   op,
    input  wire [127:0] data_in,
    input  wire [127:0] data_b,
    input  wire [7:0]   scale_factor,   // scalar for ops 1/2; {relu, 2'b0, shift[4:0]} for op 4
    input  wire [15:0]  mult,           // op 4 multiplier
    output reg  [127:0] data_out,
    output reg          done
);
    integer i;
    reg signed [7:0]  v8, w8;
    reg signed [15:0] p16;
    reg signed [32:0] s33;
    reg signed [48:0] p49;
    reg signed [48:0] sh49;

    function [7:0] sat8_16; input signed [15:0] x; begin
        sat8_16 = (x > 127) ? 8'd127 : (x < -128) ? 8'h80 : x[7:0];
    end endfunction
    function [7:0] sat8_49; input signed [48:0] x; begin
        sat8_49 = (x > 127) ? 8'd127 : (x < -128) ? 8'h80 : x[7:0];
    end endfunction

    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin data_out <= 0; done <= 0; end
        else begin
            done <= 0;
            if (start) begin
                done <= 1;
                case (op)
                    3'd0: for (i = 0; i < 16; i = i + 1) begin
                        v8 = $signed(data_in[i*8 +: 8]);
                        data_out[i*8 +: 8] <= (v8 < 0) ? 8'd0 : v8;
                    end
                    3'd1: for (i = 0; i < 16; i = i + 1) begin
                        v8 = $signed(data_in[i*8 +: 8]);
                        p16 = v8 + $signed({1'b0, scale_factor});
                        data_out[i*8 +: 8] <= sat8_16(p16);
                    end
                    3'd2: for (i = 0; i < 16; i = i + 1) begin
                        v8 = $signed(data_in[i*8 +: 8]);
                        p16 = v8 * $signed({1'b0, scale_factor});
                        data_out[i*8 +: 8] <= sat8_16(p16);
                    end
                    3'd3: for (i = 0; i < 16; i = i + 1) begin
                        v8 = $signed(data_in[i*8 +: 8]); w8 = $signed(data_b[i*8 +: 8]);
                        p16 = v8 + w8;
                        data_out[i*8 +: 8] <= sat8_16(p16);
                    end
                    3'd4: begin
                        data_out <= 0;
                        for (i = 0; i < 4; i = i + 1) begin
                            s33  = $signed(data_in[i*32 +: 32]) + $signed(data_b[i*32 +: 32]);
                            p49  = s33 * $signed(mult);
                            sh49 = p49 >>> scale_factor[4:0];
                            if (scale_factor[7] && sh49 < 0) sh49 = 0;
                            data_out[i*8 +: 8] <= sat8_49(sh49);
                        end
                    end
                    default: data_out <= data_in;
                endcase
            end
        end
    end
endmodule
