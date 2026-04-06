/* splitinfer/fpga/src/mac_array_8x8.v
 *
 * 8x8 INT8 MAC array with pipelined accumulation.
 * Pipeline: stage 1 = multiply + partial sums (4 pairs), stage 2 = final sum + accumulate.
 * done pulses 2 cycles after load_b (1-cycle pipeline latency added).
 */
`timescale 1ns / 1ps

module mac_array_8x8 (
    input wire clk, input wire rst_n,
    input wire start, input wire load_a, input wire load_b,
    input wire [63:0] row_a, input wire [63:0] row_b,
    output wire [31:0] result_0, result_1, result_2, result_3,
    output wire [31:0] result_4, result_5, result_6, result_7,
    output reg done
);

    reg signed [7:0] a_reg [0:7];
    reg signed [7:0] b_reg [0:7];
    reg signed [31:0] acc [0:7];

    /* Pipeline stage 1: partial sums (4 pairs of products) */
    reg signed [31:0] psum0, psum1, psum2, psum3;
    reg pipe_valid_s1;

    /* Pipeline stage 2: final sum + accumulate */
    reg pipe_valid_s2;

    assign result_0 = acc[0]; assign result_1 = acc[1];
    assign result_2 = acc[2]; assign result_3 = acc[3];
    assign result_4 = acc[4]; assign result_5 = acc[5];
    assign result_6 = acc[6]; assign result_7 = acc[7];

    integer i;

    /* Load registers */
    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            for (i = 0; i < 8; i = i+1) begin a_reg[i] <= 0; b_reg[i] <= 0; end
            pipe_valid_s1 <= 0;
        end else begin
            pipe_valid_s1 <= 0;
            if (load_a) for (i = 0; i < 8; i = i+1) a_reg[i] <= $signed(row_a[i*8 +: 8]);
            if (load_b) begin
                for (i = 0; i < 8; i = i+1) b_reg[i] <= $signed(row_b[i*8 +: 8]);
                pipe_valid_s1 <= 1;
            end
        end
    end

    /* Stage 1: compute 4 partial sums (2 products each) */
    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            psum0 <= 0; psum1 <= 0; psum2 <= 0; psum3 <= 0;
            pipe_valid_s2 <= 0;
        end else begin
            pipe_valid_s2 <= pipe_valid_s1;
            if (pipe_valid_s1) begin
                psum0 <= a_reg[0]*b_reg[0] + a_reg[1]*b_reg[1];
                psum1 <= a_reg[2]*b_reg[2] + a_reg[3]*b_reg[3];
                psum2 <= a_reg[4]*b_reg[4] + a_reg[5]*b_reg[5];
                psum3 <= a_reg[6]*b_reg[6] + a_reg[7]*b_reg[7];
            end
        end
    end

    /* Stage 2: final sum + accumulate */
    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            for (i = 0; i < 8; i = i+1) acc[i] <= 0;
            done <= 0;
        end else begin
            done <= 0;
            if (start) for (i = 0; i < 8; i = i+1) acc[i] <= 0;
            if (pipe_valid_s2) begin
                for (i = 0; i < 8; i = i+1)
                    acc[i] <= acc[i] + psum0 + psum1 + psum2 + psum3;
                done <= 1;
            end
        end
    end
endmodule
