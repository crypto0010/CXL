/* splitinfer/fpga/src/mac_array_8x8.v  (v2)
 *
 * 8x8 INT8 multiply-accumulate array: eight INDEPENDENT lanes.
 *   lane j :  acc[j] += dot8(a_row[j], b)
 * where a_row[j] is eight INT8 weights of output row j and b is eight INT8
 * activations.  64 multipliers per step, 2-stage pipeline, INT32 accumulate.
 *
 * v1 summed all 64 products into ONE dot product and broadcast it to all
 * eight accumulators (mac_array_8x8.v:82 in the submitted design), so the
 * "8x8 array" produced eight identical outputs.  v2 is the array the paper
 * described.
 *
 * Handshake: `start` clears the accumulators.  `load_b` latches b together
 * with the a_rows presented on the same cycle and launches one step;
 * `done` pulses two cycles later when the accumulate has committed.
 */
`timescale 1ns / 1ps

module mac_array_8x8 (
    input  wire         clk,
    input  wire         rst_n,
    input  wire         start,       // clear accumulators
    input  wire         load_b,      // launch one 8x8 step
    input  wire [511:0] a_rows,      // 8 rows x 8 INT8 (row j at [j*64 +: 64])
    input  wire [63:0]  b,           // 8 INT8 activations
    output wire [31:0]  result_0, result_1, result_2, result_3,
    output wire [31:0]  result_4, result_5, result_6, result_7,
    output reg          done
);
    reg signed [31:0] acc [0:7];
    reg signed [31:0] ps  [0:7][0:3];   // stage-1 partial sums (pairs)
    reg               v1, v2;
    integer j, k;

    assign result_0 = acc[0]; assign result_1 = acc[1];
    assign result_2 = acc[2]; assign result_3 = acc[3];
    assign result_4 = acc[4]; assign result_5 = acc[5];
    assign result_6 = acc[6]; assign result_7 = acc[7];

    /* Stage 1: 64 products folded into 8x4 pair sums. */
    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            v1 <= 0;
            for (j = 0; j < 8; j = j + 1) for (k = 0; k < 4; k = k + 1) ps[j][k] <= 0;
        end else begin
            v1 <= load_b;
            if (load_b) begin
                for (j = 0; j < 8; j = j + 1) for (k = 0; k < 4; k = k + 1)
                    ps[j][k] <= $signed(a_rows[j*64 + (2*k)*8   +: 8]) * $signed(b[(2*k)*8   +: 8])
                              + $signed(a_rows[j*64 + (2*k+1)*8 +: 8]) * $signed(b[(2*k+1)*8 +: 8]);
            end
        end
    end

    /* Stage 2: accumulate each lane. */
    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            v2 <= 0; done <= 0;
            for (j = 0; j < 8; j = j + 1) acc[j] <= 0;
        end else begin
            v2   <= v1;
            done <= v1;
            if (start) for (j = 0; j < 8; j = j + 1) acc[j] <= 0;
            else if (v1) for (j = 0; j < 8; j = j + 1)
                acc[j] <= acc[j] + ps[j][0] + ps[j][1] + ps[j][2] + ps[j][3];
        end
    end
endmodule
