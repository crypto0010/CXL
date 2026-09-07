/* splitinfer/fpga/src/elementwise.v  (v2.1 — pipelined)
 *
 * Elementwise unit on 128-bit words.  Five-stage pipeline; `done` pulses
 * five cycles after `start` with data_out valid on the same edge.  The
 * v2.0 single-cycle EPILOGUE (33-bit add, 33x16 multiply, 49-bit shift,
 * ReLU, saturate) was 14.9 ns of logic against a 12.3 ns ui_clk period and
 * was the only timing violation in the design (WNS -2.65 ns).
 *
 *   op 0  RELU8    : 16 x INT8  relu
 *   op 1  ADDS8    : 16 x INT8  + scalar, saturating
 *   op 2  MULS8    : 16 x INT8  * scalar, saturating
 *   op 3  ADDV8    : 16 x INT8  + 16 x INT8 (data_b), saturating
 *   op 4  EPILOGUE : 4 x INT32 acc (data_in) + 4 x INT32 bias (data_b),
 *                    * mult (INT16) >>> shift, optional ReLU, saturate to
 *                    INT8; 4 bytes in data_out[31:0].
 *
 * Arithmetic is bit-identical to v2.0 (and to the host reference): the
 * 33x16 product is formed as (hi16s * mult) << 17 + (lo17u * mult), which
 * maps onto two DSP48E1s in parallel instead of a cascaded pair.
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

    function [7:0] sat8_16; input signed [15:0] x; begin
        sat8_16 = (x > 127) ? 8'd127 : (x < -128) ? 8'h80 : x[7:0];
    end endfunction
    function [7:0] sat8_49; input signed [48:0] x; begin
        sat8_49 = (x > 127) ? 8'd127 : (x < -128) ? 8'h80 : x[7:0];
    end endfunction

    /* ── Stage 1: byte ops resolved; epilogue adds ── */
    reg               v1;  reg [2:0] op1; reg [7:0] sc1; reg signed [15:0] m1;
    reg [127:0]       r1;
    reg signed [32:0] s33 [0:3];
    /* ── Stage 2: partial products ── */
    reg               v2;  reg [2:0] op2; reg [7:0] sc2;
    reg [127:0]       r2;
    reg signed [33:0] pp_lo [0:3];
    reg signed [31:0] pp_hi [0:3];
    /* ── Stage 3: combine ── */
    reg               v3;  reg [2:0] op3; reg [7:0] sc3;
    reg [127:0]       r3;
    reg signed [48:0] p49 [0:3];
    /* ── Stage 4: shift ── */
    reg               v4;  reg [2:0] op4; reg [7:0] sc4;
    reg [127:0]       r4;
    reg signed [48:0] sh49 [0:3];

    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            v1 <= 0; v2 <= 0; v3 <= 0; v4 <= 0; done <= 0; data_out <= 0;
            op1 <= 0; op2 <= 0; op3 <= 0; op4 <= 0; sc1 <= 0; sc2 <= 0; sc3 <= 0; sc4 <= 0; m1 <= 0;
            r1 <= 0; r2 <= 0; r3 <= 0; r4 <= 0;
            for (i = 0; i < 4; i = i + 1) begin s33[i] <= 0; pp_lo[i] <= 0; pp_hi[i] <= 0; p49[i] <= 0; sh49[i] <= 0; end
        end else begin
            /* Stage 1 */
            v1 <= start; op1 <= op; sc1 <= scale_factor; m1 <= $signed(mult);
            if (start) begin
                case (op)
                    3'd0: for (i = 0; i < 16; i = i + 1) begin
                        v8 = $signed(data_in[i*8 +: 8]); r1[i*8 +: 8] <= (v8 < 0) ? 8'd0 : v8; end
                    3'd1: for (i = 0; i < 16; i = i + 1) begin
                        v8 = $signed(data_in[i*8 +: 8]); p16 = v8 + $signed({1'b0, scale_factor}); r1[i*8 +: 8] <= sat8_16(p16); end
                    3'd2: for (i = 0; i < 16; i = i + 1) begin
                        v8 = $signed(data_in[i*8 +: 8]); p16 = v8 * $signed({1'b0, scale_factor}); r1[i*8 +: 8] <= sat8_16(p16); end
                    3'd3: for (i = 0; i < 16; i = i + 1) begin
                        v8 = $signed(data_in[i*8 +: 8]); w8 = $signed(data_b[i*8 +: 8]); p16 = v8 + w8; r1[i*8 +: 8] <= sat8_16(p16); end
                    default: r1 <= data_in;
                endcase
                for (i = 0; i < 4; i = i + 1)
                    s33[i] <= $signed(data_in[i*32 +: 32]) + $signed(data_b[i*32 +: 32]);
            end
            /* Stage 2: two DSP-sized partial products per lane */
            v2 <= v1; op2 <= op1; sc2 <= sc1; r2 <= r1;
            for (i = 0; i < 4; i = i + 1) begin
                pp_lo[i] <= $signed({1'b0, s33[i][16:0]}) * m1;      // 18-bit unsigned-as-signed x 16s
                pp_hi[i] <= $signed(s33[i][32:17]) * m1;             // 16s x 16s
            end
            /* Stage 3: combine */
            v3 <= v2; op3 <= op2; sc3 <= sc2; r3 <= r2;
            for (i = 0; i < 4; i = i + 1)
                p49[i] <= ($signed({{15{pp_hi[i][31]}}, pp_hi[i], 17'd0})) + $signed({{15{pp_lo[i][33]}}, pp_lo[i]});
            /* Stage 4: arithmetic shift */
            v4 <= v3; op4 <= op3; sc4 <= sc3; r4 <= r3;
            for (i = 0; i < 4; i = i + 1)
                sh49[i] <= p49[i] >>> sc3[4:0];
            /* Stage 5: ReLU + saturate */
            done <= v4;
            if (v4) begin
                if (op4 == 3'd4) begin
                    data_out <= 0;
                    for (i = 0; i < 4; i = i + 1)
                        data_out[i*8 +: 8] <= sat8_49((sc4[7] && sh49[i] < 0) ? 49'sd0 : sh49[i]);
                end else data_out <= r4;
            end
        end
    end
endmodule
