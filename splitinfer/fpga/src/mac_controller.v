/* splitinfer/fpga/src/mac_controller.v  (v2)
 *
 * INT8 fully-connected layer:  out[M] (INT32) = W[M x K] (INT8) . x[K] (INT8)
 *
 * Processes EIGHT output rows per pass through K, feeding the 8-lane
 * mac_array_8x8.  Per 16-element K chunk: one 16-byte activation read and
 * eight 16-byte weight reads (one per row), then two array steps (low and
 * high 8-element halves).  Results: eight INT32 = 32 bytes, written as two
 * 16-byte words at output_addr + m*4.
 *
 * Layout / alignment requirements (the host pads to satisfy them):
 *   K % 16 == 0,  M % 8 == 0,  weight_addr / input_addr / output_addr
 *   16-byte aligned.  Weights row-major, INT8.  All addresses are BYTE
 *   addresses (see ddr2_arbiter.v).
 *
 * v1 wrote 32 bytes per row at a 4-byte stride (rows overlapped) and fed a
 * broadcast dot product; see mac_array_8x8.v.
 */
`timescale 1ns / 1ps

module mac_controller (
    input  wire         clk,
    input  wire         rst_n,
    input  wire         start,
    input  wire [31:0]  weight_addr,
    input  wire [31:0]  input_addr,
    input  wire [31:0]  output_addr,
    input  wire [31:0]  M,
    input  wire [31:0]  K,
    output reg          done,
    // Memory port (to DDR2 arbiter) — byte addresses, 16-byte aligned
    output reg          mem_rd_en,
    output reg  [26:0]  mem_rd_addr,
    input  wire [127:0] mem_rd_data,
    input  wire         mem_rd_valid,
    output reg          mem_wr_en,
    output reg  [26:0]  mem_wr_addr,
    output reg  [127:0] mem_wr_data,
    // MAC array
    output reg          mac_start,
    output reg          mac_load_b,
    output reg  [511:0] mac_a_rows,
    output reg  [63:0]  mac_b,
    input  wire [31:0]  mac_result_0, mac_result_1, mac_result_2, mac_result_3,
    input  wire [31:0]  mac_result_4, mac_result_5, mac_result_6, mac_result_7,
    input  wire         mac_done
);
    localparam S_IDLE=4'd0, S_CLEAR=4'd1, S_RD_ACT=4'd2, S_WT_ACT=4'd3,
               S_RD_W=4'd4, S_WT_W=4'd5, S_MAC_LO=4'd6, S_WT_LO=4'd7,
               S_MAC_HI=4'd8, S_WT_HI=4'd9, S_WR0=4'd10, S_WR1=4'd11,
               S_NEXT_M=4'd12, S_DONE=4'd13;

    reg [3:0]   state;
    reg [31:0]  m_idx, k_idx, M_reg, K_reg, w_addr, i_addr, o_addr;
    reg [2:0]   r_idx;
    reg [127:0] act16;               // 16 activations for this K chunk
    reg [127:0] w16 [0:7];           // 16 weights per row for this K chunk
    integer r;

    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            state <= S_IDLE; done <= 0;
            mem_rd_en <= 0; mem_rd_addr <= 0; mem_wr_en <= 0; mem_wr_addr <= 0; mem_wr_data <= 0;
            mac_start <= 0; mac_load_b <= 0; mac_a_rows <= 0; mac_b <= 0;
            m_idx <= 0; k_idx <= 0; M_reg <= 0; K_reg <= 0; w_addr <= 0; i_addr <= 0; o_addr <= 0;
            r_idx <= 0; act16 <= 0;
            for (r = 0; r < 8; r = r + 1) w16[r] <= 0;
        end else begin
            mem_rd_en <= 0; mem_wr_en <= 0; mac_start <= 0; mac_load_b <= 0; done <= 0;
            case (state)
            S_IDLE: if (start) begin
                M_reg <= M; K_reg <= K; w_addr <= weight_addr; i_addr <= input_addr;
                o_addr <= output_addr; m_idx <= 0; state <= S_CLEAR;
            end
            S_CLEAR: begin mac_start <= 1; k_idx <= 0; state <= S_RD_ACT; end
            S_RD_ACT: begin
                mem_rd_en <= 1; mem_rd_addr <= i_addr[26:0] + k_idx[26:0];
                state <= S_WT_ACT;
            end
            S_WT_ACT: if (mem_rd_valid) begin act16 <= mem_rd_data; r_idx <= 0; state <= S_RD_W; end
            S_RD_W: begin
                mem_rd_en <= 1;
                mem_rd_addr <= w_addr[26:0] + (m_idx[26:0] + r_idx) * K_reg[26:0] + k_idx[26:0];
                state <= S_WT_W;
            end
            S_WT_W: if (mem_rd_valid) begin
                w16[r_idx] <= mem_rd_data;
                if (r_idx == 3'd7) state <= S_MAC_LO;
                else begin r_idx <= r_idx + 1; state <= S_RD_W; end
            end
            S_MAC_LO: begin
                for (r = 0; r < 8; r = r + 1) mac_a_rows[r*64 +: 64] <= w16[r][63:0];
                mac_b <= act16[63:0]; mac_load_b <= 1; state <= S_WT_LO;
            end
            S_WT_LO: if (mac_done) state <= S_MAC_HI;
            S_MAC_HI: begin
                for (r = 0; r < 8; r = r + 1) mac_a_rows[r*64 +: 64] <= w16[r][127:64];
                mac_b <= act16[127:64]; mac_load_b <= 1; state <= S_WT_HI;
            end
            S_WT_HI: if (mac_done) begin
                if (k_idx + 16 < K_reg) begin k_idx <= k_idx + 16; state <= S_RD_ACT; end
                else state <= S_WR0;
            end
            S_WR0: begin
                mem_wr_en <= 1; mem_wr_addr <= o_addr[26:0] + (m_idx[26:0] << 2);
                mem_wr_data <= {mac_result_3, mac_result_2, mac_result_1, mac_result_0};
                state <= S_WR1;
            end
            S_WR1: begin
                mem_wr_en <= 1; mem_wr_addr <= o_addr[26:0] + (m_idx[26:0] << 2) + 27'd16;
                mem_wr_data <= {mac_result_7, mac_result_6, mac_result_5, mac_result_4};
                state <= S_NEXT_M;
            end
            S_NEXT_M: begin
                if (m_idx + 8 < M_reg) begin m_idx <= m_idx + 8; state <= S_CLEAR; end
                else state <= S_DONE;
            end
            S_DONE: begin done <= 1; state <= S_IDLE; end
            default: state <= S_IDLE;
            endcase
        end
    end
endmodule
