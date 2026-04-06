/* splitinfer/fpga/src/mac_controller.v
 * MAC Controller FSM – orchestrates INT8 FC layer computation.
 * Reads weight rows and activation chunks from DDR2, feeds mac_array_8x8,
 * writes INT32 results back.
 *
 * NOTE: mac_array_8x8 broadcasts the same dot-product to all 8 accumulators,
 * so this controller effectively computes one output row per m_idx step.
 * Each m_idx iteration processes a single row: acc += sum(w[j]*a[j]) for
 * j in [k_idx .. k_idx+7], accumulated over K/8 steps.
 */
`timescale 1ns / 1ps

module mac_controller (
    input  wire         clk,
    input  wire         rst_n,
    input  wire         start,
    input  wire [31:0]  weight_addr,   // base addr of weight matrix [M x K]
    input  wire [31:0]  input_addr,    // base addr of activation vector [K x 1]
    input  wire [31:0]  output_addr,   // base addr for result vector [M x 1] (INT32)
    input  wire [31:0]  M,             // number of output rows
    input  wire [31:0]  K,             // inner dimension
    output reg          done,

    // Memory port (to DDR2 arbiter)
    output reg          mem_rd_en,
    output reg  [26:0]  mem_rd_addr,
    input  wire [127:0] mem_rd_data,
    input  wire         mem_rd_valid,
    output reg          mem_wr_en,
    output reg  [26:0]  mem_wr_addr,
    output reg  [127:0] mem_wr_data,

    // MAC array interface
    output reg          mac_start,
    output reg          mac_load_a,
    output reg          mac_load_b,
    output reg  [63:0]  mac_row_a,     // 8x INT8 weight values
    output reg  [63:0]  mac_row_b,     // 8x INT8 activation values
    input  wire [31:0]  mac_result_0, mac_result_1, mac_result_2, mac_result_3,
    input  wire [31:0]  mac_result_4, mac_result_5, mac_result_6, mac_result_7,
    input  wire         mac_done
);

    // FSM states
    localparam S_IDLE       = 4'd0,
               S_CLEAR      = 4'd1,
               S_LOAD_ACT   = 4'd2,
               S_WAIT_ACT   = 4'd3,
               S_LOAD_WGT   = 4'd4,
               S_WAIT_WGT   = 4'd5,
               S_MAC        = 4'd6,
               S_NEXT_K     = 4'd7,
               S_WRITE_RES  = 4'd8,
               S_WRITE_RES2 = 4'd9,
               S_NEXT_M     = 4'd10,
               S_DONE       = 4'd11;

    reg [3:0]  state;
    reg [31:0] m_idx;       // current output row index
    reg [31:0] k_idx;       // current inner-dimension index
    reg [31:0] M_reg, K_reg;
    reg [31:0] w_addr_reg, i_addr_reg, o_addr_reg;

    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            state       <= S_IDLE;
            done        <= 0;
            mem_rd_en   <= 0;
            mem_rd_addr <= 0;
            mem_wr_en   <= 0;
            mem_wr_addr <= 0;
            mem_wr_data <= 0;
            mac_start   <= 0;
            mac_load_a  <= 0;
            mac_load_b  <= 0;
            mac_row_a   <= 0;
            mac_row_b   <= 0;
            m_idx       <= 0;
            k_idx       <= 0;
            M_reg       <= 0;
            K_reg       <= 0;
            w_addr_reg  <= 0;
            i_addr_reg  <= 0;
            o_addr_reg  <= 0;
        end else begin
            // Default de-assertions
            mem_rd_en  <= 0;
            mem_wr_en  <= 0;
            mac_start  <= 0;
            mac_load_a <= 0;
            mac_load_b <= 0;
            done       <= 0;

            case (state)
            // -----------------------------------------------------------
            S_IDLE: begin
                if (start) begin
                    M_reg      <= M;
                    K_reg      <= K;
                    w_addr_reg <= weight_addr;
                    i_addr_reg <= input_addr;
                    o_addr_reg <= output_addr;
                    m_idx      <= 0;
                    state      <= S_CLEAR;
                end
            end

            // -----------------------------------------------------------
            S_CLEAR: begin
                // Pulse mac_start to clear accumulators for new output row
                mac_start <= 1;
                k_idx     <= 0;
                state     <= S_LOAD_ACT;
            end

            // -----------------------------------------------------------
            S_LOAD_ACT: begin
                // Read 8 activation bytes: input_addr + k_idx
                // (8 bytes = low 64 bits of 128-bit read)
                mem_rd_en   <= 1;
                mem_rd_addr <= i_addr_reg[26:0] + k_idx[26:0];
                state       <= S_WAIT_ACT;
            end

            // -----------------------------------------------------------
            S_WAIT_ACT: begin
                if (mem_rd_valid) begin
                    mac_row_b <= mem_rd_data[63:0]; // lower 8 bytes
                    state     <= S_LOAD_WGT;
                end
            end

            // -----------------------------------------------------------
            S_LOAD_WGT: begin
                // Read 8 weight bytes: weight_addr + m_idx * K + k_idx
                mem_rd_en   <= 1;
                mem_rd_addr <= w_addr_reg[26:0] + m_idx * K_reg + k_idx;
                state       <= S_WAIT_WGT;
            end

            // -----------------------------------------------------------
            S_WAIT_WGT: begin
                if (mem_rd_valid) begin
                    mac_row_a  <= mem_rd_data[63:0];
                    mac_load_a <= 1;  // latch weights
                    state      <= S_MAC;
                end
            end

            // -----------------------------------------------------------
            S_MAC: begin
                // load_b triggers MAC computation
                mac_load_b <= 1;
                state      <= S_NEXT_K;
            end

            // -----------------------------------------------------------
            S_NEXT_K: begin
                // Wait for mac_done (1 cycle after mac_valid)
                if (mac_done) begin
                    if (k_idx + 8 < K_reg) begin
                        k_idx <= k_idx + 8;
                        state <= S_LOAD_ACT;
                    end else begin
                        state <= S_WRITE_RES;
                    end
                end
            end

            // -----------------------------------------------------------
            S_WRITE_RES: begin
                // Write first 4 INT32 results (128 bits)
                mem_wr_en   <= 1;
                mem_wr_addr <= o_addr_reg[26:0] + (m_idx << 2);
                mem_wr_data <= {mac_result_3, mac_result_2, mac_result_1, mac_result_0};
                state       <= S_WRITE_RES2;
            end

            // -----------------------------------------------------------
            S_WRITE_RES2: begin
                // Write next 4 INT32 results (128 bits)
                mem_wr_en   <= 1;
                mem_wr_addr <= o_addr_reg[26:0] + (m_idx << 2) + 16;
                mem_wr_data <= {mac_result_7, mac_result_6, mac_result_5, mac_result_4};
                state       <= S_NEXT_M;
            end

            // -----------------------------------------------------------
            S_NEXT_M: begin
                if (m_idx + 1 < M_reg) begin
                    m_idx <= m_idx + 1;
                    state <= S_CLEAR;
                end else begin
                    state <= S_DONE;
                end
            end

            // -----------------------------------------------------------
            S_DONE: begin
                done  <= 1;
                state <= S_IDLE;
            end

            default: state <= S_IDLE;
            endcase
        end
    end
endmodule
