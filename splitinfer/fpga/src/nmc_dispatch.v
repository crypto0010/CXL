/* splitinfer/fpga/src/nmc_dispatch.v */
`timescale 1ns / 1ps

module nmc_dispatch (
    input wire clk, input wire rst_n,
    input wire nmc_start, input wire [7:0] nmc_op,
    input wire [31:0] nmc_table_base, nmc_table_rows, nmc_table_cols,
    input wire [31:0] nmc_input_addr, nmc_input_len, nmc_output_addr,
    output wire nmc_done,
    output reg emb_start, output reg [31:0] emb_table_base, emb_embed_dim,
    output reg [31:0] emb_indices_addr, emb_num_indices, emb_output_addr,
    input wire emb_done,
    output reg mac_start,
    output reg [31:0] mac_weight_addr, mac_input_addr, mac_output_addr,
    output reg [31:0] mac_M, mac_K,
    input wire mac_done,
    output reg elt_start,
    output reg [31:0] elt_input_addr, elt_addr2, elt_output_addr, elt_num_words,
    output reg [2:0]  elt_op,
    output reg [7:0]  elt_scale,
    output reg [15:0] elt_mult,
    input wire elt_done
);

    localparam NMC_EMBEDDING = 8'h01, NMC_INT8_FC = 8'h02, NMC_ELEMENTWISE = 8'h03;

    reg [7:0] active_op; reg busy;

    assign nmc_done = (!busy) ? 1'b0 :
                      (active_op == NMC_EMBEDDING) ? emb_done :
                      (active_op == NMC_INT8_FC) ? mac_done :
                      (active_op == NMC_ELEMENTWISE) ? elt_done : 1'b0;

    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin emb_start <= 0; mac_start <= 0; elt_start <= 0; busy <= 0; end
        else begin
            emb_start <= 0; mac_start <= 0; elt_start <= 0;
            if (nmc_start && !busy) begin
                active_op <= nmc_op; busy <= 1;
                case (nmc_op)
                    NMC_EMBEDDING: begin
                        emb_start <= 1; emb_table_base <= nmc_table_base;
                        emb_embed_dim <= nmc_table_cols; emb_indices_addr <= nmc_input_addr;
                        emb_num_indices <= nmc_input_len; emb_output_addr <= nmc_output_addr;
                    end
                    NMC_INT8_FC: begin
                        mac_start <= 1;
                        mac_weight_addr <= nmc_table_base;
                        mac_M <= nmc_table_rows;
                        mac_K <= nmc_table_cols;
                        mac_input_addr <= nmc_input_addr;
                        mac_output_addr <= nmc_output_addr;
                    end
                    NMC_ELEMENTWISE: begin
                        // Field mapping (see messages.h edgecoh_nmc_exec_msg_t):
                        //   table_base[2:0]  op      table_base[15:8] scale/shift/relu
                        //   table_rows       num 16-byte input words
                        //   table_cols       second-operand address (ops 3,4)
                        //   input_len[15:0]  requant multiplier (op 4)
                        elt_start <= 1;
                        elt_input_addr <= nmc_input_addr;
                        elt_addr2 <= nmc_table_cols;
                        elt_output_addr <= nmc_output_addr;
                        elt_num_words <= nmc_table_rows;
                        elt_op <= nmc_table_base[2:0];
                        elt_scale <= nmc_table_base[15:8];
                        elt_mult <= nmc_input_len[15:0];
                    end
                    default: busy <= 0;
                endcase
            end
            if (busy && nmc_done) busy <= 0;
        end
    end
endmodule
