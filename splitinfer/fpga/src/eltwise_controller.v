/* splitinfer/fpga/src/eltwise_controller.v  (v2)
 *
 * Streams 128-bit words through the elementwise unit.  All addresses are
 * BYTE addresses, 16-byte aligned.  ops 0-2: one input stream.  op 3: two
 * input streams (input_addr, addr2).  op 4 (EPILOGUE): acc stream at
 * input_addr and bias stream at addr2, each 16 B/word (4 x INT32); every
 * four input words produce one 16-byte INT8 output word.  A partial final
 * group is written zero-padded — the host pads M to a multiple of 16.
 */
`timescale 1ns / 1ps

module eltwise_controller (
    input  wire         clk, input wire rst_n,
    input  wire         start,
    input  wire [31:0]  input_addr,
    input  wire [31:0]  addr2,
    input  wire [31:0]  output_addr,
    input  wire [31:0]  num_words,     /* 16-byte input words */
    input  wire [2:0]   op,
    input  wire [7:0]   scale,
    input  wire [15:0]  mult,
    output reg          done,
    output reg          mem_rd_en,
    output reg  [26:0]  mem_rd_addr,
    input  wire [127:0] mem_rd_data,
    input  wire         mem_rd_valid,
    output reg          mem_wr_en,
    output reg  [26:0]  mem_wr_addr,
    output reg  [127:0] mem_wr_data
);
    localparam S_IDLE=3'd0, S_READ_A=3'd1, S_WAIT_A=3'd2, S_READ_B=3'd3, S_WAIT_B=3'd4,
               S_PROC=3'd5, S_WRITE=3'd6, S_DONE=3'd7;
    reg [2:0]   state;
    reg [31:0]  word_idx, out_idx;
    reg [127:0] buf_a, buf_b, out_buf;
    reg [2:0]   op_reg; reg [7:0] scale_reg; reg [15:0] mult_reg;
    reg         elt_start;
    wire [127:0] elt_out; wire elt_done;
    wire two_operand = (op_reg == 3'd3) || (op_reg == 3'd4);
    wire epilogue    = (op_reg == 3'd4);

    elementwise u_elt (.clk(clk), .rst_n(rst_n), .start(elt_start), .op(op_reg),
        .data_in(buf_a), .data_b(buf_b), .scale_factor(scale_reg), .mult(mult_reg),
        .data_out(elt_out), .done(elt_done));

    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            state <= S_IDLE; done <= 0; mem_rd_en <= 0; mem_wr_en <= 0; elt_start <= 0;
            word_idx <= 0; out_idx <= 0; buf_a <= 0; buf_b <= 0; out_buf <= 0;
            op_reg <= 0; scale_reg <= 0; mult_reg <= 0; mem_rd_addr <= 0; mem_wr_addr <= 0; mem_wr_data <= 0;
        end else begin
            mem_rd_en <= 0; mem_wr_en <= 0; done <= 0; elt_start <= 0;
            case (state)
                S_IDLE: if (start) begin
                    word_idx <= 0; out_idx <= 0; out_buf <= 0;
                    op_reg <= op; scale_reg <= scale; mult_reg <= mult;
                    state <= S_READ_A;
                end
                S_READ_A: if (word_idx >= num_words) begin
                    // flush a partial epilogue group
                    if (epilogue && (word_idx[1:0] != 2'd0)) state <= S_WRITE; else state <= S_DONE;
                end else begin
                    mem_rd_en <= 1; mem_rd_addr <= input_addr[26:0] + (word_idx[22:0] << 4);
                    state <= S_WAIT_A;
                end
                S_WAIT_A: if (mem_rd_valid) begin
                    buf_a <= mem_rd_data;
                    state <= two_operand ? S_READ_B : S_PROC;
                    if (!two_operand) elt_start <= 1;
                end
                S_READ_B: begin
                    mem_rd_en <= 1; mem_rd_addr <= addr2[26:0] + (word_idx[22:0] << 4);
                    state <= S_WAIT_B;
                end
                S_WAIT_B: if (mem_rd_valid) begin buf_b <= mem_rd_data; elt_start <= 1; state <= S_PROC; end
                S_PROC: if (elt_done) begin
                    if (epilogue) begin
                        out_buf[word_idx[1:0]*32 +: 32] <= elt_out[31:0];
                        word_idx <= word_idx + 1;
                        if (word_idx[1:0] == 2'd3) state <= S_WRITE; else state <= S_READ_A;
                    end else begin
                        out_buf <= elt_out; word_idx <= word_idx + 1; state <= S_WRITE;
                    end
                end
                S_WRITE: begin
                    mem_wr_en <= 1;
                    mem_wr_addr <= output_addr[26:0] + (out_idx[22:0] << 4);
                    mem_wr_data <= out_buf;
                    out_idx <= out_idx + 1; out_buf <= 0;
                    // after a flushed partial group, or a normal word, continue/finish
                    if (epilogue && word_idx >= num_words) state <= S_DONE; else state <= S_READ_A;
                end
                S_DONE: begin done <= 1; state <= S_IDLE; end
            endcase
        end
    end
endmodule
