/* splitinfer/fpga/src/eltwise_controller.v
 *
 * Reads 128-bit words from DDR2, applies elementwise operation (ReLU/add/mul),
 * writes results back. Shares NMC memory port with embedding_lookup via mux.
 * CLOCK DOMAIN: ui_clk (~81.25 MHz).
 */
`timescale 1ns / 1ps

module eltwise_controller (
    input wire clk, input wire rst_n,
    input wire start,
    input wire [31:0] input_addr,    /* source address in DDR2 */
    input wire [31:0] output_addr,   /* destination address in DDR2 */
    input wire [31:0] num_words,     /* number of 128-bit words to process */
    input wire [1:0]  op,            /* 00=ReLU, 01=add, 10=mul */
    input wire [7:0]  scale,         /* scale_factor for add/mul */
    output reg done,

    /* Memory port */
    output reg         mem_rd_en,
    output reg  [26:0] mem_rd_addr,
    input  wire [127:0] mem_rd_data,
    input  wire        mem_rd_valid,
    output reg         mem_wr_en,
    output reg  [26:0] mem_wr_addr,
    output reg  [127:0] mem_wr_data
);

    localparam S_IDLE = 3'd0, S_READ = 3'd1, S_WAIT = 3'd2,
               S_PROC = 3'd3, S_WRITE = 3'd4, S_DONE = 3'd5;

    reg [2:0]   state;
    reg [31:0]  word_idx;
    reg [127:0] rd_buf;
    reg [1:0]   op_reg;
    reg [7:0]   scale_reg;

    /* Elementwise compute (instantiate the existing module) */
    wire [127:0] elt_out;
    wire         elt_done;
    reg          elt_start;

    elementwise u_elt (
        .clk(clk), .rst_n(rst_n), .start(elt_start),
        .op(op_reg), .data_in(rd_buf), .scale_factor(scale_reg),
        .data_out(elt_out), .done(elt_done)
    );

    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            state <= S_IDLE; done <= 0;
            mem_rd_en <= 0; mem_wr_en <= 0; elt_start <= 0;
        end else begin
            mem_rd_en <= 0; mem_wr_en <= 0; done <= 0; elt_start <= 0;
            case (state)
                S_IDLE: if (start) begin
                    word_idx <= 0; op_reg <= op; scale_reg <= scale;
                    state <= S_READ;
                end
                S_READ: if (word_idx >= num_words) state <= S_DONE;
                else begin
                    mem_rd_en <= 1;
                    mem_rd_addr <= input_addr[26:0] + (word_idx << 4);
                    state <= S_WAIT;
                end
                S_WAIT: if (mem_rd_valid) begin
                    rd_buf <= mem_rd_data;
                    elt_start <= 1;
                    state <= S_PROC;
                end
                S_PROC: if (elt_done) begin
                    state <= S_WRITE;
                end
                S_WRITE: begin
                    mem_wr_en <= 1;
                    mem_wr_addr <= output_addr[26:0] + (word_idx << 4);
                    mem_wr_data <= elt_out;
                    word_idx <= word_idx + 1;
                    state <= S_READ;
                end
                S_DONE: begin done <= 1; state <= S_IDLE; end
            endcase
        end
    end
endmodule
