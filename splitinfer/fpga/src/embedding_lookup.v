/* splitinfer/fpga/src/embedding_lookup.v */
`timescale 1ns / 1ps

module embedding_lookup (
    input wire clk, input wire rst_n,
    input wire start,
    input wire [31:0] table_base_addr, input wire [31:0] embed_dim,
    input wire [31:0] indices_addr, input wire [31:0] num_indices,
    input wire [31:0] output_addr, output reg done,
    output reg mem_rd_en, output reg [26:0] mem_rd_addr,
    input wire [127:0] mem_rd_data, input wire mem_rd_valid,
    output reg mem_wr_en, output reg [26:0] mem_wr_addr, output reg [127:0] mem_wr_data
);

    localparam S_IDLE=3'd0, S_READ_INDEX=3'd1, S_WAIT_INDEX=3'd2,
               S_READ_EMBED=3'd3, S_WAIT_EMBED=3'd4, S_WRITE_OUT=3'd5, S_DONE=3'd6;

    reg [2:0] state;
    reg [31:0] idx_counter, current_index, burst_counter, bursts_per_embed;

    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin state <= S_IDLE; done <= 0; mem_rd_en <= 0; mem_wr_en <= 0; end
        else begin
            mem_rd_en <= 0; mem_wr_en <= 0; done <= 0;
            case (state)
                S_IDLE: if (start) begin
                    idx_counter <= 0; bursts_per_embed <= embed_dim >> 4; state <= S_READ_INDEX;
                end
                S_READ_INDEX: if (idx_counter >= num_indices) state <= S_DONE;
                else begin
                    mem_rd_en <= 1; mem_rd_addr <= indices_addr[26:0] + (idx_counter << 2);
                    state <= S_WAIT_INDEX;
                end
                S_WAIT_INDEX: if (mem_rd_valid) begin
                    current_index <= mem_rd_data[31:0]; burst_counter <= 0; state <= S_READ_EMBED;
                end
                S_READ_EMBED: if (burst_counter >= bursts_per_embed) begin
                    idx_counter <= idx_counter + 1; state <= S_READ_INDEX;
                end else begin
                    mem_rd_en <= 1;
                    mem_rd_addr <= table_base_addr[26:0] + current_index * embed_dim + (burst_counter << 4);
                    state <= S_WAIT_EMBED;
                end
                S_WAIT_EMBED: if (mem_rd_valid) state <= S_WRITE_OUT;
                S_WRITE_OUT: begin
                    mem_wr_en <= 1;
                    mem_wr_addr <= output_addr[26:0] + idx_counter * embed_dim + (burst_counter << 4);
                    mem_wr_data <= mem_rd_data;
                    burst_counter <= burst_counter + 1; state <= S_READ_EMBED;
                end
                S_DONE: begin done <= 1; state <= S_IDLE; end
            endcase
        end
    end
endmodule
