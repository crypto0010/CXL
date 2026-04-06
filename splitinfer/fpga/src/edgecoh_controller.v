/* splitinfer/fpga/src/edgecoh_controller.v */
`timescale 1ns / 1ps

module edgecoh_controller (
    input  wire        clk, input wire rst_n,
    input  wire [7:0]  rx_data, input wire rx_valid, output reg rx_ready,
    output reg  [7:0]  tx_data, output reg tx_valid, input wire tx_ready,
    output reg         nmc_start, output reg [7:0] nmc_op,
    output reg  [31:0] nmc_table_base, output reg [31:0] nmc_table_rows,
    output reg  [31:0] nmc_table_cols, output reg [31:0] nmc_input_addr,
    output reg  [31:0] nmc_input_len, output reg [31:0] nmc_output_addr,
    input  wire        nmc_done,
    output reg         dma_wr_en, output reg [31:0] dma_wr_addr, output reg [7:0] dma_wr_data,
    output reg         dma_rd_en, output reg [31:0] dma_rd_addr,
    input  wire [7:0]  dma_rd_data, input wire dma_rd_valid,
    output reg         barrier_ack
);

    localparam MSG_SYNC_BARRIER = 8'h03, MSG_NMC_EXEC = 8'h20,
               MSG_TRANSFER_OWNERSHIP = 8'h01, MSG_ACK = 8'hFE;

    localparam S_IDLE = 4'd0, S_HEADER = 4'd1, S_PAYLOAD = 4'd2,
               S_DISPATCH = 4'd3, S_WAIT_NMC = 4'd4, S_SEND_ACK = 4'd5;

    reg [3:0]  state;
    reg [7:0]  header_buf [0:7];
    reg [2:0]  header_idx;
    reg [7:0]  payload_buf [0:31];
    reg [5:0]  payload_idx;
    reg [31:0] payload_len;
    reg [2:0]  ack_byte_idx;

    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            state <= S_IDLE; rx_ready <= 1; tx_valid <= 0;
            nmc_start <= 0; dma_wr_en <= 0; dma_rd_en <= 0;
            barrier_ack <= 0; header_idx <= 0; payload_idx <= 0; ack_byte_idx <= 0;
        end else begin
            nmc_start <= 0; dma_wr_en <= 0; dma_rd_en <= 0;
            barrier_ack <= 0; tx_valid <= 0;

            case (state)
                S_IDLE: begin
                    rx_ready <= 1; header_idx <= 0;
                    if (rx_valid) begin
                        header_buf[0] <= rx_data; header_idx <= 1; state <= S_HEADER;
                    end
                end
                S_HEADER: begin
                    rx_ready <= 1;
                    if (rx_valid) begin
                        header_buf[header_idx] <= rx_data;
                        if (header_idx == 7) begin
                            payload_len <= {rx_data, header_buf[6], header_buf[5], header_buf[4]};
                            payload_idx <= 0;
                            if ({rx_data, header_buf[6], header_buf[5], header_buf[4]} == 0)
                                state <= S_DISPATCH;
                            else state <= S_PAYLOAD;
                        end else header_idx <= header_idx + 1;
                    end
                end
                S_PAYLOAD: begin
                    rx_ready <= 1;
                    if (rx_valid) begin
                        if (payload_idx < 32) payload_buf[payload_idx] <= rx_data;
                        if (payload_idx + 1 >= payload_len[5:0]) state <= S_DISPATCH;
                        else payload_idx <= payload_idx + 1;
                    end
                end
                S_DISPATCH: begin
                    rx_ready <= 0;
                    case (header_buf[0])
                        MSG_SYNC_BARRIER: begin
                            barrier_ack <= 1; state <= S_SEND_ACK; ack_byte_idx <= 0;
                        end
                        MSG_NMC_EXEC: begin
                            nmc_op <= payload_buf[0];
                            nmc_table_base <= {payload_buf[4], payload_buf[3], payload_buf[2], payload_buf[1]};
                            nmc_table_rows <= {payload_buf[8], payload_buf[7], payload_buf[6], payload_buf[5]};
                            nmc_table_cols <= {payload_buf[12], payload_buf[11], payload_buf[10], payload_buf[9]};
                            nmc_input_addr <= {payload_buf[16], payload_buf[15], payload_buf[14], payload_buf[13]};
                            nmc_input_len  <= {payload_buf[20], payload_buf[19], payload_buf[18], payload_buf[17]};
                            nmc_output_addr <= {payload_buf[24], payload_buf[23], payload_buf[22], payload_buf[21]};
                            nmc_start <= 1; state <= S_WAIT_NMC;
                        end
                        default: begin state <= S_SEND_ACK; ack_byte_idx <= 0; end
                    endcase
                end
                S_WAIT_NMC: if (nmc_done) begin state <= S_SEND_ACK; ack_byte_idx <= 0; end
                S_SEND_ACK: if (tx_ready) begin
                    tx_valid <= 1;
                    case (ack_byte_idx)
                        0: tx_data <= MSG_ACK;
                        1: tx_data <= 8'h00;
                        2: tx_data <= header_buf[2];
                        3: tx_data <= header_buf[3];
                        4: tx_data <= 8'h00;
                        5: tx_data <= 8'h00;
                        6: tx_data <= 8'h00;
                        7: begin tx_data <= 8'h00; state <= S_IDLE; end
                    endcase
                    ack_byte_idx <= ack_byte_idx + 1;
                end
            endcase
        end
    end
endmodule
