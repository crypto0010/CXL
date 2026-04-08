/* splitinfer/fpga/src/edgecoh_controller.v
 *
 * EdgeCoh protocol FSM — receives messages from USB/UART byte stream,
 * dispatches NMC commands and DMA transfers, sends ACK/DATA_RESPONSE.
 *
 * Supported message types:
 *   0x01 TRANSFER_OWNERSHIP — ACK only (ownership tracked in software)
 *   0x03 SYNC_BARRIER       — pulse barrier_ack, send ACK
 *   0x10 DATA_WRITE          — stream payload_len bytes to DDR2 via DMA
 *   0x11 DATA_READ           — read read_len bytes from DDR2, send DATA_RESPONSE
 *   0x20 NMC_EXEC            — dispatch NMC operation, wait for done, send ACK
 */
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

    localparam MSG_TRANSFER_OWNERSHIP = 8'h01;
    localparam MSG_SYNC_BARRIER       = 8'h03;
    localparam MSG_DATA_WRITE         = 8'h10;
    localparam MSG_DATA_READ          = 8'h11;
    localparam MSG_DATA_RESPONSE      = 8'h12;
    localparam MSG_NMC_EXEC           = 8'h20;
    localparam MSG_ACK                = 8'hFE;

    localparam S_IDLE       = 4'd0;
    localparam S_HEADER     = 4'd1;
    localparam S_PAYLOAD    = 4'd2;
    localparam S_DISPATCH   = 4'd3;
    localparam S_WAIT_NMC   = 4'd4;
    localparam S_SEND_ACK   = 4'd5;
    localparam S_DMA_WRITE  = 4'd6;   // stream DATA_WRITE payload to DDR2
    localparam S_DMA_READ_REQ = 4'd7; // issue DMA read requests
    localparam S_DMA_READ_RESP = 4'd8; // send DATA_RESPONSE header + payload
    localparam S_SEND_RESP_HDR = 4'd9; // send DATA_RESPONSE 8-byte header
    localparam S_ACK_WAIT   = 4'd10;  // wait for TX to return to idle between ACK bytes

    reg [3:0]  state;
    reg [7:0]  header_buf [0:7];
    reg [2:0]  header_idx;
    reg [7:0]  payload_buf [0:31];
    reg [5:0]  payload_idx;
    reg [31:0] payload_len;
    reg [2:0]  ack_byte_idx;

    // DMA write state
    reg [31:0] dma_wr_base;      // base DDR2 address for DATA_WRITE
    reg [31:0] dma_wr_remaining; // bytes remaining to write

    // DMA read state
    reg [31:0] dma_rd_base;      // base DDR2 address for DATA_READ
    reg [31:0] dma_rd_total;     // total bytes to read
    reg [31:0] dma_rd_requested; // bytes requested so far
    reg [31:0] dma_rd_sent;      // bytes sent back to host so far
    reg [2:0]  resp_hdr_idx;     // byte index within DATA_RESPONSE header

    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            state <= S_IDLE; rx_ready <= 1; tx_valid <= 0;
            nmc_start <= 0; dma_wr_en <= 0; dma_rd_en <= 0;
            barrier_ack <= 0; header_idx <= 0; payload_idx <= 0; ack_byte_idx <= 0;
            dma_wr_remaining <= 0; dma_rd_total <= 0;
            dma_rd_requested <= 0; dma_rd_sent <= 0; resp_hdr_idx <= 0;
        end else begin
            nmc_start <= 0; dma_wr_en <= 0; dma_rd_en <= 0;
            barrier_ack <= 0; tx_valid <= 0;

            case (state)
                // ── Idle: wait for first header byte ──────────────────────
                S_IDLE: begin
                    rx_ready <= 1; header_idx <= 0;
                    if (rx_valid) begin
                        header_buf[0] <= rx_data; header_idx <= 1; state <= S_HEADER;
                    end
                end

                // ── Collect 8-byte header ────────────────────────────────
                S_HEADER: begin
                    rx_ready <= 1;
                    if (rx_valid) begin
                        header_buf[header_idx] <= rx_data;
                        if (header_idx == 7) begin
                            payload_len <= {rx_data, header_buf[6], header_buf[5], header_buf[4]};
                            payload_idx <= 0;
                            if ({rx_data, header_buf[6], header_buf[5], header_buf[4]} == 0)
                                state <= S_DISPATCH;
                            else
                                state <= S_PAYLOAD;
                        end else
                            header_idx <= header_idx + 1;
                    end
                end

                // ── Collect payload into buffer (up to 32 bytes) ─────────
                // For small payloads (NMC_EXEC ≤25B, DATA_READ=8B): collect all.
                // For DATA_WRITE: collect only the ddr2_addr (4 bytes), then
                // dispatch; the tensor data streams in S_DMA_WRITE.
                S_PAYLOAD: begin
                    rx_ready <= 1;
                    if (rx_valid) begin
                        if (payload_idx < 32) payload_buf[payload_idx] <= rx_data;
                        // Determine when to exit S_PAYLOAD:
                        // - DATA_WRITE (0x10): only need 4 bytes (ddr2_addr)
                        // - Others: collect min(payload_len, 32) bytes
                        if (header_buf[0] == MSG_DATA_WRITE) begin
                            if (payload_idx + 1 >= 4)
                                state <= S_DISPATCH;
                            else
                                payload_idx <= payload_idx + 1;
                        end else begin
                            // Cap collection at 32 bytes to avoid buffer overflow
                            if (payload_idx + 1 >= payload_len[4:0] || payload_idx >= 31)
                                state <= S_DISPATCH;
                            else
                                payload_idx <= payload_idx + 1;
                        end
                    end
                end

                // ── Dispatch based on message type ───────────────────────
                S_DISPATCH: begin
                    rx_ready <= 0;
                    case (header_buf[0])
                        MSG_SYNC_BARRIER: begin
                            barrier_ack <= 1;
                            state <= S_SEND_ACK; ack_byte_idx <= 0;
                        end

                        MSG_NMC_EXEC: begin
                            nmc_op         <= payload_buf[0];
                            nmc_table_base <= {payload_buf[4],  payload_buf[3],  payload_buf[2],  payload_buf[1]};
                            nmc_table_rows <= {payload_buf[8],  payload_buf[7],  payload_buf[6],  payload_buf[5]};
                            nmc_table_cols <= {payload_buf[12], payload_buf[11], payload_buf[10], payload_buf[9]};
                            nmc_input_addr <= {payload_buf[16], payload_buf[15], payload_buf[14], payload_buf[13]};
                            nmc_input_len  <= {payload_buf[20], payload_buf[19], payload_buf[18], payload_buf[17]};
                            nmc_output_addr<= {payload_buf[24], payload_buf[23], payload_buf[22], payload_buf[21]};
                            nmc_start <= 1;
                            state <= S_WAIT_NMC;
                        end

                        MSG_DATA_WRITE: begin
                            // payload_buf[0:3] = ddr2_addr (4 bytes LE, consumed in S_PAYLOAD).
                            // Remaining tensor data bytes = payload_len - 4.
                            // These stream inline from the UART in S_DMA_WRITE.
                            dma_wr_base <= {payload_buf[3], payload_buf[2], payload_buf[1], payload_buf[0]};
                            dma_wr_remaining <= (payload_len > 4) ? (payload_len - 4) : 0;
                            if (payload_len > 4) begin
                                state <= S_DMA_WRITE;
                                rx_ready <= 1;
                            end else begin
                                // No data bytes — just ACK
                                state <= S_SEND_ACK; ack_byte_idx <= 0;
                            end
                        end

                        MSG_DATA_READ: begin
                            // payload_buf: ddr2_addr (4B LE) + read_len (4B LE)
                            dma_rd_base      <= {payload_buf[3], payload_buf[2], payload_buf[1], payload_buf[0]};
                            dma_rd_total     <= {payload_buf[7], payload_buf[6], payload_buf[5], payload_buf[4]};
                            dma_rd_requested <= 0;
                            dma_rd_sent      <= 0;
                            resp_hdr_idx     <= 0;
                            state <= S_SEND_RESP_HDR;
                        end

                        default: begin
                            // TRANSFER_OWNERSHIP or unknown — just ACK
                            state <= S_SEND_ACK; ack_byte_idx <= 0;
                        end
                    endcase
                end

                // ── Wait for NMC completion ──────────────────────────────
                S_WAIT_NMC: begin
                    if (nmc_done) begin
                        state <= S_SEND_ACK; ack_byte_idx <= 0;
                    end
                end

                // ── Send 8-byte ACK ──────────────────────────────────────
                // Proper valid/ready handshake: assert tx_valid with tx_data,
                // wait until we observe tx_ready=0 (meaning usb_interface has
                // latched the byte and moved to TX_START).  At that point
                // de-assert tx_valid, wait for tx_ready=1 again (TX_IDLE),
                // then advance to the next byte.  This avoids the 1-cycle
                // pulse race that was causing bytes to be dropped.
                S_SEND_ACK: begin
                    case (ack_byte_idx)
                        0: tx_data <= MSG_ACK;
                        1: tx_data <= 8'h00;
                        2: tx_data <= header_buf[2]; // tensor_id lo
                        3: tx_data <= header_buf[3]; // tensor_id hi
                        4: tx_data <= 8'h00;         // payload_len = 0
                        5: tx_data <= 8'h00;
                        6: tx_data <= 8'h00;
                        7: tx_data <= 8'h00;
                    endcase
                    tx_valid <= 1;                 // hold high (override default)
                    if (tx_valid && !tx_ready) begin
                        // usb_interface has latched the byte and is now busy.
                        // De-assert tx_valid and wait for TX_IDLE again.
                        tx_valid <= 0;
                        state    <= S_ACK_WAIT;
                    end
                end

                // Wait for usb_interface TX to return to idle, then advance
                // to the next ACK byte (or finish).
                S_ACK_WAIT: begin
                    if (tx_ready) begin
                        if (ack_byte_idx == 7) begin
                            ack_byte_idx <= 0;
                            state <= S_IDLE;
                        end else begin
                            ack_byte_idx <= ack_byte_idx + 1;
                            state <= S_SEND_ACK;
                        end
                    end
                end

                // ── Stream DATA_WRITE payload bytes to DDR2 ──────────────
                S_DMA_WRITE: begin
                    rx_ready <= 1;
                    if (rx_valid && dma_wr_remaining > 0) begin
                        dma_wr_en   <= 1;
                        dma_wr_addr <= dma_wr_base;
                        dma_wr_data <= rx_data;
                        dma_wr_base <= dma_wr_base + 1;
                        dma_wr_remaining <= dma_wr_remaining - 1;
                        if (dma_wr_remaining == 1) begin
                            // Last byte written — send ACK
                            rx_ready <= 0;
                            state <= S_SEND_ACK; ack_byte_idx <= 0;
                        end
                    end
                end

                // ── Send DATA_RESPONSE header (8 bytes) ──────────────────
                S_SEND_RESP_HDR: begin
                    if (tx_ready) begin
                        tx_valid <= 1;
                        case (resp_hdr_idx)
                            0: tx_data <= MSG_DATA_RESPONSE;
                            1: tx_data <= 8'h00;             // flags
                            2: tx_data <= header_buf[2];      // tensor_id lo
                            3: tx_data <= header_buf[3];      // tensor_id hi
                            4: tx_data <= dma_rd_total[7:0];  // payload_len LE
                            5: tx_data <= dma_rd_total[15:8];
                            6: tx_data <= dma_rd_total[23:16];
                            7: begin
                                tx_data <= dma_rd_total[31:24];
                                state <= S_DMA_READ_REQ;
                            end
                        endcase
                        resp_hdr_idx <= resp_hdr_idx + 1;
                    end
                end

                // ── Issue DMA read requests and send data back ───────────
                // Simplified: issue one read at a time, wait for valid,
                // send byte over TX, repeat until all bytes sent.
                S_DMA_READ_REQ: begin
                    if (dma_rd_requested < dma_rd_total) begin
                        dma_rd_en   <= 1;
                        dma_rd_addr <= dma_rd_base + dma_rd_requested;
                        dma_rd_requested <= dma_rd_requested + 1;
                        state <= S_DMA_READ_RESP;
                    end else begin
                        // All bytes read and sent — done
                        state <= S_IDLE;
                    end
                end

                S_DMA_READ_RESP: begin
                    if (dma_rd_valid) begin
                        if (tx_ready) begin
                            tx_valid <= 1;
                            tx_data  <= dma_rd_data;
                            dma_rd_sent <= dma_rd_sent + 1;
                            // Issue next read or finish
                            if (dma_rd_sent + 1 >= dma_rd_total)
                                state <= S_IDLE;
                            else
                                state <= S_DMA_READ_REQ;
                        end
                        // else: wait for tx_ready
                    end
                end

            endcase
        end
    end
endmodule
