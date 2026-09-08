/* Behavioural model of the MIG 7-series user interface (DDR2, 16-bit, BL8).
 * Backed by a byte array.  Honors app_wdf_mask.  Injects random app_rdy /
 * app_wdf_rdy stalls and a configurable read latency so the arbiter's
 * hold-until-accepted handshake is actually exercised.
 * Address unit: 16-bit words; a burst = 8 words = 16 bytes.  */
`timescale 1ns / 1ps
module ddr2_ui_model #(parameter BYTES = 1 << 20, parameter RD_LAT = 12, parameter STALL_PCT = 25) (
    input  wire         clk, input wire rst_n,
    input  wire [26:0]  app_addr, input wire [2:0] app_cmd, input wire app_en,
    input  wire [127:0] app_wdf_data, input wire [15:0] app_wdf_mask,
    input  wire         app_wdf_wren, input wire app_wdf_end,
    output reg  [127:0] app_rd_data, output reg app_rd_data_valid,
    output reg          app_rdy, output reg app_wdf_rdy
);
    reg [7:0] mem [0:BYTES-1];
    // pending read pipeline
    reg [26:0] rd_q_addr [0:63]; reg [7:0] rd_q_t [0:63]; reg rd_q_v [0:63];
    integer i, b;
    reg [127:0] wdata_q; reg [15:0] wmask_q; reg wdata_pending;
    reg [26:0] waddr_q; reg wcmd_pending;

    task do_write; input [26:0] a; input [127:0] d; input [15:0] m; integer k; begin
        for (k = 0; k < 16; k = k + 1) if (!m[k]) mem[a*2 + k] = d[k*8 +: 8];
    end endtask

    initial begin
        for (i = 0; i < BYTES; i = i + 1) mem[i] = 8'h00;
        for (i = 0; i < 64; i = i + 1) rd_q_v[i] = 0;
    end

    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            app_rdy <= 1; app_wdf_rdy <= 1; app_rd_data_valid <= 0; app_rd_data <= 0;
            wdata_pending <= 0; wcmd_pending <= 0;
        end else begin
            app_rdy     <= ($urandom % 100) >= STALL_PCT;
            app_wdf_rdy <= ($urandom % 100) >= STALL_PCT;
            app_rd_data_valid <= 0;
            app_rd_data <= {4{$urandom}};       // NOT held after the valid cycle (matches the real MIG)
            // age read queue
            for (i = 0; i < 64; i = i + 1) if (rd_q_v[i]) begin
                if (rd_q_t[i] == 0) begin
                    rd_q_v[i] <= 0; app_rd_data_valid <= 1;
                    for (b = 0; b < 16; b = b + 1) app_rd_data[b*8 +: 8] <= mem[rd_q_addr[i]*2 + b];
                end else rd_q_t[i] <= rd_q_t[i] - 1;
            end
            // accept command / data (either order, or the same cycle)
            begin : wr_logic
                reg cmd_now, dat_now;
                cmd_now = app_en && app_rdy && (app_cmd != 3'b001);
                dat_now = app_wdf_wren && app_wdf_rdy;
                if (app_en && app_rdy && app_addr[2:0] != 0) $display("MODEL ERROR: unaligned app_addr %h", app_addr);
                if (app_en && app_rdy && app_cmd == 3'b001) begin : find
                    integer s; s = -1;
                    for (i = 63; i >= 0; i = i - 1) if (!rd_q_v[i]) s = i;
                    rd_q_v[s] <= 1; rd_q_addr[s] <= app_addr; rd_q_t[s] <= RD_LAT + ($urandom % 4);
                end
                if (cmd_now && dat_now) do_write(app_addr, app_wdf_data, app_wdf_mask);
                else if (cmd_now && wdata_pending) begin do_write(app_addr, wdata_q, wmask_q); wdata_pending <= 0; end
                else if (cmd_now) begin wcmd_pending <= 1; waddr_q <= app_addr; end
                else if (dat_now && wcmd_pending) begin do_write(waddr_q, app_wdf_data, app_wdf_mask); wcmd_pending <= 0; end
                else if (dat_now) begin wdata_pending <= 1; wdata_q <= app_wdf_data; wmask_q <= app_wdf_mask; end
            end
        end
    end
endmodule
