/* splitinfer/fpga/src/ddr2_arbiter.v  (v2)
 *
 * Arbitrates DDR2 access between the NMC engines, the MAC controller and the
 * EdgeCoh DMA byte stream, and owns the MIG user-interface handshake.
 *
 * CLOCK DOMAIN: MIG ui_clk (~81.25 MHz).
 *
 * Address convention (v2): EVERY port presents a 27-bit BYTE address.
 *   The MIG UI address is in DDR2 16-bit-word units and a BL8 burst covers
 *   16 bytes, so  app_addr = {byte_addr[26:4], 3'b000}.  The v1 design passed
 *   byte addresses straight through, which placed every access at 2x the
 *   intended offset and mis-aligned every burst.
 *
 * DMA byte bridge (v2): the EdgeCoh controller streams single bytes.  v1
 *   turned each byte into a full 128-bit write with the mask disabled,
 *   zeroing its 15 neighbours, and served every byte read from lane 0 of the
 *   burst.  v2 stages bytes into a 16-byte word and writes it with a byte
 *   mask (coalescing at high link rates, degrading gracefully to masked
 *   single-byte writes at UART rates), and serves reads from a one-word
 *   cache with correct lane selection.  Reads flush any staged write first,
 *   so read-after-write ordering holds.
 *
 * MIG handshake (v2): app_en is held until app_rdy is sampled high with it;
 *   app_wdf_wren is held until app_wdf_rdy.  v1 asserted each for a single
 *   cycle based on the *previous* cycle's ready, which drops commands when
 *   the controller back-pressures.
 *
 * Reads: one outstanding read at a time, tagged with its owner, so data can
 *   never be attributed to the wrong port.
 */
`timescale 1ns / 1ps

module ddr2_arbiter (
    input  wire         clk,
    input  wire         rst_n,
    // MIG user interface
    output reg  [26:0]  app_addr,
    output reg  [2:0]   app_cmd,
    output reg          app_en,
    output reg  [127:0] app_wdf_data,
    output reg  [15:0]  app_wdf_mask,     // 1 = do NOT write this byte
    output reg          app_wdf_wren,
    output reg          app_wdf_end,
    input  wire [127:0] app_rd_data,
    input  wire         app_rd_data_valid,
    input  wire         app_rdy,
    input  wire         app_wdf_rdy,
    // NMC port (embedding / elementwise) — 16-byte aligned byte addresses
    input  wire         nmc_rd_en,
    input  wire [26:0]  nmc_rd_addr,
    output wire [127:0] nmc_rd_data,
    output wire         nmc_rd_valid,
    input  wire         nmc_wr_en,
    input  wire [26:0]  nmc_wr_addr,
    input  wire [127:0] nmc_wr_data,
    // MAC port — 16-byte aligned byte addresses
    input  wire         mac_rd_en,
    input  wire [26:0]  mac_rd_addr,
    output wire [127:0] mac_rd_data,
    output wire         mac_rd_valid,
    input  wire         mac_wr_en,
    input  wire [26:0]  mac_wr_addr,
    input  wire [127:0] mac_wr_data,
    // DMA byte stream — valid/ready on both directions
    input  wire         dma_wr_en,        // valid
    input  wire [26:0]  dma_wr_addr,
    input  wire [7:0]   dma_wr_byte,
    output wire         dma_wr_ready,
    input  wire         dma_rd_en,        // valid
    input  wire [26:0]  dma_rd_addr,
    output wire         dma_rd_ready,
    output reg  [7:0]   dma_rd_byte,
    output reg          dma_rd_valid
);

    localparam CMD_WRITE = 3'b000, CMD_READ = 3'b001;
    localparam OWN_NMC = 2'd0, OWN_MAC = 2'd1, OWN_DMA = 2'd2;
    localparam WA_IDLE_LIMIT = 6'd31;

    function [26:0] to_mig; input [26:0] b; begin to_mig = {b[26:4], 3'b000}; end endfunction

    // ── Read return routing (single outstanding read) ────────────────────
    reg        rd_busy;
    reg [1:0]  rd_owner;
    assign nmc_rd_data  = app_rd_data;
    assign mac_rd_data  = app_rd_data;
    assign nmc_rd_valid = app_rd_data_valid && rd_busy && (rd_owner == OWN_NMC);
    assign mac_rd_valid = app_rd_data_valid && rd_busy && (rd_owner == OWN_MAC);
    wire   dma_rd_fill  = app_rd_data_valid && rd_busy && (rd_owner == OWN_DMA);

    // ── Latched engine requests ──────────────────────────────────────────
    // Reads: one pending per port (engines wait for data before the next).
    // Writes: TWO-deep queue per port.  Engines issue write pairs on
    // consecutive cycles (e.g. mac_controller S_WR0/S_WR1); a single latch
    // that is cleared on issue drops the second of the pair whenever the
    // first issues on the cycle the second arrives — v1 had exactly that.
    reg         nmc_rd_req, mac_rd_req;
    reg [26:0]  nmc_rd_req_addr, mac_rd_req_addr;
    reg [1:0]   nmc_wq_cnt, mac_wq_cnt;               // 0..2 entries
    reg [26:0]  nmc_wq_addr0, nmc_wq_addr1, mac_wq_addr0, mac_wq_addr1;
    reg [127:0] nmc_wq_data0, nmc_wq_data1, mac_wq_data0, mac_wq_data1;
    wire        nmc_wr_req = (nmc_wq_cnt != 0);
    wire        mac_wr_req = (mac_wq_cnt != 0);
    wire [26:0] nmc_wr_req_addr = nmc_wq_addr0;  wire [127:0] nmc_wr_req_data = nmc_wq_data0;
    wire [26:0] mac_wr_req_addr = mac_wq_addr0;  wire [127:0] mac_wr_req_data = mac_wq_data0;
    reg         nmc_wq_pop, mac_wq_pop;             // set by the issue logic this cycle

    // ── DMA write assembler ──────────────────────────────────────────────
    reg         wa_valid;
    reg [26:0]  wa_word;
    reg [127:0] wa_data;
    reg [15:0]  wa_lanes;          // 1 = byte present
    reg [5:0]   wa_idle;
    reg         wa_flush_req;      // staged word must be written out
    wire        wa_same   = (dma_wr_addr[26:4] == wa_word[26:4]);
    assign      dma_wr_ready = !wa_flush_req && (!wa_valid || wa_same);
    wire        wa_accept = dma_wr_en && dma_wr_ready;

    // ── DMA read cache ───────────────────────────────────────────────────
    reg         rc_valid;
    reg [26:0]  rc_word;
    reg [127:0] rc_data;
    reg         dr_busy;           // DMA read miss in flight
    reg         dr_req;            // miss waiting to be issued
    reg [26:0]  dr_addr;
    reg [3:0]   dr_off;
    wire        rc_hit    = rc_valid && (dma_rd_addr[26:4] == rc_word[26:4]);
    // A read may be accepted only when no DMA read is in flight and nothing
    // is staged for write (read-after-write ordering).
    assign      dma_rd_ready = !dr_busy && !wa_valid && !wa_flush_req;
    wire        dr_accept = dma_rd_en && dma_rd_ready;

    // Which write queue pops this cycle (mirrors the issue priority chain).
    wire nmc_wq_pop_now = !issuing && nmc_wr_req;
    wire mac_wq_pop_now = !issuing && !nmc_wr_req && !(nmc_rd_req && !rd_busy) && mac_wr_req;

    // ── MIG issue state ──────────────────────────────────────────────────
    reg issuing;      // command / data being presented
    wire cmd_accepted = app_en && app_rdy;
    wire wdf_accepted = app_wdf_wren && app_wdf_rdy;
    wire issue_free   = !issuing;

    integer i;

    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            app_en <= 0; app_wdf_wren <= 0; app_wdf_end <= 0; app_wdf_mask <= 16'hFFFF;
            app_addr <= 0; app_cmd <= CMD_READ; app_wdf_data <= 0;
            rd_busy <= 0; rd_owner <= OWN_NMC;
            nmc_rd_req <= 0; mac_rd_req <= 0;
            nmc_wq_cnt <= 0; mac_wq_cnt <= 0; nmc_wq_pop <= 0; mac_wq_pop <= 0;
            nmc_wq_addr0 <= 0; nmc_wq_addr1 <= 0; mac_wq_addr0 <= 0; mac_wq_addr1 <= 0;
            nmc_wq_data0 <= 0; nmc_wq_data1 <= 0; mac_wq_data0 <= 0; mac_wq_data1 <= 0;
            wa_valid <= 0; wa_word <= 0; wa_data <= 0; wa_lanes <= 0; wa_idle <= 0; wa_flush_req <= 0;
            rc_valid <= 0; rc_word <= 0; rc_data <= 0;
            dr_busy <= 0; dr_req <= 0; dr_addr <= 0; dr_off <= 0;
            dma_rd_byte <= 0; dma_rd_valid <= 0;
            issuing <= 0;
        end else begin
            dma_rd_valid <= 0;

            // ── MIG handshake completion ──
            if (cmd_accepted) app_en <= 0;
            if (wdf_accepted) begin app_wdf_wren <= 0; app_wdf_end <= 0; end
            if (issuing && !(app_en && !cmd_accepted) && !(app_wdf_wren && !wdf_accepted))
                issuing <= 0;
            if (app_rd_data_valid) rd_busy <= 0;

            // ── Latch engine pulses ──
            if (nmc_rd_en) begin nmc_rd_req <= 1; nmc_rd_req_addr <= nmc_rd_addr; end
            if (mac_rd_en) begin mac_rd_req <= 1; mac_rd_req_addr <= mac_rd_addr; end
            // Write queues: pop (issue) and push (new pulse) may coincide.
            nmc_wq_pop <= 0; mac_wq_pop <= 0;
            case ({nmc_wr_en, nmc_wq_pop_now})
                2'b10: begin if (nmc_wq_cnt == 0) begin nmc_wq_addr0 <= nmc_wr_addr; nmc_wq_data0 <= nmc_wr_data; end
                             else begin nmc_wq_addr1 <= nmc_wr_addr; nmc_wq_data1 <= nmc_wr_data; end
                             nmc_wq_cnt <= nmc_wq_cnt + 1; end
                2'b01: begin nmc_wq_addr0 <= nmc_wq_addr1; nmc_wq_data0 <= nmc_wq_data1; nmc_wq_cnt <= nmc_wq_cnt - 1; end
                2'b11: begin if (nmc_wq_cnt == 1) begin nmc_wq_addr0 <= nmc_wr_addr; nmc_wq_data0 <= nmc_wr_data; end
                             else begin nmc_wq_addr0 <= nmc_wq_addr1; nmc_wq_data0 <= nmc_wq_data1;
                                        nmc_wq_addr1 <= nmc_wr_addr; nmc_wq_data1 <= nmc_wr_data; end end
                default: ;
            endcase
            case ({mac_wr_en, mac_wq_pop_now})
                2'b10: begin if (mac_wq_cnt == 0) begin mac_wq_addr0 <= mac_wr_addr; mac_wq_data0 <= mac_wr_data; end
                             else begin mac_wq_addr1 <= mac_wr_addr; mac_wq_data1 <= mac_wr_data; end
                             mac_wq_cnt <= mac_wq_cnt + 1; end
                2'b01: begin mac_wq_addr0 <= mac_wq_addr1; mac_wq_data0 <= mac_wq_data1; mac_wq_cnt <= mac_wq_cnt - 1; end
                2'b11: begin if (mac_wq_cnt == 1) begin mac_wq_addr0 <= mac_wr_addr; mac_wq_data0 <= mac_wr_data; end
                             else begin mac_wq_addr0 <= mac_wq_addr1; mac_wq_data0 <= mac_wq_data1;
                                        mac_wq_addr1 <= mac_wr_addr; mac_wq_data1 <= mac_wr_data; end end
                default: ;
            endcase

            // ── DMA write assembler: merge / stage ──
            if (wa_accept) begin
                if (!wa_valid) begin
                    wa_word  <= {dma_wr_addr[26:4], 4'b0000};
                    wa_data  <= 0;
                    wa_lanes <= 0;
                end
                wa_valid <= 1;
                wa_idle  <= 0;
                wa_data[dma_wr_addr[3:0]*8 +: 8] <= dma_wr_byte;
                wa_lanes[dma_wr_addr[3:0]]       <= 1'b1;
            end else if (wa_valid && wa_idle != WA_IDLE_LIMIT) begin
                wa_idle <= wa_idle + 1;
            end
            // Flush triggers: word change, idle timeout, or a read waiting.
            if (wa_valid && !wa_flush_req &&
                ((dma_wr_en && !wa_same) || (wa_idle == WA_IDLE_LIMIT) || dma_rd_en))
                wa_flush_req <= 1;

            // ── DMA read accept ──
            if (dr_accept) begin
                if (rc_hit) begin
                    dma_rd_byte  <= rc_data[dma_rd_addr[3:0]*8 +: 8];
                    dma_rd_valid <= 1;
                end else begin
                    dr_busy <= 1; dr_req <= 1;
                    dr_addr <= dma_rd_addr; dr_off <= dma_rd_addr[3:0];
                end
            end
            if (dma_rd_fill) begin
                rc_data  <= app_rd_data;
                rc_word  <= {dr_addr[26:4], 4'b0000};
                rc_valid <= 1;
                dma_rd_byte  <= app_rd_data[dr_off*8 +: 8];
                dma_rd_valid <= 1;
                dr_busy <= 0;
            end

            // ── Issue one MIG operation (static priority) ──
            if (issue_free) begin
                if (nmc_wr_req) begin
                    app_addr <= to_mig(nmc_wr_req_addr); app_cmd <= CMD_WRITE; app_en <= 1;
                    app_wdf_data <= nmc_wr_req_data; app_wdf_mask <= 16'h0000;
                    app_wdf_wren <= 1; app_wdf_end <= 1; issuing <= 1;
                    rc_valid <= 0;
                end else if (nmc_rd_req && !rd_busy) begin
                    app_addr <= to_mig(nmc_rd_req_addr); app_cmd <= CMD_READ; app_en <= 1;
                    issuing <= 1; rd_busy <= 1; rd_owner <= OWN_NMC; nmc_rd_req <= nmc_rd_en;
                end else if (mac_wr_req) begin
                    app_addr <= to_mig(mac_wr_req_addr); app_cmd <= CMD_WRITE; app_en <= 1;
                    app_wdf_data <= mac_wr_req_data; app_wdf_mask <= 16'h0000;
                    app_wdf_wren <= 1; app_wdf_end <= 1; issuing <= 1;
                    rc_valid <= 0;
                end else if (mac_rd_req && !rd_busy) begin
                    app_addr <= to_mig(mac_rd_req_addr); app_cmd <= CMD_READ; app_en <= 1;
                    issuing <= 1; rd_busy <= 1; rd_owner <= OWN_MAC; mac_rd_req <= mac_rd_en;
                end else if (wa_flush_req) begin
                    app_addr <= to_mig(wa_word); app_cmd <= CMD_WRITE; app_en <= 1;
                    app_wdf_data <= wa_data; app_wdf_mask <= ~wa_lanes;
                    app_wdf_wren <= 1; app_wdf_end <= 1; issuing <= 1;
                    wa_flush_req <= 0; wa_valid <= 0; wa_lanes <= 0; wa_idle <= 0;
                    rc_valid <= 0;
                end else if (dr_req && !rd_busy) begin
                    app_addr <= to_mig(dr_addr); app_cmd <= CMD_READ; app_en <= 1;
                    issuing <= 1; rd_busy <= 1; rd_owner <= OWN_DMA; dr_req <= 0;
                end
            end
        end
    end
endmodule
