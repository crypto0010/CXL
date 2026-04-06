/* splitinfer/fpga/src/ddr2_arbiter.v
 *
 * Arbitrates DDR2 access between NMC compute units, MAC controller, and EdgeCoh DMA.
 * Priority: NMC (embedding) > MAC > DMA.
 * CLOCK DOMAIN: Runs on MIG ui_clk (~81.25 MHz), NOT sys_clk (100 MHz).
 * MIG user interface: 128-bit data (BL8 x 16-bit DDR2), ~27-bit address.
 */
`timescale 1ns / 1ps

module ddr2_arbiter (
    input wire clk,       /* MIG ui_clk (~81.25 MHz) */
    input wire rst_n,     /* MIG ui_clk_sync_rst (active-high from MIG, invert externally) */
    output reg [26:0] app_addr, output reg [2:0] app_cmd,
    output reg app_en, output reg [127:0] app_wdf_data, output reg app_wdf_wren,
    output reg         app_wdf_end,  /* MIG DDR2 requires wdf_end asserted with wdf_wren for BL8 */
    input wire [127:0] app_rd_data, input wire app_rd_data_valid,
    input wire app_rdy, input wire app_wdf_rdy,
    input wire nmc_rd_en, input wire [26:0] nmc_rd_addr,
    output wire [127:0] nmc_rd_data, output wire nmc_rd_valid,
    input wire nmc_wr_en, input wire [26:0] nmc_wr_addr, input wire [127:0] nmc_wr_data,
    // MAC ports (mac_controller)
    input wire mac_rd_en, input wire [26:0] mac_rd_addr,
    output wire [127:0] mac_rd_data, output wire mac_rd_valid,
    input wire mac_wr_en, input wire [26:0] mac_wr_addr, input wire [127:0] mac_wr_data,
    // DMA ports (edgecoh)
    input wire dma_rd_en, input wire [26:0] dma_rd_addr,
    output wire [127:0] dma_rd_data, output wire dma_rd_valid,
    input wire dma_wr_en, input wire [26:0] dma_wr_addr, input wire [7:0] dma_wr_byte
);

    assign nmc_rd_data = app_rd_data;
    assign mac_rd_data = app_rd_data;
    assign dma_rd_data = app_rd_data;

    reg nmc_rd_pending, mac_rd_pending, dma_rd_pending;
    assign nmc_rd_valid = app_rd_data_valid && nmc_rd_pending;
    assign mac_rd_valid = app_rd_data_valid && mac_rd_pending;
    assign dma_rd_valid = app_rd_data_valid && dma_rd_pending;

    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            app_en <= 0; app_wdf_wren <= 0; app_wdf_end <= 0;
            nmc_rd_pending <= 0; mac_rd_pending <= 0; dma_rd_pending <= 0;
        end else begin
            app_en <= 0; app_wdf_wren <= 0; app_wdf_end <= 0;
            if (app_rd_data_valid) begin nmc_rd_pending <= 0; mac_rd_pending <= 0; dma_rd_pending <= 0; end

            /* NMC (embedding) has highest priority */
            if (nmc_wr_en && app_rdy && app_wdf_rdy) begin
                app_addr <= nmc_wr_addr; app_cmd <= 3'b000; app_en <= 1;
                app_wdf_data <= nmc_wr_data; app_wdf_wren <= 1; app_wdf_end <= 1;
            end else if (nmc_rd_en && app_rdy) begin
                app_addr <= nmc_rd_addr; app_cmd <= 3'b001; app_en <= 1; nmc_rd_pending <= 1;
            /* MAC has second priority */
            end else if (mac_wr_en && app_rdy && app_wdf_rdy) begin
                app_addr <= mac_wr_addr; app_cmd <= 3'b000; app_en <= 1;
                app_wdf_data <= mac_wr_data; app_wdf_wren <= 1; app_wdf_end <= 1;
            end else if (mac_rd_en && app_rdy) begin
                app_addr <= mac_rd_addr; app_cmd <= 3'b001; app_en <= 1; mac_rd_pending <= 1;
            /* DMA has lowest priority */
            end else if (dma_wr_en && app_rdy && app_wdf_rdy) begin
                app_addr <= dma_wr_addr; app_cmd <= 3'b000; app_en <= 1;
                app_wdf_data <= {120'd0, dma_wr_byte}; app_wdf_wren <= 1; app_wdf_end <= 1;
            end else if (dma_rd_en && app_rdy) begin
                app_addr <= dma_rd_addr; app_cmd <= 3'b001; app_en <= 1; dma_rd_pending <= 1;
            end
        end
    end
endmodule
