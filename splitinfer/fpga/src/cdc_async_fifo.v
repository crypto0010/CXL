/* splitinfer/fpga/src/cdc_async_fifo.v
 *
 * Gray-code pointer async FIFO for clock domain crossing.
 * Used for DMA byte-stream path between sys_clk and ui_clk.
 */
`timescale 1ns / 1ps

module cdc_async_fifo #(
    parameter WIDTH      = 8,
    parameter DEPTH_LOG2 = 4   /* FIFO depth = 2^DEPTH_LOG2 */
) (
    input  wire             wr_clk,
    input  wire             wr_rst_n,
    input  wire             wr_en,
    input  wire [WIDTH-1:0] wr_data,
    output reg              wr_full,
    input  wire             rd_clk,
    input  wire             rd_rst_n,
    input  wire             rd_en,
    output wire [WIDTH-1:0] rd_data,
    output reg              rd_empty
);

    localparam DEPTH = 1 << DEPTH_LOG2;
    localparam PTR_W = DEPTH_LOG2 + 1;  /* extra MSB for full/empty */

    /* --- Dual-port memory --- */
    reg [WIDTH-1:0] mem [0:DEPTH-1];

    /* --- Write domain --- */
    reg  [PTR_W-1:0] wr_bin, wr_gray;
    wire             wr_incr = wr_en & ~wr_full;
    wire [PTR_W-1:0] wr_bin_next  = wr_bin + wr_incr;
    wire [PTR_W-1:0] wr_gray_next = wr_bin_next ^ (wr_bin_next >> 1);

    always @(posedge wr_clk or negedge wr_rst_n)
        if (!wr_rst_n) begin
            wr_bin  <= {PTR_W{1'b0}};
            wr_gray <= {PTR_W{1'b0}};
        end else begin
            wr_bin  <= wr_bin_next;
            wr_gray <= wr_gray_next;
        end

    /* Write to memory; wr_bin indexes the current slot before NBA update */
    always @(posedge wr_clk)
        if (wr_incr)
            mem[wr_bin[DEPTH_LOG2-1:0]] <= wr_data;

    /* --- Read domain --- */
    reg  [PTR_W-1:0] rd_bin, rd_gray;
    wire             rd_incr = rd_en & ~rd_empty;
    wire [PTR_W-1:0] rd_bin_next  = rd_bin + rd_incr;
    wire [PTR_W-1:0] rd_gray_next = rd_bin_next ^ (rd_bin_next >> 1);

    always @(posedge rd_clk or negedge rd_rst_n)
        if (!rd_rst_n) begin
            rd_bin  <= {PTR_W{1'b0}};
            rd_gray <= {PTR_W{1'b0}};
        end else begin
            rd_bin  <= rd_bin_next;
            rd_gray <= rd_gray_next;
        end

    /* Combinational read */
    assign rd_data = mem[rd_bin[DEPTH_LOG2-1:0]];

    /* --- 2FF synchronizers --- */

    /* Sync wr_gray into rd_clk domain */
    reg [PTR_W-1:0] wr_gray_sync1, wr_gray_sync2;
    always @(posedge rd_clk or negedge rd_rst_n)
        if (!rd_rst_n) begin
            wr_gray_sync1 <= {PTR_W{1'b0}};
            wr_gray_sync2 <= {PTR_W{1'b0}};
        end else begin
            wr_gray_sync1 <= wr_gray;
            wr_gray_sync2 <= wr_gray_sync1;
        end

    /* Sync rd_gray into wr_clk domain */
    reg [PTR_W-1:0] rd_gray_sync1, rd_gray_sync2;
    always @(posedge wr_clk or negedge wr_rst_n)
        if (!wr_rst_n) begin
            rd_gray_sync1 <= {PTR_W{1'b0}};
            rd_gray_sync2 <= {PTR_W{1'b0}};
        end else begin
            rd_gray_sync1 <= rd_gray;
            rd_gray_sync2 <= rd_gray_sync1;
        end

    /* --- Full / Empty flags (registered to break comb loops) --- */

    /* Full: wr_gray_next top 2 bits inverted match rd_gray_sync2, rest same */
    wire wr_full_val = (wr_gray_next[PTR_W-1:PTR_W-2] ==
                        ~rd_gray_sync2[PTR_W-1:PTR_W-2]) &&
                       (wr_gray_next[PTR_W-3:0] ==
                        rd_gray_sync2[PTR_W-3:0]);

    always @(posedge wr_clk or negedge wr_rst_n)
        if (!wr_rst_n) wr_full <= 1'b0;
        else           wr_full <= wr_full_val;

    /* Empty: rd_gray_next equals wr_gray_sync2 */
    wire rd_empty_val = (rd_gray_next == wr_gray_sync2);

    always @(posedge rd_clk or negedge rd_rst_n)
        if (!rd_rst_n) rd_empty <= 1'b1;
        else           rd_empty <= rd_empty_val;

endmodule
