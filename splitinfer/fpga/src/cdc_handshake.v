/* splitinfer/fpga/src/cdc_handshake.v
 *
 * Handshake-based CDC for multi-bit payload transfer between clock domains.
 * Toggle-based req/ack with 2FF synchronizers on each crossing.
 */
`timescale 1ns / 1ps

module cdc_handshake #(parameter WIDTH = 1) (
    input  wire             src_clk,
    input  wire             src_rst_n,
    input  wire             src_valid,   /* pulse: latch data, start handshake */
    input  wire [WIDTH-1:0] src_data,
    output wire             src_ready,   /* high when idle, safe to send */
    input  wire             dst_clk,
    input  wire             dst_rst_n,
    output reg              dst_valid,   /* pulse: data available */
    output reg  [WIDTH-1:0] dst_data
);

    /* --- Registers --- */
    reg              req_toggle;
    reg [WIDTH-1:0]  src_data_hold;
    reg              ack_toggle;

    /* --- Source domain --- */

    /* Synchronize ack_toggle into src domain (2FF) */
    reg [1:0] ack_sync;
    always @(posedge src_clk or negedge src_rst_n)
        if (!src_rst_n) ack_sync <= 2'b0;
        else            ack_sync <= {ack_sync[0], ack_toggle};

    assign src_ready = (req_toggle == ack_sync[1]);

    always @(posedge src_clk or negedge src_rst_n)
        if (!src_rst_n) begin
            req_toggle   <= 1'b0;
            src_data_hold <= {WIDTH{1'b0}};
        end else if (src_valid && src_ready) begin
            req_toggle   <= ~req_toggle;
            src_data_hold <= src_data;
        end

    /* --- Destination domain --- */

    /* Synchronize req_toggle into dst domain (2FF) */
    reg [1:0] req_sync;
    always @(posedge dst_clk or negedge dst_rst_n)
        if (!dst_rst_n) req_sync <= 2'b0;
        else            req_sync <= {req_sync[0], req_toggle};

    /* Detect edge: req arrived when synced value differs from local ack */
    wire req_arrived = (req_sync[1] != ack_toggle);

    always @(posedge dst_clk or negedge dst_rst_n)
        if (!dst_rst_n) begin
            ack_toggle <= 1'b0;
            dst_valid  <= 1'b0;
            dst_data   <= {WIDTH{1'b0}};
        end else begin
            dst_valid <= 1'b0;
            if (req_arrived) begin
                dst_data   <= src_data_hold;
                dst_valid  <= 1'b1;
                ack_toggle <= ~ack_toggle;
            end
        end

endmodule
