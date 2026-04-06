/* splitinfer/fpga/src/usb_interface.v */
`timescale 1ns / 1ps

module usb_interface #(
    parameter CLK_FREQ  = 100_000_000,
    parameter BAUD_RATE = 921_600      /* FT2232HQ supports up to 12 Mbaud; 921600 is reliable and fast */
)(
    input  wire       clk,
    input  wire       rst_n,
    input  wire       uart_rx,
    output wire       uart_tx,
    output reg  [7:0] rx_data,
    output reg        rx_valid,
    input  wire       rx_ready,
    input  wire [7:0] tx_data,
    input  wire       tx_valid,
    output reg        tx_ready
);

    localparam CLKS_PER_BIT = CLK_FREQ / BAUD_RATE;

    /* UART RX */
    reg [2:0]  rx_state;
    reg [15:0] rx_clk_count;
    reg [2:0]  rx_bit_idx;
    reg [7:0]  rx_shift;
    reg        uart_rx_r1, uart_rx_r2;

    localparam RX_IDLE = 3'd0, RX_START = 3'd1, RX_DATA = 3'd2, RX_STOP = 3'd3;

    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin uart_rx_r1 <= 1; uart_rx_r2 <= 1; end
        else begin uart_rx_r1 <= uart_rx; uart_rx_r2 <= uart_rx_r1; end
    end

    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            rx_state <= RX_IDLE; rx_valid <= 0; rx_data <= 0;
            rx_clk_count <= 0; rx_bit_idx <= 0; rx_shift <= 0;
        end else begin
            rx_valid <= 0;
            case (rx_state)
                RX_IDLE: if (uart_rx_r2 == 0) begin rx_state <= RX_START; rx_clk_count <= 0; end
                RX_START: if (rx_clk_count == CLKS_PER_BIT/2) begin
                    rx_state <= RX_DATA; rx_clk_count <= 0; rx_bit_idx <= 0;
                end else rx_clk_count <= rx_clk_count + 1;
                RX_DATA: if (rx_clk_count == CLKS_PER_BIT-1) begin
                    rx_shift[rx_bit_idx] <= uart_rx_r2; rx_clk_count <= 0;
                    if (rx_bit_idx == 7) rx_state <= RX_STOP;
                    else rx_bit_idx <= rx_bit_idx + 1;
                end else rx_clk_count <= rx_clk_count + 1;
                RX_STOP: if (rx_clk_count == CLKS_PER_BIT-1) begin
                    rx_data <= rx_shift; rx_valid <= 1; rx_state <= RX_IDLE;
                end else rx_clk_count <= rx_clk_count + 1;
            endcase
        end
    end

    /* UART TX */
    reg [2:0]  tx_state;
    reg [15:0] tx_clk_count;
    reg [2:0]  tx_bit_idx;
    reg [7:0]  tx_shift;
    reg        tx_out;
    assign uart_tx = tx_out;

    localparam TX_IDLE = 3'd0, TX_START = 3'd1, TX_DATA = 3'd2, TX_STOP = 3'd3;

    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            tx_state <= TX_IDLE; tx_ready <= 1; tx_out <= 1;
            tx_clk_count <= 0; tx_bit_idx <= 0; tx_shift <= 0;
        end else begin
            case (tx_state)
                TX_IDLE: begin
                    tx_out <= 1; tx_ready <= 1;
                    if (tx_valid && tx_ready) begin
                        tx_shift <= tx_data; tx_state <= TX_START;
                        tx_ready <= 0; tx_clk_count <= 0;
                    end
                end
                TX_START: begin
                    tx_out <= 0;
                    if (tx_clk_count == CLKS_PER_BIT-1) begin
                        tx_state <= TX_DATA; tx_clk_count <= 0; tx_bit_idx <= 0;
                    end else tx_clk_count <= tx_clk_count + 1;
                end
                TX_DATA: begin
                    tx_out <= tx_shift[tx_bit_idx];
                    if (tx_clk_count == CLKS_PER_BIT-1) begin
                        tx_clk_count <= 0;
                        if (tx_bit_idx == 7) tx_state <= TX_STOP;
                        else tx_bit_idx <= tx_bit_idx + 1;
                    end else tx_clk_count <= tx_clk_count + 1;
                end
                TX_STOP: begin
                    tx_out <= 1;
                    if (tx_clk_count == CLKS_PER_BIT-1) tx_state <= TX_IDLE;
                    else tx_clk_count <= tx_clk_count + 1;
                end
            endcase
        end
    end
endmodule
