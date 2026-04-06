`timescale 1ns / 1ps

module tb_edgecoh_controller;
    reg         clk;
    reg         rst_n;
    reg  [7:0]  rx_data;
    reg         rx_valid;
    wire        rx_ready;
    wire [7:0]  tx_data;
    wire        tx_valid;
    reg         tx_ready;
    wire        nmc_start;
    wire [7:0]  nmc_op;
    wire [31:0] nmc_table_base;
    wire [31:0] nmc_table_rows;
    wire [31:0] nmc_table_cols;
    wire [31:0] nmc_input_addr;
    wire [31:0] nmc_input_len;
    wire [31:0] nmc_output_addr;
    reg         nmc_done;
    wire        dma_wr_en;
    wire [31:0] dma_wr_addr;
    wire [7:0]  dma_wr_data;
    wire        dma_rd_en;
    wire [31:0] dma_rd_addr;
    reg  [7:0]  dma_rd_data;
    reg         dma_rd_valid;
    wire        barrier_ack;

    edgecoh_controller uut (
        .clk(clk), .rst_n(rst_n),
        .rx_data(rx_data), .rx_valid(rx_valid), .rx_ready(rx_ready),
        .tx_data(tx_data), .tx_valid(tx_valid), .tx_ready(tx_ready),
        .nmc_start(nmc_start), .nmc_op(nmc_op),
        .nmc_table_base(nmc_table_base), .nmc_table_rows(nmc_table_rows),
        .nmc_table_cols(nmc_table_cols), .nmc_input_addr(nmc_input_addr),
        .nmc_input_len(nmc_input_len), .nmc_output_addr(nmc_output_addr),
        .nmc_done(nmc_done),
        .dma_wr_en(dma_wr_en), .dma_wr_addr(dma_wr_addr), .dma_wr_data(dma_wr_data),
        .dma_rd_en(dma_rd_en), .dma_rd_addr(dma_rd_addr),
        .dma_rd_data(dma_rd_data), .dma_rd_valid(dma_rd_valid),
        .barrier_ack(barrier_ack)
    );

    always #5 clk = ~clk;

    task send_byte(input [7:0] data);
        begin
            @(posedge clk);
            rx_data  <= data;
            rx_valid <= 1;
            @(posedge clk);
            while (!rx_ready) @(posedge clk);
            rx_valid <= 0;
        end
    endtask

    task send_barrier;
        begin
            send_byte(8'h03); send_byte(8'h00);
            send_byte(8'h00); send_byte(8'h00);
            send_byte(8'h00); send_byte(8'h00);
            send_byte(8'h00); send_byte(8'h00);
        end
    endtask

    task send_nmc_embedding;
        begin
            send_byte(8'h20); send_byte(8'h00);
            send_byte(8'h07); send_byte(8'h00);
            send_byte(8'h19); send_byte(8'h00);
            send_byte(8'h00); send_byte(8'h00);
            send_byte(8'h01);
            send_byte(8'h00); send_byte(8'h00); send_byte(8'h10); send_byte(8'h00);
            send_byte(8'h10); send_byte(8'h27); send_byte(8'h00); send_byte(8'h00);
            send_byte(8'h40); send_byte(8'h00); send_byte(8'h00); send_byte(8'h00);
            send_byte(8'h00); send_byte(8'h00); send_byte(8'h50); send_byte(8'h00);
            send_byte(8'h80); send_byte(8'h00); send_byte(8'h00); send_byte(8'h00);
            send_byte(8'h00); send_byte(8'h00); send_byte(8'h60); send_byte(8'h00);
        end
    endtask

    integer pass_count;

    initial begin
        clk = 0; rst_n = 0;
        rx_data = 0; rx_valid = 0; tx_ready = 1;
        nmc_done = 0; dma_rd_data = 0; dma_rd_valid = 0;
        pass_count = 0;

        #20 rst_n = 1; #20;

        $display("Test 1: SYNC_BARRIER...");
        send_barrier;
        #100;
        pass_count = pass_count + 1;

        #200;

        $display("Test 2: NMC_EXEC embedding lookup...");
        send_nmc_embedding;
        #100;
        @(posedge clk); nmc_done <= 1;
        @(posedge clk); nmc_done <= 0;
        #100;
        pass_count = pass_count + 1;

        $display("All %0d edgecoh_controller tests completed.", pass_count);
        $finish;
    end

    always @(posedge nmc_start) begin
        $display("  NMC started: op=%h table_base=%h rows=%0d cols=%0d",
                 nmc_op, nmc_table_base, nmc_table_rows, nmc_table_cols);
    end
endmodule
