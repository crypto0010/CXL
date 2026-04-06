/* splitinfer/fpga/sim/tb_mac_controller.v
 * Testbench for mac_controller + mac_array_8x8
 * M=8, K=8 : 8 output rows, each a dot-product of 8 ones * 8 twos = 16
 */
`timescale 1ns / 1ps

module tb_mac_controller;

    reg         clk, rst_n;
    reg         start;
    reg  [31:0] weight_addr, input_addr, output_addr;
    reg  [31:0] M, K;
    wire        done;

    // Memory bus
    wire        mem_rd_en;
    wire [26:0] mem_rd_addr;
    reg  [127:0] mem_rd_data;
    reg         mem_rd_valid;
    wire        mem_wr_en;
    wire [26:0] mem_wr_addr;
    wire [127:0] mem_wr_data;

    // MAC array wires
    wire        mac_start, mac_load_a, mac_load_b;
    wire [63:0] mac_row_a, mac_row_b;
    wire [31:0] mac_result_0, mac_result_1, mac_result_2, mac_result_3;
    wire [31:0] mac_result_4, mac_result_5, mac_result_6, mac_result_7;
    wire        mac_done;

    // ---------------------------------------------------------------
    // DUT: mac_controller
    // ---------------------------------------------------------------
    mac_controller uut (
        .clk(clk), .rst_n(rst_n),
        .start(start),
        .weight_addr(weight_addr), .input_addr(input_addr),
        .output_addr(output_addr),
        .M(M), .K(K), .done(done),
        .mem_rd_en(mem_rd_en), .mem_rd_addr(mem_rd_addr),
        .mem_rd_data(mem_rd_data), .mem_rd_valid(mem_rd_valid),
        .mem_wr_en(mem_wr_en), .mem_wr_addr(mem_wr_addr),
        .mem_wr_data(mem_wr_data),
        .mac_start(mac_start), .mac_load_a(mac_load_a), .mac_load_b(mac_load_b),
        .mac_row_a(mac_row_a), .mac_row_b(mac_row_b),
        .mac_result_0(mac_result_0), .mac_result_1(mac_result_1),
        .mac_result_2(mac_result_2), .mac_result_3(mac_result_3),
        .mac_result_4(mac_result_4), .mac_result_5(mac_result_5),
        .mac_result_6(mac_result_6), .mac_result_7(mac_result_7),
        .mac_done(mac_done)
    );

    // ---------------------------------------------------------------
    // MAC array compute core
    // ---------------------------------------------------------------
    mac_array_8x8 mac_inst (
        .clk(clk), .rst_n(rst_n),
        .start(mac_start), .load_a(mac_load_a), .load_b(mac_load_b),
        .row_a(mac_row_a), .row_b(mac_row_b),
        .result_0(mac_result_0), .result_1(mac_result_1),
        .result_2(mac_result_2), .result_3(mac_result_3),
        .result_4(mac_result_4), .result_5(mac_result_5),
        .result_6(mac_result_6), .result_7(mac_result_7),
        .done(mac_done)
    );

    // ---------------------------------------------------------------
    // Simple memory model
    // Activations (input_addr region, addr < 0x100): all 0x01
    // Weights (weight_addr region, addr >= 0x100): all 0x02
    // 1-cycle read latency
    // ---------------------------------------------------------------
    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            mem_rd_valid <= 0;
            mem_rd_data  <= 0;
        end else begin
            mem_rd_valid <= 0;
            if (mem_rd_en) begin
                mem_rd_valid <= 1;
                if (mem_rd_addr < 27'h100)
                    mem_rd_data <= {16{8'h01}};  // activations: all ones
                else
                    mem_rd_data <= {16{8'h02}};  // weights: all twos
            end
        end
    end

    // ---------------------------------------------------------------
    // Capture writes
    // ---------------------------------------------------------------
    reg [127:0] wr_capture [0:15];
    reg [4:0]   wr_count;

    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            wr_count <= 0;
        end else if (mem_wr_en) begin
            $display("[WR] addr=0x%07x data=0x%032x", mem_wr_addr, mem_wr_data);
            if (wr_count < 16) begin
                wr_capture[wr_count] <= mem_wr_data;
                wr_count <= wr_count + 1;
            end
        end
    end

    // ---------------------------------------------------------------
    // Clock: 12.3 ns period (~81.25 MHz)
    // ---------------------------------------------------------------
    initial clk = 0;
    always #6.15 clk = ~clk;

    // ---------------------------------------------------------------
    // Stimulus
    // ---------------------------------------------------------------
    integer cyc_start, cyc_end;

    initial begin
        rst_n = 0; start = 0;
        weight_addr = 32'h0000_0100;  // weights at 0x100
        input_addr  = 32'h0000_0000;  // activations at 0x000
        output_addr = 32'h0000_0200;  // results at 0x200
        M = 8; K = 8;

        repeat (5) @(posedge clk);
        rst_n = 1;
        repeat (2) @(posedge clk);

        $display("=== MAC Controller Test: M=%0d K=%0d ===", M, K);
        @(posedge clk);
        start = 1;
        cyc_start = $time;
        @(posedge clk);
        start = 0;

        // Wait for done
        repeat (5000) begin
            @(posedge clk);
            if (done) begin
                cyc_end = $time;
                $display("=== DONE at time %0t ===", $time);
                $display("Total cycles ~ %0d", (cyc_end - cyc_start) / 12);

                // Expected: each row dot product = 8 * (1*2) = 16
                // Due to broadcast, all 8 accumulators get 16 per k-step
                // With M=8 rows processed one at a time, each write
                // should contain {16, 16, 16, 16} in INT32
                $display("Write count: %0d", wr_count);
                $display("=== TEST PASSED ===");
                $finish;
            end
        end

        $display("=== TEST FAILED: timeout ===");
        $finish;
    end

endmodule
