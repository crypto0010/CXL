/* splitinfer/fpga/sim/tb_embedding_lookup.v */
`timescale 1ns / 1ps

module tb_embedding_lookup;
    reg clk, rst_n, start;
    reg [31:0] table_base_addr, embed_dim, indices_addr, num_indices, output_addr;
    wire done;
    wire mem_rd_en; wire [26:0] mem_rd_addr;
    reg [127:0] mem_rd_data; reg mem_rd_valid;
    wire mem_wr_en; wire [26:0] mem_wr_addr; wire [127:0] mem_wr_data;

    embedding_lookup uut (
        .clk(clk), .rst_n(rst_n), .start(start),
        .table_base_addr(table_base_addr), .embed_dim(embed_dim),
        .indices_addr(indices_addr), .num_indices(num_indices),
        .output_addr(output_addr), .done(done),
        .mem_rd_en(mem_rd_en), .mem_rd_addr(mem_rd_addr),
        .mem_rd_data(mem_rd_data), .mem_rd_valid(mem_rd_valid),
        .mem_wr_en(mem_wr_en), .mem_wr_addr(mem_wr_addr), .mem_wr_data(mem_wr_data)
    );

    always #5 clk = ~clk;

    reg [7:0] fake_mem [0:1023];
    reg [3:0] rd_delay; reg rd_pending; reg [26:0] rd_pending_addr;

    always @(posedge clk) begin
        mem_rd_valid <= 0;
        if (mem_rd_en) begin rd_delay <= 4; rd_pending <= 1; rd_pending_addr <= mem_rd_addr; end
        if (rd_pending && rd_delay > 0) begin
            rd_delay <= rd_delay - 1;
            if (rd_delay == 1) begin
                mem_rd_data <= {
                    fake_mem[rd_pending_addr+15], fake_mem[rd_pending_addr+14],
                    fake_mem[rd_pending_addr+13], fake_mem[rd_pending_addr+12],
                    fake_mem[rd_pending_addr+11], fake_mem[rd_pending_addr+10],
                    fake_mem[rd_pending_addr+9],  fake_mem[rd_pending_addr+8],
                    fake_mem[rd_pending_addr+7],  fake_mem[rd_pending_addr+6],
                    fake_mem[rd_pending_addr+5],  fake_mem[rd_pending_addr+4],
                    fake_mem[rd_pending_addr+3],  fake_mem[rd_pending_addr+2],
                    fake_mem[rd_pending_addr+1],  fake_mem[rd_pending_addr+0]
                };
                mem_rd_valid <= 1; rd_pending <= 0;
            end
        end
    end

    integer i;
    initial begin
        clk = 0; rst_n = 0; start = 0; mem_rd_valid = 0; rd_pending = 0;

        for (i = 0; i < 1024; i = i + 1) fake_mem[i] = 0;
        fake_mem[0] = 8'h02; fake_mem[4] = 8'h00;
        for (i = 0; i < 16; i = i + 1) fake_mem[256 + i] = 8'hAA;
        for (i = 0; i < 16; i = i + 1) fake_mem[288 + i] = 8'hCC;

        #20 rst_n = 1; #20;

        table_base_addr <= 32'h100; embed_dim <= 32'd16;
        indices_addr <= 32'h000; num_indices <= 32'd2; output_addr <= 32'h200;

        @(posedge clk); start <= 1; @(posedge clk); start <= 0;

        wait(done); #20;
        $display("Embedding lookup test completed.");
        $finish;
    end
endmodule
