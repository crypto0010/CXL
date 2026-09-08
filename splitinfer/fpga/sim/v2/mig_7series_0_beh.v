/* Behavioural stand-in for the MIG IP with its exact port list, wrapping
 * ddr2_ui_model.  ui_clk is a free-running 81.25 MHz clock, asynchronous
 * to sys_clk, so the CDC FIFOs are exercised for real. */
`timescale 1ns / 1ps
module mig_7series_0(ddr2_dq, ddr2_dqs_n, ddr2_dqs_p, ddr2_addr, ddr2_ba, ddr2_ras_n, ddr2_cas_n, ddr2_we_n,
  ddr2_ck_p, ddr2_ck_n, ddr2_cke, ddr2_cs_n, ddr2_dm, ddr2_odt, sys_clk_i, app_addr, app_cmd, app_en,
  app_wdf_data, app_wdf_end, app_wdf_mask, app_wdf_wren, app_rd_data, app_rd_data_end, app_rd_data_valid,
  app_rdy, app_wdf_rdy, app_sr_req, app_ref_req, app_zq_req, app_sr_active, app_ref_ack, app_zq_ack, ui_clk,
  ui_clk_sync_rst, init_calib_complete, sys_rst);
  inout [15:0] ddr2_dq; inout [1:0] ddr2_dqs_n; inout [1:0] ddr2_dqs_p;
  output [12:0] ddr2_addr; output [2:0] ddr2_ba; output ddr2_ras_n, ddr2_cas_n, ddr2_we_n;
  output [0:0] ddr2_ck_p, ddr2_ck_n, ddr2_cke, ddr2_cs_n; output [1:0] ddr2_dm; output [0:0] ddr2_odt;
  input sys_clk_i; input [26:0] app_addr; input [2:0] app_cmd; input app_en;
  input [127:0] app_wdf_data; input app_wdf_end; input [15:0] app_wdf_mask; input app_wdf_wren;
  output [127:0] app_rd_data; output app_rd_data_end, app_rd_data_valid, app_rdy, app_wdf_rdy;
  input app_sr_req, app_ref_req, app_zq_req; output app_sr_active, app_ref_ack, app_zq_ack;
  output reg ui_clk; output reg ui_clk_sync_rst; output reg init_calib_complete; input sys_rst;
  assign ddr2_addr = 0; assign ddr2_ba = 0; assign ddr2_ras_n = 1; assign ddr2_cas_n = 1; assign ddr2_we_n = 1;
  assign ddr2_ck_p = 0; assign ddr2_ck_n = 1; assign ddr2_cke = 0; assign ddr2_cs_n = 1; assign ddr2_dm = 0; assign ddr2_odt = 0;
  assign app_sr_active = 0; assign app_ref_ack = 0; assign app_zq_ack = 0;
  assign app_rd_data_end = app_rd_data_valid;
  initial ui_clk = 0; always #6.154 ui_clk = ~ui_clk;
  integer c;
  initial begin ui_clk_sync_rst = 1; init_calib_complete = 0; for (c = 0; c < 40; c = c + 1) @(posedge ui_clk); ui_clk_sync_rst = 0;
                for (c = 0; c < 200; c = c + 1) @(posedge ui_clk); init_calib_complete = 1; end
  ddr2_ui_model #(.BYTES(1 << 20), .RD_LAT(22), .STALL_PCT(20)) mem (.clk(ui_clk), .rst_n(~ui_clk_sync_rst),
    .app_addr(app_addr), .app_cmd(app_cmd), .app_en(app_en), .app_wdf_data(app_wdf_data), .app_wdf_mask(app_wdf_mask),
    .app_wdf_wren(app_wdf_wren), .app_wdf_end(app_wdf_end), .app_rd_data(app_rd_data), .app_rd_data_valid(app_rd_data_valid),
    .app_rdy(app_rdy), .app_wdf_rdy(app_wdf_rdy));
endmodule
