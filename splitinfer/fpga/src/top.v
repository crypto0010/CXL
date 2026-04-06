/* splitinfer/fpga/src/top.v
 *
 * SplitInfer FPGA top-level module for Nexys 4 DDR (XC7A100T).
 *
 * CLOCK DOMAINS:
 *   sys_clk  (100 MHz) — USB interface, EdgeCoh controller
 *   ui_clk   (~81.25 MHz from MIG) — DDR2 arbiter, NMC dispatch, embedding lookup
 *
 * In SIM_MODE=1, ui_clk is tied to sys_clk and init_calib_complete is
 * forced to 1 so that simulation can proceed without a real MIG instance.
 */
`timescale 1ns / 1ps

module top #(
    parameter SIM_MODE = 0
) (
    input  wire        sys_clk,
    input  wire        sys_rst_n,
    input  wire        uart_rx,
    output wire        uart_tx,
    output wire [3:0]  led,

    /* DDR2 physical interface — directly connected to MIG IP */
    inout  wire [15:0] ddr2_dq,
    inout  wire [1:0]  ddr2_dqs_p,
    inout  wire [1:0]  ddr2_dqs_n,
    output wire [12:0] ddr2_addr,
    output wire [2:0]  ddr2_ba,
    output wire        ddr2_ras_n,
    output wire        ddr2_cas_n,
    output wire        ddr2_we_n,
    output wire [0:0]  ddr2_ck_p,
    output wire [0:0]  ddr2_ck_n,
    output wire [0:0]  ddr2_cke,
    output wire [0:0]  ddr2_cs_n,
    output wire [1:0]  ddr2_dm,
    output wire [0:0]  ddr2_odt
);

    /* ------------------------------------------------------------------ */
    /* Clock / reset generation                                            */
    /* ------------------------------------------------------------------ */

    wire ui_clk;
    wire ui_clk_sync_rst;
    wire ui_rst_n;
    wire init_calib_complete;

    /* Buffered sys_clk for all sys_clk-domain logic                      */
    /* In HW mode: IBUF -> BUFG -> sys_clk_bufg                          */
    /* In SIM mode: sys_clk passed through directly                       */
    wire sys_clk_bufg;

    /* ------------------------------------------------------------------ */
    /* MIG user-interface signals                                          */
    /* ------------------------------------------------------------------ */

    wire [26:0]  app_addr;
    wire [2:0]   app_cmd;
    wire         app_en;
    wire [127:0] app_wdf_data;
    wire [15:0]  app_wdf_mask;
    wire         app_wdf_wren;
    wire         app_wdf_end;
    wire [127:0] app_rd_data;
    wire         app_rd_data_valid;
    wire         app_rd_data_end;
    wire         app_rdy;
    wire         app_wdf_rdy;

    generate
        if (SIM_MODE == 1) begin : sim_clk
            /* Simulation: reuse sys_clk, assert calibration done */
            assign sys_clk_bufg        = sys_clk;
            assign ui_clk              = sys_clk;
            assign ui_clk_sync_rst     = ~sys_rst_n;
            assign ui_rst_n            = sys_rst_n;
            assign init_calib_complete = 1'b1;

            /* Simulation MIG stub: always ready, no read data */
            assign app_rdy             = 1'b1;
            assign app_wdf_rdy         = 1'b1;
            assign app_rd_data         = 128'd0;
            assign app_rd_data_valid   = 1'b0;
            assign app_rd_data_end     = 1'b0;

        end else begin : hw_mig
            /* ---------------------------------------------------------- */
            /* Clock infrastructure                                        */
            /* ---------------------------------------------------------- */

            /* IBUF -> BUFG for sys_clk domain (100 MHz)                  */
            wire sys_clk_ibuf;
            IBUF u_ibuf_sysclk (.I(sys_clk), .O(sys_clk_ibuf));
            BUFG u_bufg_sysclk (.I(sys_clk_ibuf), .O(sys_clk_bufg));

            /* MMCM: 100 MHz -> 200 MHz for MIG sys_clk_i                */
            /* VCO = 100 * 10 = 1000 MHz, CLKOUT0 = 1000 / 5 = 200 MHz  */
            wire clk_200mhz;
            wire mmcm_locked;
            wire mmcm_fb;

            MMCME2_BASE #(
                .CLKIN1_PERIOD  (10.000),
                .CLKFBOUT_MULT_F(10.0),
                .CLKOUT0_DIVIDE_F(5.0),
                .STARTUP_WAIT   ("FALSE")
            ) u_mmcm (
                .CLKIN1   (sys_clk_ibuf),
                .RST      (~sys_rst_n),
                .CLKFBOUT (mmcm_fb),
                .CLKFBIN  (mmcm_fb),
                .CLKOUT0  (clk_200mhz),
                .LOCKED   (mmcm_locked),
                .CLKFBOUTB(), .CLKOUT0B(), .CLKOUT1(), .CLKOUT1B(),
                .CLKOUT2(), .CLKOUT2B(), .CLKOUT3(), .CLKOUT3B(),
                .CLKOUT4(), .CLKOUT5(), .CLKOUT6(),
                .PWRDWN   (1'b0)
            );

            /* ---------------------------------------------------------- */
            /* MIG 7 Series DDR2 controller                                */
            /* ---------------------------------------------------------- */
            mig_7series_0 u_mig (
                /* DDR2 physical interface */
                .ddr2_dq             (ddr2_dq),
                .ddr2_dqs_p          (ddr2_dqs_p),
                .ddr2_dqs_n          (ddr2_dqs_n),
                .ddr2_addr           (ddr2_addr),
                .ddr2_ba             (ddr2_ba),
                .ddr2_ras_n          (ddr2_ras_n),
                .ddr2_cas_n          (ddr2_cas_n),
                .ddr2_we_n           (ddr2_we_n),
                .ddr2_ck_p           (ddr2_ck_p),
                .ddr2_ck_n           (ddr2_ck_n),
                .ddr2_cke            (ddr2_cke),
                .ddr2_cs_n           (ddr2_cs_n),
                .ddr2_dm             (ddr2_dm),
                .ddr2_odt            (ddr2_odt),

                /* Application interface */
                .app_addr            (app_addr),
                .app_cmd             (app_cmd),
                .app_en              (app_en),
                .app_wdf_data        (app_wdf_data),
                .app_wdf_mask        (app_wdf_mask),
                .app_wdf_wren        (app_wdf_wren),
                .app_wdf_end         (app_wdf_end),
                .app_rd_data         (app_rd_data),
                .app_rd_data_valid   (app_rd_data_valid),
                .app_rd_data_end     (app_rd_data_end),
                .app_rdy             (app_rdy),
                .app_wdf_rdy         (app_wdf_rdy),
                .app_sr_req          (1'b0),
                .app_ref_req         (1'b0),
                .app_zq_req          (1'b0),
                .app_sr_active       (),
                .app_ref_ack         (),
                .app_zq_ack          (),

                /* Clock / reset / calibration */
                .ui_clk              (ui_clk),
                .ui_clk_sync_rst     (ui_clk_sync_rst),
                .init_calib_complete (init_calib_complete),

                /* System clock input (200 MHz from MMCM, No Buffer) */
                .sys_clk_i           (clk_200mhz),
                .sys_rst             (sys_rst_n)   /* MIG sys_rst is active-low for ACTIVE LOW config */
            );

            assign ui_rst_n = ~ui_clk_sync_rst;
        end
    endgenerate

    /* Tie off unused mask — write all bytes */
    assign app_wdf_mask = 16'h0000;

    /* ------------------------------------------------------------------ */
    /* sys_clk domain wires                                                */
    /* ------------------------------------------------------------------ */

    /* USB interface <-> EdgeCoh controller */
    wire [7:0] usb_rx_data;
    wire       usb_rx_valid;
    wire       usb_rx_ready;
    wire [7:0] usb_tx_data;
    wire       usb_tx_valid;
    wire       usb_tx_ready;

    /* EdgeCoh controller -> NMC dispatch (crosses to ui_clk via CDC) */
    wire        nmc_start;
    wire [7:0]  nmc_op;
    wire [31:0] nmc_table_base;
    wire [31:0] nmc_table_rows;
    wire [31:0] nmc_table_cols;
    wire [31:0] nmc_input_addr;
    wire [31:0] nmc_input_len;
    wire [31:0] nmc_output_addr;

    /* EdgeCoh DMA ports (sys_clk side) — cross to arbiter (ui_clk) via CDC FIFOs */
    wire        dma_wr_en;
    wire [31:0] dma_wr_addr_wide;
    wire [7:0]  dma_wr_data;
    wire        dma_rd_en;
    wire [31:0] dma_rd_addr_wide;
    wire [7:0]  dma_rd_data_bytes;
    wire        dma_rd_valid_sig;
    wire        barrier_ack;

    /* ------------------------------------------------------------------ */
    /* ui_clk domain wires                                                 */
    /* ------------------------------------------------------------------ */

    /* NMC dispatch -> embedding_lookup */
    wire        emb_start;
    wire [31:0] emb_table_base;
    wire [31:0] emb_embed_dim;
    wire [31:0] emb_indices_addr;
    wire [31:0] emb_num_indices;
    wire [31:0] emb_output_addr;
    wire        emb_done;

    /* MAC controller <-> mac_array_8x8 wires */
    wire        mac_start;          /* nmc_dispatch -> mac_controller */
    wire        mac_ctrl_done;      /* mac_controller -> nmc_dispatch */
    wire        mac_start_array;    /* mac_controller -> mac_array_8x8 */
    wire        mac_load_a, mac_load_b;
    wire [63:0] mac_row_a, mac_row_b;
    wire [31:0] mac_res0, mac_res1, mac_res2, mac_res3;
    wire [31:0] mac_res4, mac_res5, mac_res6, mac_res7;
    wire        mac_compute_done;

    /* MAC controller <-> DDR2 arbiter memory port */
    wire        mac_mem_rd_en;
    wire [26:0] mac_mem_rd_addr;
    wire [127:0] mac_mem_rd_data;
    wire        mac_mem_rd_valid;
    wire        mac_mem_wr_en;
    wire [26:0] mac_mem_wr_addr;
    wire [127:0] mac_mem_wr_data;

    /* nmc_dispatch -> mac_controller parameters */
    wire [31:0] mac_weight_addr, mac_input_addr_d, mac_output_addr_d;
    wire [31:0] mac_M, mac_K;

    /* Elementwise controller wires */
    wire        elt_start;
    wire [31:0] elt_input_addr, elt_output_addr, elt_num_words;
    wire [1:0]  elt_op;
    wire [7:0]  elt_scale;
    wire        elt_done;

    wire        elt_mem_rd_en;
    wire [26:0] elt_mem_rd_addr;
    wire        elt_mem_wr_en;
    wire [26:0] elt_mem_wr_addr;
    wire [127:0] elt_mem_wr_data;

    /* embedding_lookup memory port wires */
    wire        emb_mem_rd_en;
    wire [26:0] emb_mem_rd_addr;
    wire        emb_mem_wr_en;
    wire [26:0] emb_mem_wr_addr;
    wire [127:0] emb_mem_wr_data;

    /* Shared NMC memory port (mux embedding and elementwise — never active simultaneously) */
    wire        nmc_rd_en   = emb_mem_rd_en  | elt_mem_rd_en;
    wire [26:0] nmc_rd_addr = emb_mem_rd_en ? emb_mem_rd_addr : elt_mem_rd_addr;
    wire [127:0] nmc_rd_data;
    wire        nmc_rd_valid;
    wire        nmc_wr_en   = emb_mem_wr_en  | elt_mem_wr_en;
    wire [26:0] nmc_wr_addr = emb_mem_wr_en ? emb_mem_wr_addr : elt_mem_wr_addr;
    wire [127:0] nmc_wr_data = emb_mem_wr_en ? emb_mem_wr_data : elt_mem_wr_data;

    /* dma read data back to EdgeCoh (128-bit -> 8-bit, take LSB) */
    wire [127:0] dma_rd_data_128;
    wire         dma_rd_valid_arb;

    /* ------------------------------------------------------------------ */
    /* CDC bridges (sys_clk_bufg <-> ui_clk)                               */
    /* ------------------------------------------------------------------ */

    /* --- 1. NMC Command CDC (sys_clk -> ui_clk, handshake) ------------ */
    wire [199:0] nmc_cmd_packed = {nmc_output_addr, nmc_input_len, nmc_input_addr,
                                    nmc_table_cols, nmc_table_rows, nmc_table_base, nmc_op};
    wire         nmc_cmd_cdc_ready;
    wire         nmc_cmd_valid_ui;
    wire [199:0] nmc_cmd_ui;

    cdc_handshake #(.WIDTH(200)) u_cdc_nmc_cmd (
        .src_clk   (sys_clk_bufg),
        .src_rst_n (sys_rst_n),
        .src_valid (nmc_start),
        .src_data  (nmc_cmd_packed),
        .src_ready (nmc_cmd_cdc_ready),
        .dst_clk   (ui_clk),
        .dst_rst_n (ui_rst_n),
        .dst_valid (nmc_cmd_valid_ui),
        .dst_data  (nmc_cmd_ui)
    );

    wire [7:0]  nmc_op_ui          = nmc_cmd_ui[7:0];
    wire [31:0] nmc_table_base_ui  = nmc_cmd_ui[39:8];
    wire [31:0] nmc_table_rows_ui  = nmc_cmd_ui[71:40];
    wire [31:0] nmc_table_cols_ui  = nmc_cmd_ui[103:72];
    wire [31:0] nmc_input_addr_ui  = nmc_cmd_ui[135:104];
    wire [31:0] nmc_input_len_ui   = nmc_cmd_ui[167:136];
    wire [31:0] nmc_output_addr_ui = nmc_cmd_ui[199:168];

    /* --- 2. NMC Done CDC (ui_clk -> sys_clk, handshake) --------------- */
    wire nmc_done_ui;   /* from u_nmc */
    wire nmc_done_sys;  /* synced to sys_clk */

    cdc_handshake #(.WIDTH(1)) u_cdc_nmc_done (
        .src_clk   (ui_clk),
        .src_rst_n (ui_rst_n),
        .src_valid (nmc_done_ui),
        .src_data  (1'b1),
        .src_ready (),
        .dst_clk   (sys_clk_bufg),
        .dst_rst_n (sys_rst_n),
        .dst_valid (nmc_done_sys),
        .dst_data  ()
    );

    /* --- 3. DMA Write FIFO (sys_clk -> ui_clk, async FIFO) ----------- */
    wire        dma_wr_fifo_full, dma_wr_fifo_empty;
    wire [35:0] dma_wr_fifo_rd_data;

    cdc_async_fifo #(.WIDTH(36), .DEPTH_LOG2(4)) u_dma_wr_fifo (
        .wr_clk   (sys_clk_bufg),
        .wr_rst_n (sys_rst_n),
        .wr_en    (dma_wr_en && !dma_wr_fifo_full),
        .wr_data  ({dma_wr_data, dma_wr_addr_wide[26:0], 1'b1}),
        .wr_full  (dma_wr_fifo_full),
        .rd_clk   (ui_clk),
        .rd_rst_n (ui_rst_n),
        .rd_en    (!dma_wr_fifo_empty),
        .rd_data  (dma_wr_fifo_rd_data),
        .rd_empty (dma_wr_fifo_empty)
    );

    wire        dma_wr_en_ui   = !dma_wr_fifo_empty;
    wire [26:0] dma_wr_addr_ui = dma_wr_fifo_rd_data[27:1];
    wire [7:0]  dma_wr_data_ui = dma_wr_fifo_rd_data[35:28];

    /* --- 4. DMA Read Cmd FIFO (sys_clk -> ui_clk, async FIFO) -------- */
    wire        dma_rd_cmd_full, dma_rd_cmd_empty;
    wire [26:0] dma_rd_cmd_rd_data;

    cdc_async_fifo #(.WIDTH(27), .DEPTH_LOG2(4)) u_dma_rd_cmd_fifo (
        .wr_clk   (sys_clk_bufg),
        .wr_rst_n (sys_rst_n),
        .wr_en    (dma_rd_en && !dma_rd_cmd_full),
        .wr_data  (dma_rd_addr_wide[26:0]),
        .wr_full  (dma_rd_cmd_full),
        .rd_clk   (ui_clk),
        .rd_rst_n (ui_rst_n),
        .rd_en    (!dma_rd_cmd_empty),
        .rd_data  (dma_rd_cmd_rd_data),
        .rd_empty (dma_rd_cmd_empty)
    );

    wire        dma_rd_en_ui   = !dma_rd_cmd_empty;
    wire [26:0] dma_rd_addr_ui = dma_rd_cmd_rd_data;

    /* --- 5. DMA Read Data FIFO (ui_clk -> sys_clk, async FIFO) ------- */
    wire        dma_rd_data_fifo_full, dma_rd_data_fifo_empty;
    wire [7:0]  dma_rd_data_fifo_out;

    cdc_async_fifo #(.WIDTH(8), .DEPTH_LOG2(4)) u_dma_rd_data_fifo (
        .wr_clk   (ui_clk),
        .wr_rst_n (ui_rst_n),
        .wr_en    (dma_rd_valid_arb && !dma_rd_data_fifo_full),
        .wr_data  (dma_rd_data_128[7:0]),
        .wr_full  (dma_rd_data_fifo_full),
        .rd_clk   (sys_clk_bufg),
        .rd_rst_n (sys_rst_n),
        .rd_en    (!dma_rd_data_fifo_empty),
        .rd_data  (dma_rd_data_fifo_out),
        .rd_empty (dma_rd_data_fifo_empty)
    );

    /* --- 6. Updated DMA read data assignments ------------------------- */
    assign dma_rd_data_bytes = dma_rd_data_fifo_out;
    assign dma_rd_valid_sig  = !dma_rd_data_fifo_empty;

    /* --- 10. 2FF sync for init_calib_complete (ui_clk -> sys_clk) ----- */
    reg [1:0] calib_sync;
    always @(posedge sys_clk_bufg or negedge sys_rst_n)
        if (!sys_rst_n) calib_sync <= 2'b0;
        else            calib_sync <= {calib_sync[0], init_calib_complete};

    /* ------------------------------------------------------------------ */
    /* Module instantiations — sys_clk domain                             */
    /* ------------------------------------------------------------------ */

    usb_interface u_usb (
        .clk      (sys_clk_bufg),
        .rst_n    (sys_rst_n),
        .uart_rx  (uart_rx),
        .uart_tx  (uart_tx),
        .rx_data  (usb_rx_data),
        .rx_valid (usb_rx_valid),
        .rx_ready (usb_rx_ready),
        .tx_data  (usb_tx_data),
        .tx_valid (usb_tx_valid),
        .tx_ready (usb_tx_ready)
    );

    edgecoh_controller u_edgecoh (
        .clk           (sys_clk_bufg),
        .rst_n         (sys_rst_n),
        .rx_data       (usb_rx_data),
        .rx_valid      (usb_rx_valid),
        .rx_ready      (usb_rx_ready),
        .tx_data       (usb_tx_data),
        .tx_valid      (usb_tx_valid),
        .tx_ready      (usb_tx_ready),
        .nmc_start     (nmc_start),
        .nmc_op        (nmc_op),
        .nmc_table_base(nmc_table_base),
        .nmc_table_rows(nmc_table_rows),
        .nmc_table_cols(nmc_table_cols),
        .nmc_input_addr(nmc_input_addr),
        .nmc_input_len (nmc_input_len),
        .nmc_output_addr(nmc_output_addr),
        .nmc_done      (nmc_done_sys),
        .dma_wr_en     (dma_wr_en),
        .dma_wr_addr   (dma_wr_addr_wide),
        .dma_wr_data   (dma_wr_data),
        .dma_rd_en     (dma_rd_en),
        .dma_rd_addr   (dma_rd_addr_wide),
        .dma_rd_data   (dma_rd_data_bytes),
        .dma_rd_valid  (dma_rd_valid_sig),
        .barrier_ack   (barrier_ack)
    );

    /* ------------------------------------------------------------------ */
    /* Module instantiations — ui_clk domain                              */
    /* ------------------------------------------------------------------ */

    nmc_dispatch u_nmc (
        .clk             (ui_clk),
        .rst_n           (ui_rst_n),
        .nmc_start       (nmc_cmd_valid_ui),
        .nmc_op          (nmc_op_ui),
        .nmc_table_base  (nmc_table_base_ui),
        .nmc_table_rows  (nmc_table_rows_ui),
        .nmc_table_cols  (nmc_table_cols_ui),
        .nmc_input_addr  (nmc_input_addr_ui),
        .nmc_input_len   (nmc_input_len_ui),
        .nmc_output_addr (nmc_output_addr_ui),
        .nmc_done        (nmc_done_ui),
        .emb_start       (emb_start),
        .emb_table_base  (emb_table_base),
        .emb_embed_dim   (emb_embed_dim),
        .emb_indices_addr(emb_indices_addr),
        .emb_num_indices (emb_num_indices),
        .emb_output_addr (emb_output_addr),
        .emb_done        (emb_done),
        .mac_start       (mac_start),
        .mac_weight_addr (mac_weight_addr),
        .mac_input_addr  (mac_input_addr_d),
        .mac_output_addr (mac_output_addr_d),
        .mac_M           (mac_M),
        .mac_K           (mac_K),
        .mac_done        (mac_ctrl_done),
        .elt_start       (elt_start),
        .elt_input_addr  (elt_input_addr),
        .elt_output_addr (elt_output_addr),
        .elt_num_words   (elt_num_words),
        .elt_op          (elt_op),
        .elt_scale       (elt_scale),
        .elt_done        (elt_done)
    );

    embedding_lookup u_emb (
        .clk            (ui_clk),
        .rst_n          (ui_rst_n),
        .start          (emb_start),
        .table_base_addr(emb_table_base),
        .embed_dim      (emb_embed_dim),
        .indices_addr   (emb_indices_addr),
        .num_indices    (emb_num_indices),
        .output_addr    (emb_output_addr),
        .done           (emb_done),
        .mem_rd_en      (emb_mem_rd_en),
        .mem_rd_addr    (emb_mem_rd_addr),
        .mem_rd_data    (nmc_rd_data),
        .mem_rd_valid   (nmc_rd_valid),
        .mem_wr_en      (emb_mem_wr_en),
        .mem_wr_addr    (emb_mem_wr_addr),
        .mem_wr_data    (emb_mem_wr_data)
    );

    mac_controller u_mac_ctrl (
        .clk(ui_clk), .rst_n(ui_rst_n), .start(mac_start),
        .weight_addr(mac_weight_addr), .input_addr(mac_input_addr_d),
        .output_addr(mac_output_addr_d), .M(mac_M), .K(mac_K),
        .done(mac_ctrl_done),
        .mem_rd_en(mac_mem_rd_en), .mem_rd_addr(mac_mem_rd_addr),
        .mem_rd_data(mac_mem_rd_data), .mem_rd_valid(mac_mem_rd_valid),
        .mem_wr_en(mac_mem_wr_en), .mem_wr_addr(mac_mem_wr_addr),
        .mem_wr_data(mac_mem_wr_data),
        .mac_start(mac_start_array), .mac_load_a(mac_load_a), .mac_load_b(mac_load_b),
        .mac_row_a(mac_row_a), .mac_row_b(mac_row_b),
        .mac_result_0(mac_res0), .mac_result_1(mac_res1),
        .mac_result_2(mac_res2), .mac_result_3(mac_res3),
        .mac_result_4(mac_res4), .mac_result_5(mac_res5),
        .mac_result_6(mac_res6), .mac_result_7(mac_res7),
        .mac_done(mac_compute_done)
    );

    mac_array_8x8 u_mac (
        .clk(ui_clk), .rst_n(ui_rst_n), .start(mac_start_array),
        .load_a(mac_load_a), .load_b(mac_load_b),
        .row_a(mac_row_a), .row_b(mac_row_b),
        .result_0(mac_res0), .result_1(mac_res1),
        .result_2(mac_res2), .result_3(mac_res3),
        .result_4(mac_res4), .result_5(mac_res5),
        .result_6(mac_res6), .result_7(mac_res7),
        .done(mac_compute_done)
    );

    eltwise_controller u_elt_ctrl (
        .clk(ui_clk), .rst_n(ui_rst_n), .start(elt_start),
        .input_addr(elt_input_addr), .output_addr(elt_output_addr),
        .num_words(elt_num_words), .op(elt_op), .scale(elt_scale),
        .done(elt_done),
        .mem_rd_en(elt_mem_rd_en), .mem_rd_addr(elt_mem_rd_addr),
        .mem_rd_data(nmc_rd_data), .mem_rd_valid(nmc_rd_valid),
        .mem_wr_en(elt_mem_wr_en), .mem_wr_addr(elt_mem_wr_addr),
        .mem_wr_data(elt_mem_wr_data)
    );

    ddr2_arbiter u_arb (
        .clk             (ui_clk),
        .rst_n           (ui_rst_n),
        .app_addr        (app_addr),
        .app_cmd         (app_cmd),
        .app_en          (app_en),
        .app_wdf_data    (app_wdf_data),
        .app_wdf_wren    (app_wdf_wren),
        .app_wdf_end     (app_wdf_end),
        .app_rd_data     (app_rd_data),
        .app_rd_data_valid(app_rd_data_valid),
        .app_rdy         (app_rdy),
        .app_wdf_rdy     (app_wdf_rdy),
        /* NMC ports — driven by embedding_lookup */
        .nmc_rd_en       (nmc_rd_en),
        .nmc_rd_addr     (nmc_rd_addr),
        .nmc_rd_data     (nmc_rd_data),
        .nmc_rd_valid    (nmc_rd_valid),
        .nmc_wr_en       (nmc_wr_en),
        .nmc_wr_addr     (nmc_wr_addr),
        .nmc_wr_data     (nmc_wr_data),
        /* MAC ports — driven by mac_controller */
        .mac_rd_en       (mac_mem_rd_en),
        .mac_rd_addr     (mac_mem_rd_addr),
        .mac_rd_data     (mac_mem_rd_data),
        .mac_rd_valid    (mac_mem_rd_valid),
        .mac_wr_en       (mac_mem_wr_en),
        .mac_wr_addr     (mac_mem_wr_addr),
        .mac_wr_data     (mac_mem_wr_data),
        /* DMA ports — via CDC FIFOs from edgecoh_controller */
        .dma_rd_en       (dma_rd_en_ui),
        .dma_rd_addr     (dma_rd_addr_ui),
        .dma_rd_data     (dma_rd_data_128),
        .dma_rd_valid    (dma_rd_valid_arb),
        .dma_wr_en       (dma_wr_en_ui),
        .dma_wr_addr     (dma_wr_addr_ui),
        .dma_wr_byte     (dma_wr_data_ui)
    );

    /* ------------------------------------------------------------------ */
    /* LED status outputs                                                  */
    /* ------------------------------------------------------------------ */

    assign led[0] = calib_sync[1];       /* DDR2 calibration done (2FF)    */
    assign led[1] = nmc_start;           /* NMC operation in progress      */
    assign led[2] = barrier_ack;         /* EdgeCoh barrier acknowledged   */
    assign led[3] = 1'b1;               /* Heartbeat — design is running   */

endmodule
