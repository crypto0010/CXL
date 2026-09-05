/* Behavioural stand-ins for Xilinx primitives so top.v can be linted and
 * simulated under Icarus.  NOT for synthesis. */
`timescale 1ns / 1ps
module IBUF (input wire I, output wire O); assign O = I; endmodule
module BUFG (input wire I, output wire O); assign O = I; endmodule
module MMCME2_BASE #(parameter BANDWIDTH="OPTIMIZED", parameter CLKFBOUT_MULT_F=10.0, parameter CLKFBOUT_PHASE=0.0,
    parameter CLKIN1_PERIOD=10.0, parameter CLKOUT0_DIVIDE_F=5.0, parameter CLKOUT1_DIVIDE=1, parameter CLKOUT2_DIVIDE=1,
    parameter CLKOUT3_DIVIDE=1, parameter CLKOUT4_DIVIDE=1, parameter CLKOUT5_DIVIDE=1, parameter CLKOUT6_DIVIDE=1,
    parameter CLKOUT0_DUTY_CYCLE=0.5, parameter CLKOUT1_DUTY_CYCLE=0.5, parameter CLKOUT2_DUTY_CYCLE=0.5,
    parameter CLKOUT3_DUTY_CYCLE=0.5, parameter CLKOUT4_DUTY_CYCLE=0.5, parameter CLKOUT5_DUTY_CYCLE=0.5, parameter CLKOUT6_DUTY_CYCLE=0.5,
    parameter CLKOUT0_PHASE=0.0, parameter CLKOUT1_PHASE=0.0, parameter CLKOUT2_PHASE=0.0, parameter CLKOUT3_PHASE=0.0,
    parameter CLKOUT4_PHASE=0.0, parameter CLKOUT5_PHASE=0.0, parameter CLKOUT6_PHASE=0.0, parameter CLKOUT4_CASCADE="FALSE",
    parameter DIVCLK_DIVIDE=1, parameter REF_JITTER1=0.0, parameter STARTUP_WAIT="FALSE")
   (input wire CLKIN1, input wire CLKFBIN, input wire RST, input wire PWRDWN,
    output wire CLKOUT0, output wire CLKOUT0B, output wire CLKOUT1, output wire CLKOUT1B, output wire CLKOUT2, output wire CLKOUT2B,
    output wire CLKOUT3, output wire CLKOUT3B, output wire CLKOUT4, output wire CLKOUT5, output wire CLKOUT6,
    output wire CLKFBOUT, output wire CLKFBOUTB, output wire LOCKED);
    assign CLKOUT0 = CLKIN1; assign CLKOUT1 = CLKIN1; assign CLKOUT2 = CLKIN1; assign CLKOUT3 = CLKIN1;
    assign CLKOUT4 = CLKIN1; assign CLKOUT5 = CLKIN1; assign CLKOUT6 = CLKIN1; assign CLKFBOUT = CLKFBIN;
    assign CLKOUT0B = ~CLKIN1; assign CLKOUT1B = ~CLKIN1; assign CLKOUT2B = ~CLKIN1; assign CLKOUT3B = ~CLKIN1; assign CLKFBOUTB = ~CLKFBIN;
    assign LOCKED = ~RST;
endmodule
