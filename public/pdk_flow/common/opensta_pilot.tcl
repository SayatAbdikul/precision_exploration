# Phase 1 register-to-register timing harness for the mapped smoke block.
# Paths and period are explicit inputs so each constraint point has a distinct,
# reproducible invocation and report.
set liberty_path $::env(LIBERTY_PATH)
set netlist_path $::env(NETLIST_PATH)
set target_period_ns $::env(TARGET_PERIOD_NS)
set report_path $::env(REPORT_PATH)

read_liberty $liberty_path
read_verilog $netlist_path
link_design registered_smoke_arithmetic

create_clock -name clk -period $target_period_ns [get_ports clk]
set_clock_uncertainty 0.10 [get_clocks clk]
set_input_delay 0.20 -clock clk [get_ports {reset input_valid a[*] b[*] select_mul}]
set_output_delay 0.20 -clock clk [get_ports {output_valid result[*]}]
set_load 0.01 [get_ports {output_valid result[*]}]

check_setup -verbose >$report_path
report_checks -path_delay max -group_path_count 5 -endpoint_path_count 2 -fields {slew cap input_pins} -digits 6 >>$report_path
report_worst_slack -max -digits 6 >>$report_path
report_tns -max -digits 6 >>$report_path
report_clock_min_period [get_clocks clk] -digits 6 >>$report_path
