# Generic launch/DUT/capture post-synthesis timing witness.  No parasitics are
# loaded, so results are Liberty cell-delay estimates only.
foreach required {TOP CLOCK_PORT LIBERTY_PATH NETLIST_PATH TARGET_PERIOD_NS REPORT_PATH} {
  if {![info exists ::env($required)]} { error "missing $required" }
}
read_liberty $::env(LIBERTY_PATH)
read_verilog $::env(NETLIST_PATH)
link_design $::env(TOP)
create_clock -name clk -period $::env(TARGET_PERIOD_NS) [get_ports $::env(CLOCK_PORT)]
set_clock_uncertainty 0.10 [get_clocks clk]
set data_inputs [remove_from_collection [all_inputs] [get_ports $::env(CLOCK_PORT)]]
set_input_delay 0.20 -clock clk $data_inputs
set_output_delay 0.20 -clock clk [all_outputs]
set_load 0.01 [all_outputs]
check_setup -verbose >$::env(REPORT_PATH)
report_checks -path_delay max -group_path_count 5 -endpoint_path_count 2 -fields {slew cap input_pins} -digits 6 >>$::env(REPORT_PATH)
report_worst_slack -max -digits 6 >>$::env(REPORT_PATH)
report_tns -max -digits 6 >>$::env(REPORT_PATH)
report_clock_min_period [get_clocks clk] -digits 6 >>$::env(REPORT_PATH)
