foreach required {TOP NETLIST LIBERTY TECH_LEF CELLS_LEF OUTPUT_DEF} {
  if {![info exists ::env($required)]} {
    error "missing required environment variable $required"
  }
}

read_lef $::env(TECH_LEF)
read_lef $::env(CELLS_LEF)
read_liberty $::env(LIBERTY)
read_verilog $::env(NETLIST)
link_design $::env(TOP)

initialize_floorplan -utilization 35 -aspect_ratio 1.0 -core_space 5
place_pins -random
global_placement -density 0.45
detailed_placement
check_placement
report_design_area
report_checks -path_delay max -format full_clock_expanded
write_def $::env(OUTPUT_DEF)
