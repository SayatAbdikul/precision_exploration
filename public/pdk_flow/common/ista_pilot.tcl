# Generic post-synthesis iSTA/iPA pilot. Paths are invocation metadata; reports
# carry the numerical RTL/library/constraint hashes. No parasitics are loaded.
set netlist_path [lindex $argv 0]
set liberty_path [lindex $argv 1]
set sdc_path [lindex $argv 2]
set report_directory [lindex $argv 3]
set top [lindex $argv 4]

set_design_workspace $report_directory
read_netlist $netlist_path
read_liberty [list $liberty_path]
link_design $top
read_sdc $sdc_path
report_timing -max_path 5
report_power -toggle 0.1
