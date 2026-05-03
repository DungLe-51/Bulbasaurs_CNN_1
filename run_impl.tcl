open_project project_1/project_1.xpr

# Trục xuất hoàn toàn file quanitize.sv rác ra khỏi project
remove_files [get_files -quiet {/root/Bulbasaurs_CNN/CNN_AGAiN.srcs/sources_1/imports/new/quanitize.sv}]

# Add file constraint
add_files -fileset constrs_1 -norecurse constraints.xdc
set_property USED_IN_SYNTHESIS true [get_files constraints.xdc]
set_property USED_IN_IMPLEMENTATION true [get_files constraints.xdc]

# Chạy Synthesis và Implementation (Đã xóa tàn dư nên sẽ không bị kẹt nữa)
launch_runs synth_1 -jobs 8
wait_on_run synth_1

launch_runs impl_1 -jobs 8
wait_on_run impl_1

# Xuất báo cáo Timing
open_run impl_1
report_timing_summary -file timing_impl.rpt

close_project
exit
