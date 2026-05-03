open_project project_1/project_1.xpr

reset_run synth_1
launch_runs synth_1 -jobs 8
wait_on_run synth_1

launch_runs impl_1 -jobs 8
wait_on_run impl_1

open_run impl_1
report_timing_summary -file timing_impl_50mhz.rpt

close_project
exit
