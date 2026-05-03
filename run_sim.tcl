open_project project_1/project_1.xpr
set mem_dir "/root/Bulbasaurs_CNN/mem_export"
set sim_dir "project_1/project_1.sim/sim_1/behav/xsim"

# Copy Data Lớp 0
file copy -force $mem_dir/ifm.mem $sim_dir/
file copy -force $mem_dir/weight_l0.mem $sim_dir/
file copy -force $mem_dir/bias_l0.mem $sim_dir/
file copy -force $mem_dir/mult_l0.mem $sim_dir/
file copy -force $mem_dir/shift_l0.mem $sim_dir/

# Copy Data Lớp 2
file copy -force $mem_dir/weight_l2.mem $sim_dir/
file copy -force $mem_dir/bias_l2.mem $sim_dir/
file copy -force $mem_dir/mult_l2.mem $sim_dir/
file copy -force $mem_dir/shift_l2.mem $sim_dir/

launch_simulation
run all
file copy -force $sim_dir/rtl_output.mem $mem_dir/
close_project
exit
