`timescale 1ns/1ps
module tb_cnn1d_core_top;
    logic clk = 0; logic rst_n = 0; always #5 clk = ~clk; 
    logic start_i = 0; logic busy_o, done_o, error_o; logic [3:0] state_dbg_o;
    logic desc_we_i = 0; logic [1:0] desc_op_i = 0;
    logic desc_ifm_bank_i = 0, desc_ofm_bank_i = 1, desc_relu_en_i = 0;
    logic [15:0] desc_cin_i = 0, desc_cout_i = 0, desc_len_in_i = 0, desc_len_out_i = 0;
    logic [7:0] desc_kernel_i = 0, desc_stride_i = 0, desc_dilation_i = 0, desc_pad_left_i = 0;
    logic [15:0] desc_ifm_base_i = 0, desc_ofm_base_i = 0, desc_wgt_base_i = 0;
    logic [15:0] desc_param_base_i = 0;

    logic host_we_i = 0, host_re_i = 0; logic [3:0] host_sel_i = 0;
    logic [15:0] host_addr_i = 0; logic [31:0] host_wdata_i = 0; logic [31:0] host_rdata_o;

    cnn1d_core_top #(.DATA_W(8), .WGT_W(8), .ACC_W(32), .ADDR_W(16), .PARAM_ADDR_W(12), .LEN_W(16), .CH_W(16), .K_W(8)) u_dut (.*); 

    // Các mảng chứa file riêng lẻ
    logic [7:0] mem_ifm [0:4095];
    logic [7:0] w0_mem [0:4095]; logic [31:0] b0_mem [0:255]; logic [31:0] m0_mem [0:255]; logic [31:0] sh0_mem[0:255];
    logic [7:0] w2_mem [0:4095]; logic [31:0] b2_mem [0:255]; logic [31:0] m2_mem [0:255]; logic [31:0] sh2_mem[0:255];
    integer i, file_out;

    task write_host(input [3:0] sel, input [15:0] addr, input [31:0] data);
        @(posedge clk); #1; host_sel_i = sel; host_addr_i = addr; host_wdata_i = data; host_we_i = 1; 
        @(posedge clk); #1; host_we_i = 0;
    endtask

    task switch_core_bank(input logic bank);
        @(posedge clk); #1; desc_ifm_bank_i = bank; desc_we_i = 1;
        @(posedge clk); #1; desc_we_i = 0;
    endtask

    task run_layer();
        @(posedge clk); #1; desc_we_i = 1; @(posedge clk); #1; desc_we_i = 0; 
        @(posedge clk); #1; start_i = 1; @(posedge clk); #1; start_i = 0;
        repeat(10) @(posedge clk); wait(done_o == 1 || error_o == 1);
        if (error_o) begin $display("❌ LỖI FSM!"); $finish; end
        repeat(10) @(posedge clk);
    endtask

    initial begin
        // Đọc từng file riêng rẽ
        $readmemh("ifm.mem", mem_ifm);
        $readmemh("weight_l0.mem", w0_mem); $readmemh("bias_l0.mem", b0_mem); $readmemh("mult_l0.mem", m0_mem); $readmemh("shift_l0.mem", sh0_mem);
        $readmemh("weight_l2.mem", w2_mem); $readmemh("bias_l2.mem", b2_mem); $readmemh("mult_l2.mem", m2_mem); $readmemh("shift_l2.mem", sh2_mem);

        rst_n = 0; repeat(10) @(posedge clk); rst_n = 1; repeat(5) @(posedge clk);

        // BƯỚC 0: Nạp IFM và Param của Layer 0
        switch_core_bank(1); 
        for (i = 0; i < 256; i++) write_host(4'd0, i[15:0], {24'd0, mem_ifm[i]});
        for (i = 0; i < 96; i++)  write_host(4'd2, i[15:0], {24'd0, w0_mem[i]});
        for (i = 0; i < 8; i++) begin
            write_host(4'd3, i[15:0], b0_mem[i]); write_host(4'd4, i[15:0], m0_mem[i]); write_host(4'd5, i[15:0], sh0_mem[i]); 
        end

        // 🔥 LAYER 0: STANDARD CONV1D
        $display("🚀 Đang chạy Layer 0...");
        desc_ifm_bank_i = 0; desc_ofm_bank_i = 1;
        desc_op_i = 2'd0; desc_cin_i = 4; desc_cout_i = 8; desc_len_in_i = 64; desc_len_out_i = 64; 
        desc_kernel_i = 3; desc_stride_i = 1; desc_dilation_i = 1; desc_pad_left_i = 1; desc_relu_en_i = 1; 
        desc_wgt_base_i = 0; desc_param_base_i = 0; 
        run_layer();

        // 🔄 DMA: Copy OFM -> IFM (Chuẩn bị Lớp 1)
        switch_core_bank(1);
        $display("🔄 DMA: Copy Lớp 0 -> IFM...");
        for (i = 0; i < 512; i++) write_host(4'd0, i[15:0], {24'd0, u_dut.u_ofm_bank.u_pong_bram.mem[i]});

        // 🔥 LAYER 1: MAXPOOL1D
        $display("🚀 Đang chạy Layer 1...");
        desc_ifm_bank_i = 0; desc_ofm_bank_i = 1;
        desc_op_i = 2'd2; desc_cin_i = 8; desc_cout_i = 8; desc_len_in_i = 64; desc_len_out_i = 32; 
        desc_kernel_i = 2; desc_stride_i = 2; desc_dilation_i = 1; desc_pad_left_i = 0; desc_relu_en_i = 0;
        desc_wgt_base_i = 0; desc_param_base_i = 0; 
        run_layer();

        // 🔄 DMA: Copy Lớp 1 -> IFM & Dọn Rác & **NẠP TRỌNG SỐ MỚI CHO LỚP 2**
        switch_core_bank(1);
        $display("🔄 DMA: Copy Lớp 1 -> IFM (Dọn rác) & Reload Params Lớp 2...");
        for (i = 0; i < 512; i++) begin
            if (i < 256) write_host(4'd0, i[15:0], {24'd0, u_dut.u_ofm_bank.u_pong_bram.mem[i]});
            else         write_host(4'd0, i[15:0], 32'd0);
        end
        // Nạp đè Trọng số Lớp 2 vào mốc 0 (Ghi đè Lớp 0 cũ)
        for (i = 0; i < 24; i++) write_host(4'd2, i[15:0], {24'd0, w2_mem[i]});
        for (i = 0; i < 8; i++) begin
            write_host(4'd3, i[15:0], b2_mem[i]); write_host(4'd4, i[15:0], m2_mem[i]); write_host(4'd5, i[15:0], sh2_mem[i]); 
        end

        // 🔥 LAYER 2: DEPTHWISE CONV1D
        $display("🚀 Đang chạy Layer 2...");
        desc_ifm_bank_i = 0; desc_ofm_bank_i = 1;
        desc_op_i = 2'd1; desc_cin_i = 8; desc_cout_i = 8; desc_len_in_i = 32; desc_len_out_i = 32; 
        desc_kernel_i = 3; desc_stride_i = 1; desc_dilation_i = 1; desc_pad_left_i = 1; desc_relu_en_i = 1;
        desc_wgt_base_i = 0; desc_param_base_i = 0; // Đọc lại từ mốc 0 vì đã nạp đè
        run_layer();

        // XUẤT KẾT QUẢ CUỐI CÙNG
        file_out = $fopen("rtl_output.mem", "w");
        for (i = 0; i < 256; i++) begin $fdisplay(file_out, "%02X", u_dut.u_ofm_bank.u_pong_bram.mem[i]); end
        $fclose(file_out); 
        $display("✅ Hoàn thành toàn bộ quy trình AI System!");
        $finish;
    end 
endmodule