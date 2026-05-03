import os
import torch
import torch.nn as nn
import numpy as np

# 1. Định nghĩa lại Core để load trọng số
class CNN1D_FPGA_Core(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv1d(4, 8, kernel_size=3, padding=1, bias=True)
        self.bn1 = nn.BatchNorm1d(8)
        self.relu1 = nn.ReLU()
        self.pool1 = nn.MaxPool1d(kernel_size=2, stride=2)
        self.dwconv = nn.Conv1d(8, 8, kernel_size=3, padding=1, groups=8, bias=True)
        self.bn2 = nn.BatchNorm1d(8)
        self.relu2 = nn.ReLU()

def export_hex(filename, data, bit_width=8):
    os.makedirs('mem_export', exist_ok=True)
    path = os.path.join('mem_export', filename)
    with open(path, 'w') as f:
        for val in data.flatten():
            if bit_width == 8:
                val = int(val) & 0xFF
                f.write(f"{val:02x}\n")
            elif bit_width == 32:
                val = int(val) & 0xFFFFFFFF
                f.write(f"{val:08x}\n")
    print(f"💾 Đã xuất: {path}")

# ==========================================================
# GỘP BATCHNORM VÀ LƯỢNG TỬ HÓA VỀ INT8/INT32
# ==========================================================
def fold_and_quantize(conv, bn, layer_name):
    # Lấy thông số từ PyTorch
    w = conv.weight.detach().numpy()
    b = conv.bias.detach().numpy() if conv.bias is not None else np.zeros(w.shape[0])
    
    gamma = bn.weight.detach().numpy()
    beta = bn.bias.detach().numpy()
    mean = bn.running_mean.numpy()
    var = bn.running_var.numpy()
    eps = bn.eps

    # Tính hệ số gộp (Fold)
    scale = gamma / np.sqrt(var + eps)
    
    # Gộp vào Weight và Bias
    w_folded = w * scale[:, None, None]
    b_folded = (b - mean) * scale + beta

    # Lượng tử hóa Weight (Float -> INT8)
    w_max = np.max(np.abs(w_folded))
    w_scale = 127.0 / w_max if w_max > 0 else 1.0
    w_int8 = np.clip(np.round(w_folded * w_scale), -128, 127).astype(np.int8)

    # Lượng tử hóa Bias (Float -> INT32)
    # Trong thực tế phải scale theo IFM, ở đây ta giả lập tỷ lệ an toàn cho RTL
    b_int32 = np.clip(np.round(b_folded * 1000), -2147483648, 2147483647).astype(np.int32)

    # Sinh Mult và Shift giả lập để chống tràn số (Mạch RTL cần cái này)
    mult = np.array([1 << 29] * w.shape[0], dtype=np.int32)
    shift = np.array([30] * w.shape[0], dtype=np.int32)

    # Xuất ra file
    export_hex(f'weight_{layer_name}.mem', w_int8, 8)
    export_hex(f'bias_{layer_name}.mem', b_int32, 32)
    export_hex(f'mult_{layer_name}.mem', mult, 32)
    export_hex(f'shift_{layer_name}.mem', shift, 32)
    
    return w_int8, b_int32, mult, shift

if __name__ == '__main__':
    print("🚀 BẮT ĐẦU TRÍCH XUẤT TRI THỨC TỪ AI XUỐNG PHẦN CỨNG...")
    
    # 1. Load trọng số đã học (Bỏ qua phần Head CPU)
    core = CNN1D_FPGA_Core()
    state_dict = torch.load('best_model_weights.pth')
    
    # Khớp tên key từ Full_System xuống FPGA_Core
    core_state_dict = {k.replace('fpga_core.', ''): v for k, v in state_dict.items() if 'fpga_core' in k}
    core.load_state_dict(core_state_dict)
    core.eval() # Bắt buộc để chốt Running Mean/Var
    
    # 2. Xử lý Layer 0 (Conv1D)
    w0, b0, m0, s0 = fold_and_quantize(core.conv1, core.bn1, 'l0')
    
    # 3. Xử lý Layer 2 (DWConv1D)
    w2, b2, m2, s2 = fold_and_quantize(core.dwconv, core.bn2, 'l2')

    # 4. Lấy 1 SÓNG ĐỊA CHẤN THẬT TỪ TẬP TEST
    print("\n🌊 Lấy 1 mẫu sóng Động Đất thực tế (P-Wave)...")
    X_test = np.load("dataset/X_test.npy")
    Y_test = np.load("dataset/Y_test.npy")
    
    # Tìm mẫu đầu tiên có nhãn 1 (Động đất)
    idx = np.where(Y_test == 1)[0][0]
    real_wave = X_test[idx] # Shape: (4, 64)
    
    # Ép Float32 [-1, 1] về INT8 [-128, 127]
    ifm_int8 = np.clip(np.round(real_wave * 127), -128, 127).astype(np.int8)
    export_hex('ifm.mem', ifm_int8, 8)
    
    # 5. Dùng Golden Model Python để tính toán trước kết quả đích
    from golden_model import conv1d_int8_golden, maxpool1d_int8_golden, dwconv1d_int8_golden
    
    print("\n🧠 Chạy Golden Model Python để tạo đáp án...")
    out_l0 = conv1d_int8_golden(ifm_int8, w0, b0, m0, s0, stride=1, pad_left=1, relu_en=True)
    out_l1 = maxpool1d_int8_golden(out_l0, kernel=2, stride=2)
    out_l2 = dwconv1d_int8_golden(out_l1, w2, b2, m2, s2, stride=1, pad_left=1, relu_en=True)
    
    export_hex('golden_e2e.mem', out_l2, 8)
    
    print("🎉 HOÀN TẤT! SẴN SÀNG CHẠY VIVADO VỚI SÓNG ĐỊA CHẤN THẬT!")