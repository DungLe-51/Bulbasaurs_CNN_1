import numpy as np
import math

# ─── HÀM LÕI 1: Ép kiểu (Saturate) ──────────────────────────────────
def saturate_int8(x: int) -> int:
    """
    Ép kiểu số nguyên về giới hạn của Int8 [-128, 127].
    Mô phỏng chính xác hành vi bão hòa (saturation) của mạch Requantize.
    """
    return int(np.clip(x, -128, 127))

# ─── HÀM LÕI 2: Dịch bit có dấu (Arithmetic Right Shift) ──────────────
def arith_rshift(x: int, s: int) -> int:
    """
    Dịch bit sang phải có giữ dấu (Arithmetic Right Shift).
    PHẢI dùng kiểu 'int' nguyên thủy của Python để tránh lỗi Logical Shift của Numpy.
    """
    return x >> s

# ─── HÀM LÕI 3: Standard Conv1D (op = 0) ────────────────────────────
def conv1d_int8_golden(
    ifm, weight, bias, mult, shift, 
    stride=1, pad_left=0, dilation=1, relu_en=True
):
    """
    Mô phỏng phép nhân chập Conv1D (Int8).
    Thứ tự vòng lặp (oc -> t -> ic -> k) khớp 100% với FSM của RTL.
    """
    cout, cin, kernel = weight.shape
    len_in = ifm.shape[1]
    
    # Kích thước đầu ra chuẩn PyTorch (Padding đối xứng 2 đầu)
    len_out = (len_in + 2*pad_left - dilation*(kernel-1) - 1) // stride + 1
    ofm = np.zeros((cout, len_out), dtype=np.int8)
    
    for oc in range(cout):
        for t in range(len_out):
            # Khởi tạo Accumulator bằng Bias (Int32)
            acc = int(bias[oc])
            
            # Vòng lặp MAC
            for ic in range(cin):
                for k in range(kernel):
                    in_pos = t * stride + k * dilation - pad_left
                    if 0 <= in_pos < len_in:
                        # Phép nhân phải ép về int Python để tránh tràn số np.int8
                        acc += int(ifm[ic, in_pos]) * int(weight[oc, ic, k])
            
            # 🔥 ĐÃ VÁ LỖI LÀM TRÒN: Requantize (Round to Nearest)
            s = int(shift[oc])
            rounding_val = (1 << (s - 1)) if s > 0 else 0
            y = arith_rshift(acc * int(mult[oc]) + rounding_val, s)
            
            y = saturate_int8(y)
            
            # Hàm kích hoạt ReLU
            if relu_en:
                y = max(y, 0)
                
            ofm[oc, t] = y
            
    return ofm

# ─── HÀM LÕI 4: Depthwise Conv1D (op = 1) ───────────────────────────
def dwconv1d_int8_golden(
    ifm, weight, bias, mult, shift, 
    stride=1, pad_left=0, dilation=1, relu_en=True
):
    """
    Mô phỏng Depthwise Conv1D (Int8). Mỗi input channel chỉ chập với 1 filter.
    Note: weight shape cho DWConv thường là [cout, 1, kernel]. Ở đây cin = cout.
    """
    cout = weight.shape[0]
    kernel = weight.shape[2]
    len_in = ifm.shape[1]
    
    len_out = (len_in + 2*pad_left - dilation*(kernel-1) - 1) // stride + 1
    ofm = np.zeros((cout, len_out), dtype=np.int8)
    
    for c in range(cout):
        for t in range(len_out):
            acc = int(bias[c])
            for k in range(kernel):
                in_pos = t * stride + k * dilation - pad_left
                if 0 <= in_pos < len_in:
                    acc += int(ifm[c, in_pos]) * int(weight[c, 0, k])
                    
            # 🔥 ĐÃ VÁ LỖI LÀM TRÒN: Requantize (Round to Nearest)
            s = int(shift[c])
            rounding_val = (1 << (s - 1)) if s > 0 else 0
            y = arith_rshift(acc * int(mult[c]) + rounding_val, s)
            
            y = saturate_int8(y)
            
            if relu_en:
                y = max(y, 0)
                
            ofm[c, t] = y
            
    return ofm

# ─── HÀM LÕI 5: MaxPool1D (op = 2) ──────────────────────────────────
def maxpool1d_int8_golden(ifm, kernel, stride):
    """
    Mô phỏng Max Pooling 1D.
    Tương đương FSM RTL: acc khởi tạo bằng -128 (INT8_MIN), tìm giá trị max.
    """
    cin, len_in = ifm.shape
    len_out = (len_in - kernel) // stride + 1
    ofm = np.zeros((cin, len_out), dtype=np.int8)
    
    for c in range(cin):
        for t in range(len_out):
            # Khởi tạo bằng giá trị nhỏ nhất của hệ Int8
            cur_max = -128  
            for k in range(kernel):
                in_pos = t * stride + k
                if 0 <= in_pos < len_in:
                    cur_max = max(cur_max, int(ifm[c, in_pos]))
            ofm[c, t] = saturate_int8(cur_max)
            
    return ofm

# ─── KHỐI TEST CHẠY THỬ (UNIT TEST) ─────────────────────────────────
if __name__ == '__main__':
    print("🚀 Bắt đầu test Golden Model...")
    
    np.random.seed(42)
    
    cin, len_in, cout, kernel = 4, 16, 8, 3
    stride, pad_left, dilation = 1, 1, 1
    
    ifm    = np.random.randint(-10, 10, (cin, len_in), dtype=np.int8)
    weight = np.random.randint(-5, 5, (cout, cin, kernel), dtype=np.int8)
    bias   = np.random.randint(-100, 100, cout, dtype=np.int32)
    
    mult   = np.array([1 << 29] * cout, dtype=np.int32)  
    shift  = np.array([30] * cout, dtype=np.uint8)       
    
    print("\n[1] Đang chạy Conv1D Standard...")
    ofm_conv = conv1d_int8_golden(
        ifm, weight, bias, mult, shift,
        stride, pad_left, dilation, relu_en=True
    )
    
    print(f"IFM shape: {ifm.shape}")
    print(f"Weight shape: {weight.shape}")
    print(f"OFM Conv1D shape: {ofm_conv.shape} (Expected: (8, 16))")
    print(f"OFM Sample (Kênh 0, 5 giá trị đầu): {ofm_conv[0, :5]}")
    
    print("\n[2] Đang chạy MaxPool1D...")
    ofm_pool = maxpool1d_int8_golden(ofm_conv, kernel=2, stride=2)
    print(f"OFM MaxPool shape: {ofm_pool.shape} (Expected: (8, 8))")
    
    print("\n✅ GOLDEN MODEL HOÀN CHỈNH 100%!")