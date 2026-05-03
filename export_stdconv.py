import numpy as np
import os
from golden_model import conv1d_int8_golden

# Cấu hình Standard Conv1D
cin = 4; cout = 8; len_in = 64; kernel = 3; stride = 1; pad_left = 1; dilation = 1
np.random.seed(42)

# Khởi tạo ma trận ngẫu nhiên
ifm = np.random.randint(-50, 50, (cin, len_in), dtype=np.int8)
weight = np.random.randint(-10, 10, (cout, cin, kernel), dtype=np.int8)
bias = np.random.randint(-100, 100, cout, dtype=np.int32)
mult = np.array([1 << 29] * cout, dtype=np.int32)
shift = np.array([30] * cout, dtype=np.uint8)

# Đáp án Vàng (Bật ReLU = True)
golden_ofm = conv1d_int8_golden(ifm, weight, bias, mult, shift, stride, pad_left, dilation, relu_en=True)

def export_hex(filename, data, bit_width=8):
    os.makedirs('mem_export', exist_ok=True)
    mask = (1 << bit_width) - 1
    chars = bit_width // 4
    with open(f'mem_export/{filename}', 'w') as f:
        for val in data.flatten():
            f.write(f"{(int(val) & mask):0{chars}X}\n")

export_hex('ifm.mem', ifm, 8)
export_hex('weight.mem', weight, 8)
export_hex('bias.mem', bias, 32)
export_hex('mult.mem', mult, 32)
export_hex('shift.mem', shift, 32)
export_hex('golden_stdconv.mem', golden_ofm, 8)
print("✅ [1/3] Đã tạo data Standard Conv1D (op=0) thành công!")
