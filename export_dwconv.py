import numpy as np
import os
from golden_model import dwconv1d_int8_golden

cin = 8; cout = 8; len_in = 64; kernel = 3; stride = 1; pad_left = 1; dilation = 1
np.random.seed(99)

ifm = np.random.randint(-50, 50, (cin, len_in), dtype=np.int8)
weight = np.random.randint(-10, 10, (cout, 1, kernel), dtype=np.int8)
bias = np.random.randint(-100, 100, cout, dtype=np.int32)
mult = np.array([1 << 29] * cout, dtype=np.int32)
shift = np.array([30] * cout, dtype=np.uint8)

# 🔥 QUAY LẠI BẢN CHUẨN: Bật ReLU = True
golden_ofm = dwconv1d_int8_golden(ifm, weight, bias, mult, shift, stride, pad_left, dilation, relu_en=True)

def export_hex(filename, data, bit_width=8):
    os.makedirs('mem_export', exist_ok=True)
    mask = (1 << bit_width) - 1
    chars = bit_width // 4
    with open(f'mem_export/{filename}', 'w') as f:
        for val in data.flatten():
            # Đã tối ưu cú pháp f-string chuẩn Pythonic
            f.write(f"{(int(val) & mask):0{chars}X}\n")

export_hex('ifm.mem', ifm, bit_width=8)
export_hex('weight.mem', weight, bit_width=8)
export_hex('bias.mem', bias, bit_width=32)
export_hex('mult.mem', mult, bit_width=32)
export_hex('shift.mem', shift, bit_width=32)
export_hex('golden_dwconv.mem', golden_ofm, bit_width=8)
