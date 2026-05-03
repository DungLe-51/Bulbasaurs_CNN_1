import numpy as np
import os
from golden_model import maxpool1d_int8_golden

# Cấu hình MaxPool1D
cin = 8; len_in = 64; kernel = 2; stride = 2
np.random.seed(42)

# Khởi tạo IFM ngẫu nhiên (từ -128 đến 127)
ifm = np.random.randint(-128, 127, (cin, len_in), dtype=np.int8)

# Đáp án Vàng
golden_ofm = maxpool1d_int8_golden(ifm, kernel=kernel, stride=stride)

def export_hex(filename, data, bit_width=8):
    os.makedirs('mem_export', exist_ok=True)
    mask = (1 << bit_width) - 1
    chars = bit_width // 4
    with open(f'mem_export/{filename}', 'w') as f:
        for val in data.flatten():
            f.write(f"{(int(val) & mask):0{chars}X}\n")

export_hex('ifm.mem', ifm, 8)
export_hex('golden_maxpool.mem', golden_ofm, 8)

# Tạo các file rỗng cho Weight, Bias (để TCL script copy không báo lỗi)
export_hex('weight.mem', np.zeros(1), 8)
export_hex('bias.mem', np.zeros(1), 32)
export_hex('mult.mem', np.zeros(1), 32)
export_hex('shift.mem', np.zeros(1), 32)

print("✅ [1/3] Đã tạo data MaxPool1D (op=2) thành công!")
