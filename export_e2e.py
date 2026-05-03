import numpy as np
import os
from golden_model import conv1d_int8_golden, maxpool1d_int8_golden, dwconv1d_int8_golden

np.random.seed(42)
BASE = "mem_export"
os.makedirs(BASE, exist_ok=True)

def to_hex(val, bits):
    mask = (1 << bits) - 1
    chars = bits // 4
    return f"{(int(val) & mask):0{chars}X}"

def export_hex(filename, data, bit_width):
    path = os.path.join(BASE, filename)
    with open(path, "w") as f:
        for val in np.array(data).flatten():
            f.write(to_hex(val, bit_width) + "\n")
    count = np.array(data).size

# ─── LAYER 0: Standard Conv1D ───
L0 = dict(cin=4, cout=8, len_in=64, len_out=64, kernel=3, stride=1, dilation=1, pad_left=1, op=0, relu=True)
ifm = np.random.randint(-50, 50, (L0["cin"], L0["len_in"]), dtype=np.int8)
w0  = np.random.randint(-10, 10, (L0["cout"], L0["cin"], L0["kernel"]), dtype=np.int8)
b0  = np.random.randint(-100, 100, L0["cout"], dtype=np.int32)
m0  = np.full(L0["cout"], 1 << 29, dtype=np.int32)
sh0 = np.full(L0["cout"], 30, dtype=np.uint8)

export_hex("ifm.mem",       ifm, 8)
export_hex("weight_l0.mem", w0,  8)
export_hex("bias_l0.mem",   b0,  32)
export_hex("mult_l0.mem",   m0,  32)
export_hex("shift_l0.mem",  sh0, 32)

# ─── LAYER 1: MaxPool1D ───
L1 = dict(cin=8, cout=8, len_in=64, len_out=32, kernel=2, stride=2, dilation=1, pad_left=0, op=2, relu=False)

# ─── LAYER 2: Depthwise Conv1D ───
L2 = dict(cin=8, cout=8, len_in=32, len_out=32, kernel=3, stride=1, dilation=1, pad_left=1, op=1, relu=True)
np.random.seed(99)
w2  = np.random.randint(-10, 10, (L2["cout"], 1, L2["kernel"]), dtype=np.int8)
b2  = np.random.randint(-100, 100, L2["cout"], dtype=np.int32)
m2  = np.full(L2["cout"], 1 << 29, dtype=np.int32)
sh2 = np.full(L2["cout"], 30, dtype=np.uint8)

export_hex("weight_l2.mem", w2,  8)
export_hex("bias_l2.mem",   b2,  32)
export_hex("mult_l2.mem",   m2,  32)
export_hex("shift_l2.mem",  sh2, 32)

ofm0 = conv1d_int8_golden(ifm, w0, b0, m0, sh0, stride=L0["stride"], pad_left=L0["pad_left"], dilation=L0["dilation"], relu_en=L0["relu"])
ofm1 = maxpool1d_int8_golden(ofm0, kernel=L1["kernel"], stride=L1["stride"])
ofm2 = dwconv1d_int8_golden(ofm1, w2, b2, m2, sh2, stride=L2["stride"], pad_left=L2["pad_left"], dilation=L2["dilation"], relu_en=L2["relu"])

export_hex("golden_e2e.mem", ofm2, 8)
print("✅ TẠO LẠI DATA HOÀN TẤT!")
