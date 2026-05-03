import numpy as np

def read_hex_file(filename, length):
    data = []
    with open(filename, 'r') as f:
        for line in f:
            line = line.strip()
            if line:
                val = int(line, 16)
                if val > 127: val -= 256
                data.append(val)
            if len(data) == length: break
    return np.array(data, dtype=np.int8)

rtl_ofm = read_hex_file('mem_export/rtl_output.mem', 512)
golden_ofm = read_hex_file('mem_export/golden_dwconv.mem', 512)

print("=====================================================")
print("🕵️‍♂️ SO SÁNH RTL vs GOLDEN MODEL (DEPTHWISE CONV1D)")
print("=====================================================")
mismatches = np.sum(rtl_ofm != golden_ofm)
if mismatches == 0:
    print("🏆 MISMATCH = 0 — DWCONV BIT-EXACT PASS 100%!")
else:
    print(f"❌ CÓ LỖI: Phát hiện {mismatches} điểm sai lệch!")
