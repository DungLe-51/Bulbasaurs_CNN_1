import numpy as np

def read_hex_file(filename, length):
    data = []
    with open(filename, 'r') as f:
        for line in f:
            line = line.strip().lower()
            if line:
                if 'x' in line:
                    val = 0
                else:
                    val = int(line, 16)
                    if val > 127: val -= 256
                data.append(val)
            if len(data) == length: break
    return np.array(data, dtype=np.int8)

rtl_ofm = read_hex_file('mem_export/rtl_output.mem', 512)
golden_ofm = read_hex_file('mem_export/golden_stdconv.mem', 512)

print("\n=====================================================")
print("🕵️‍♂️ TÒA ÁN: RTL vs GOLDEN MODEL (STANDARD CONV1D - op=0)")
print("=====================================================")
mismatches = np.sum(rtl_ofm != golden_ofm)
if mismatches == 0:
    print("🏆 MISMATCH = 0 — STANDARD CONV1D BIT-EXACT PASS 100%!")
else:
    print(f"❌ CÓ LỖI: Phát hiện {mismatches} điểm sai lệch!")
