import numpy as np
from golden_model import conv1d_int8_golden

def read_hex_file(filename, dtype, shape=None):
    with open(filename, 'r') as f:
        lines = [l.strip() for l in f if l.strip()]

    data = []
    x_count = 0
    for val_str in lines:
        if 'x' in val_str.lower():
            data.append(0)
            x_count += 1
            continue
        val = int(val_str, 16)
        if dtype == np.int8 and val >= 128:
            val -= 256
        elif dtype == np.int32 and val >= 0x80000000:
            val -= 0x100000000
        data.append(val)

    if x_count > 0:
        print(f"   ⚠️ {x_count} giá trị 'xx' trong {filename} → BRAM chưa được ghi!")

    arr = np.array(data, dtype=dtype)
    if shape:
        arr = arr.reshape(shape)
    return arr

if __name__ == '__main__':
    print("=====================================================")
    print("🕵️‍♂️ SO SÁNH RTL vs GOLDEN MODEL (FINAL AUDIT)")
    print("=====================================================")

    ifm    = read_hex_file('mem_export/ifm.mem',    np.int8,  (4, 64))
    weight = read_hex_file('mem_export/weight.mem', np.int8,  (8, 4, 3))
    bias   = read_hex_file('mem_export/bias.mem',   np.int32, (8,))
    mult   = read_hex_file('mem_export/mult.mem',   np.int32, (8,))
    shift  = read_hex_file('mem_export/shift.mem',  np.uint8, (8,))
    rtl_ofm = read_hex_file('mem_export/rtl_output.mem', np.int8, (8, 64))

    print("\n🧠 Đang tính toán Đáp Án Chuẩn (Python)...")
    golden_ofm = conv1d_int8_golden(
        ifm, weight, bias, mult, shift,
        stride=1, pad_left=1, dilation=1, relu_en=True
    )

    print("\n🔍 CHECK NHANH 8 GIÁ TRỊ ĐẦU TIÊN CỦA KÊNH 0:")
    print(f"   Golden Model: {golden_ofm[0, :8]}")
    print(f"   RTL Hardware: {rtl_ofm[0, :8]}")

    mismatch = np.sum(golden_ofm != rtl_ofm)
    print("-----------------------------------------------------")

    if mismatch == 0:
        print("🏆 MISMATCH = 0 — BIT-EXACT PASS 100%!")
        print("🌟 DỰ ÁN CỦA BẠN ĐÃ THÀNH CÔNG RỰC RỠ!")
    else:
        print(f"⚠️ Phát hiện {mismatch}/{golden_ofm.size} điểm sai lệch.")
    print("=====================================================")