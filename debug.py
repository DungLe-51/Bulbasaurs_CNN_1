import numpy as np

def read_hex_file(filename, length):
    data = []
    with open(filename, 'r') as f:
        for line in f:
            line = line.strip().lower()
            if line:
                if 'x' in line:
                    data.append(-999) # Ký hiệu cho mảng chưa được khởi tạo
                else:
                    val = int(line, 16)
                    if val > 127: val -= 256
                    data.append(val)
            if len(data) == length: break
    return np.array(data, dtype=np.int16) # Dùng int16 để chứa số -999

# Đọc 256 bytes của Output Lớp 2 (DWConv) -> Kích thước (8 kênh, 32 t)
try:
    rtl_ofm = read_hex_file('mem_export/rtl_output.mem', 256).reshape(8, 32)
    golden_ofm = read_hex_file('mem_export/golden_e2e.mem', 256).reshape(8, 32)
    
    print("\n=====================================================")
    print("🔍 BÁO CÁO GIẢI PHẪU DỮ LIỆU (DEEP MISMATCH ANALYSIS)")
    print("=====================================================")
    
    errors = 0
    zero_count = 0
    uninit_count = 0

    for c in range(8):
        for t in range(32):
            val_rtl = rtl_ofm[c, t]
            val_gld = golden_ofm[c, t]
            
            if val_rtl == -999:
                uninit_count += 1
            if val_rtl == 0 and val_gld != 0:
                zero_count += 1
                
            if val_rtl != val_gld:
                if errors < 15: # Chỉ in 15 lỗi đầu tiên để tránh trôi màn hình
                    print(f"❌ Kênh [Ch:{c}][T:{t:2}] | RTL sinh ra: {val_rtl:4} | Đáp án Vàng: {val_gld:4}")
                errors += 1

    print("-----------------------------------------------------")
    print(f"Tổng số lỗi          : {errors} / 256")
    print(f"Số ô nhớ bị trống (X): {uninit_count} (Cảnh báo BRAM chưa ghi)")
    print(f"Số lỗi do RTL trả về 0: {zero_count} (Cảnh báo mất data đầu vào)")
    
    if uninit_count == 256:
        print("\n🚨 KẾT LUẬN: Mạch của bạn chưa hề ghi một byte nào vào BRAM ở Lớp 2!")
    elif zero_count > 80:
        print("\n🚨 KẾT LUẬN: Lớp 2 toàn nhân với số 0. Lỗi 100% nằm ở Host DMA chép sai địa chỉ!")
    else:
        print("\n🚨 KẾT LUẬN: Bị tràn Padding hoặc sai số bộ Requantize.")

except Exception as e:
    print(f"Lỗi khi chạy Radar: {e}")
