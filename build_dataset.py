import os
import glob
import h5py
import numpy as np
from scipy.signal import butter, sosfiltfilt
import random

# =====================================================================
# 1. CẤU HÌNH THÔNG SỐ (KHỚP 100% VỚI RTL VIRTEX-7)
# =====================================================================
DATA_DIR = "data_v1"
OUT_DIR = "dataset"
FS = 25.0
LOWCUT = 1.0
HIGHCUT = 12.0

CIN = 4          # Khớp desc_cin_i = 4
LEN_IN = 64      # Khớp desc_len_in_i = 64
HALF_LEN = LEN_IN // 2

# =====================================================================
# 2. BỘ LỌC TÍN HIỆU SÓNG (ZERO-PHASE)
# =====================================================================
def get_sos_filter(lowcut, highcut, fs, order=4):
    nyq = 0.5 * fs
    return butter(order, [lowcut/nyq, highcut/nyq], btype='band', output='sos')

def process_full_patch(patch, sos):
    """ Lọc Bandpass và Trừ Nhiễu chung (CMR) trên TOÀN BỘ Patch """
    # 1. Bandpass filter (Zero-phase để không làm trễ pha sóng)
    filtered = np.zeros_like(patch)
    for ch in range(patch.shape[0]):
        filtered[ch, :] = sosfiltfilt(sos, patch[ch, :])
    
    # 2. Common-mode rejection (Trừ đi nhiễu nền chung của toàn hệ thống cáp)
    cmr_patch = filtered - filtered.mean(axis=0, keepdims=True)
    return cmr_patch

def min_max_normalize(window):
    """ Chuẩn hóa Min-Max về [-1, 1] """
    w_min, w_max = window.min(), window.max()
    if w_max - w_min < 1e-10:
        return np.zeros_like(window)
    return 2.0 * ((window - w_min) / (w_max - w_min)) - 1.0

# =====================================================================
# 3. HÀM RÚT TRÍCH DỮ LIỆU ĐỘNG ĐẤT VÀ NHIỄU
# =====================================================================
def extract_windows(h5_path, sos):
    X_list, Y_list = [], []
    
    with h5py.File(h5_path, 'r') as f:
        patch = f['patch'][:]  # Dữ liệu sóng: (300, 4000)
        mask = f['mask'][:]    # Nhãn dán: (3, 300, 4000)
        
    # Tiền xử lý full patch trước khi cắt
    clean_patch = process_full_patch(patch, sos)
    
    # Quét qua các kênh (chỉ đến 300-CIN để khi gom 4 kênh không bị lố mảng)
    for ch in range(300 - CIN + 1):
        mask_p = mask[0, ch, :]
        mask_n = mask[2, ch, :]
        
        # --- CẮT SÓNG ĐỘNG ĐẤT (P-WAVE: NHÃN 1) ---
        peak_idx = np.argmax(mask_p)
        if mask_p[peak_idx] > 0.8 and HALF_LEN <= peak_idx <= 4000 - HALF_LEN:
            # Gom 4 kênh liên tiếp (cin=4), cắt độ dài 64 (len_in=64)
            window = clean_patch[ch : ch+CIN, peak_idx - HALF_LEN : peak_idx + HALF_LEN]
            X_list.append(min_max_normalize(window))
            Y_list.append(1)
            
        # --- CẮT SÓNG NHIỄU (NOISE: NHÃN 0) ---
        valid_noise = np.where((mask_n > 0.99) & 
                               (np.arange(4000) >= HALF_LEN) & 
                               (np.arange(4000) <= 4000 - HALF_LEN))[0]
        if len(valid_noise) > 0:
            noise_idx = random.choice(valid_noise)
            # Gom 4 kênh liên tiếp, cắt độ dài 64 tại vùng nhiễu nền
            window = clean_patch[ch : ch+CIN, noise_idx - HALF_LEN : noise_idx + HALF_LEN]
            X_list.append(min_max_normalize(window))
            Y_list.append(0)
            
    return X_list, Y_list

# =====================================================================
# 4. CHIA TẬP CHRONOLOGICAL VÀ XUẤT FILE NPY
# =====================================================================
if __name__ == '__main__':
    random.seed(42)
    sos = get_sos_filter(LOWCUT, HIGHCUT, FS)
    
    # Sắp xếp file theo thời gian (Tránh rò rỉ dữ liệu tương lai)
    all_files = sorted(glob.glob(os.path.join(DATA_DIR, "*.h5")))
    print(f"🔍 Tìm thấy {len(all_files)} file HDF5.")

    # Phân chia 70% Train, 15% Val, 15% Test
    splits = {
        'train': all_files[:52],
        'val': all_files[52:63],
        'test': all_files[63:]
    }
    
    for split_name, files in splits.items():
        print(f"\n🚀 Đang xử lý tập {split_name.upper()} ({len(files)} files)...")
        X_data, Y_data = [], []
        
        for f in files:
            x, y = extract_windows(f, sos)
            X_data.extend(x)
            Y_data.extend(y)
            
        # Ép về shape chuẩn xác (N, 4, 64) Float32 cho PyTorch CNN-1D
        X_npy = np.array(X_data, dtype=np.float32)
        Y_npy = np.array(Y_data, dtype=np.int64)
        
        np.save(os.path.join(OUT_DIR, f'X_{split_name}.npy'), X_npy)
        np.save(os.path.join(OUT_DIR, f'Y_{split_name}.npy'), Y_npy)
        
        print(f"✅ Hoàn thành tập {split_name}: Shape = {X_npy.shape}")
        print(f"   📊 P-wave (1): {np.sum(Y_npy==1)} mẫu | Noise (0): {np.sum(Y_npy==0)} mẫu")
        
    print("\n🎉 HOÀN TẤT 100%! DỮ LIỆU ĐÃ SẴN SÀNG CHO BỘ NÃO AI!")