import os
import torch
import numpy as np
import math
import json
from train import CNN1D_DAS

# ─── HÀM BỔ TRỢ 1: ĐỊNH DẠNG HEXADECIMAL (BÙ 2) ─────────────────────────
def to_hex(val, bits):
    """Chuyển số nguyên (có dấu) thành chuỗi Hex chuẩn FPGA"""
    val = int(val)
    if bits == 8:
        return f"{val & 0xFF:02X}"
    elif bits == 32:
        return f"{val & 0xFFFFFFFF:08X}"

# ─── HÀM BỔ TRỢ 2: TFLITE DYNAMIC MULTIPLIER & SHIFT ────────────────────
def quantize_multiplier_safe(eff_scale):
    """
    Thuật toán TFLite: Đưa eff_scale về dạng (mult * 2^-shift).
    Bảo vệ phần cứng: Giới hạn shift trong khoảng [0, 31].
    """
    if eff_scale == 0:
        return 0, 0
    
    # frexp đưa số về dạng: significand * 2^exp (significand thuộc [0.5, 1.0))
    significand, exp = math.frexp(eff_scale)
    
    # Kéo significand lên int32 (nhân với 2^31)
    mult_raw = int(round(significand * (1 << 31)))
    shift_raw = 31 - exp
    
    # Clamp bảo vệ giới hạn thanh ghi 5-bit của chip (0 - 31)
    if shift_raw < 0:
        mult_raw >>= (-shift_raw)
        shift_raw = 0
    elif shift_raw > 31:
        mult_raw >>= (shift_raw - 31)
        shift_raw = 31
        
    return mult_raw, shift_raw

# ─── LUỒNG CHẠY CHÍNH ──────────────────────────────────────────────────
if __name__ == '__main__':
    print("🚀 Bắt đầu Quantize & Export .mem (Chuẩn TFLite)...")
    out_dir = "mem_export"
    os.makedirs(out_dir, exist_ok=True)
    
    # 1. LOAD MÔ HÌNH VÀ FOLD BATCHNORM
    model = CNN1D_DAS(cin=4, num_classes=3)
    model.load_state_dict(torch.load('das_model_float.pth'))
    model.eval() # BẮT BUỘC ĐỂ KHÓA RUNNING STATS
    
    w_conv = model.conv1.weight.detach().numpy()
    b_conv = model.conv1.bias.detach().numpy()   
    gamma  = model.bn1.weight.detach().numpy()
    beta   = model.bn1.bias.detach().numpy()
    mean   = model.bn1.running_mean.numpy()
    var    = model.bn1.running_var.numpy()
    eps    = model.bn1.eps

    std = np.sqrt(var + eps)
    bn_scale = gamma / std
    
    w_fold = w_conv * bn_scale[:, None, None]
    b_fold = (b_conv - mean) * bn_scale + beta

    # 2. QUANTIZATION (PER-CHANNEL INT8 / INT32)
    abs_max = np.max(np.abs(w_fold), axis=(1, 2))
    w_scale = abs_max / 127.0
    
    w_int8 = np.clip(np.round(w_fold / w_scale[:, None, None]), -128, 127).astype(np.int8)
    
    input_scale = 1.0  # Giả định cho test
    output_scale = 1.0 # Giả định cho test
    eff_scale = (input_scale * w_scale) / output_scale
    
    b_int32 = np.round(b_fold / (input_scale * w_scale)).astype(np.int32)
    
    mult_int32 = np.zeros_like(b_int32)
    shift_uint8 = np.zeros(len(b_int32), dtype=np.uint8)
    
    for i in range(len(b_int32)):
        m, s = quantize_multiplier_safe(eff_scale[i])
        mult_int32[i] = m
        shift_uint8[i] = s

    # 3. TẠO LAYER DESCRIPTOR JSON (Giúp ích cực lớn cho Testbench)
    layer_desc = {
        "name": "conv1",
        "op": 0,
        "cin": 4, "cout": 8, "kernel": 3,
        "stride": 1, "dilation": 1, "pad_left": 1,
        "relu_en": 1,
        "len_in": 64, "len_out": 64,
        "wgt_base": 0, "param_base": 0,
        "ifm_bank": 0, "ofm_bank": 1
    }
    with open(f"{out_dir}/layer_desc.json", "w") as f:
        json.dump(layer_desc, f, indent=4)

    # 4. XUẤT FILE HEX .MEM
    np.random.seed(42)
    ifm_int8 = np.random.randint(-50, 50, size=(4, 64), dtype=np.int8)

    with open(f"{out_dir}/weight.mem", "w") as f:
        for val in w_int8.flatten(): f.write(to_hex(val, 8) + "\n")
            
    with open(f"{out_dir}/bias.mem", "w") as f:
        for val in b_int32: f.write(to_hex(val, 32) + "\n")
            
    with open(f"{out_dir}/mult.mem", "w") as f:
        for val in mult_int32: f.write(to_hex(val, 32) + "\n")
            
    with open(f"{out_dir}/shift.mem", "w") as f:
        for val in shift_uint8: f.write(to_hex(val, 32) + "\n")
            
    with open(f"{out_dir}/ifm.mem", "w") as f:
        for val in ifm_int8.flatten(): f.write(to_hex(val, 8) + "\n")

    print(f"🎉 HOÀN TẤT GIAI ĐOẠN 3! Data đã lưu chuẩn xác vào {out_dir}/")