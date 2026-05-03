import torch
import torch.nn as nn
import torch.optim as optim

# ─── ĐỊNH NGHĨA KIẾN TRÚC MÔ HÌNH (TƯƠNG THÍCH 100% RTL) ──────────────
class CNN1D_DAS(nn.Module):
    def __init__(self, cin=4, num_classes=3):
        super().__init__()
        
        # Layer 1: Standard Conv1D -> BatchNorm -> ReLU
        # Tương đương op=0 trong FSM RTL
        self.conv1 = nn.Conv1d(cin, 8, kernel_size=3, padding=1, bias=True)
        self.bn1   = nn.BatchNorm1d(8)
        self.relu1 = nn.ReLU()
        
        # Layer 2: Depthwise Conv1D -> BatchNorm -> ReLU
        # Tương đương op=1 trong FSM RTL (groups = số kênh đầu vào)
        self.dw1   = nn.Conv1d(8, 8, kernel_size=3, padding=1, groups=8, bias=True)
        self.bn2   = nn.BatchNorm1d(8)
        self.relu2 = nn.ReLU()
        
        # Layer 3: MaxPool1D
        # Tương đương op=2 trong FSM RTL
        self.pool1 = nn.MaxPool1d(kernel_size=2, stride=2)
        
        # Layer 4: Pointwise Conv1D (k=1) để gom về 3 classes (P-wave, S-wave, Noise)
        self.pw1   = nn.Conv1d(8, num_classes, kernel_size=1, bias=True)

    def forward(self, x):
        x = self.relu1(self.bn1(self.conv1(x)))
        x = self.relu2(self.bn2(self.dw1(x)))
        x = self.pool1(x)
        x = self.pw1(x)
        return x  # Kích thước đầu ra: [Batch, 3_classes, Len_out]

# ─── KỊCH BẢN CHẠY & LƯU TRỮ ──────────────────────────────────────────
if __name__ == '__main__':
    print("🚀 Khởi tạo mô hình CNN1D_DAS...")
    # Khóa seed để kết quả ổn định qua các lần chạy
    torch.manual_seed(42)
    
    model = CNN1D_DAS(cin=4, num_classes=3)
    
    # 1. TẠO DỮ LIỆU GIẢ LẬP (Dummy Data)
    # Giả lập 16 đoạn tín hiệu DAS, mỗi đoạn có 4 kênh, độ dài 64 mẫu
    B, C, L_in = 16, 4, 64
    dummy_inputs = torch.randn(B, C, L_in)
    
    # Kích thước sau MaxPool(stride=2) sẽ giảm 1 nửa: L_out = 32
    # Mục tiêu: Phân loại từng điểm (Point-wise classification) thành 3 nhãn (0, 1, 2)
    L_out = 32
    dummy_targets = torch.randint(0, 3, (B, L_out))
    
    # 2. CẤU HÌNH HUẤN LUYỆN
    optimizer = optim.Adam(model.parameters(), lr=0.01)
    criterion = nn.CrossEntropyLoss()
    
    print("⏳ Đang Huấn luyện (Train) để làm xô lệch tham số BatchNorm...")
    model.train() # Kích hoạt chế độ Train
    
    # Chạy 5 Epochs để Model "học" và cập nhật Running Mean/Var
    for epoch in range(5):
        optimizer.zero_grad()
        outputs = model(dummy_inputs)
        loss = criterion(outputs, dummy_targets)
        loss.backward()
        optimizer.step()
        print(f"   Epoch {epoch+1}/5 | Loss: {loss.item():.4f}")
        
    print("✅ Training Dummy Data hoàn tất!")
    
    # 3. TỬ HUYỆT PHẦN CỨNG: BẮT BUỘC ĐÓNG BĂNG MÔ HÌNH
    model.eval()
    
    # 4. LƯU TRỌNG SỐ FLOAT32
    save_path = 'das_model_float.pth'
    torch.save(model.state_dict(), save_path)
    print(f"💾 Đã lưu trọng số (Đã chốt Running Stats) vào: {save_path}")