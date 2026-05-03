import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader

# ==========================================================
# 1. KIẾN TRÚC MẠNG TƯƠNG THÍCH 100% VỚI RTL VIRTEX-7
# ==========================================================
class CNN1D_FPGA_Core(nn.Module):
    """ Nửa này sẽ được lượng tử hóa và nạp xuống chip FPGA. 
        Đầu vào: (Batch, 4, 64) """
    def __init__(self):
        super().__init__()
        # Layer 0: Conv1D (cin=4, cout=8, k=3, pad=1)
        self.conv1 = nn.Conv1d(4, 8, kernel_size=3, padding=1, bias=True)
        self.bn1 = nn.BatchNorm1d(8)
        self.relu1 = nn.ReLU()
        
        # Layer 1: MaxPool1D (k=2, stride=2)
        # Độ dài 64 bị giảm còn 32
        self.pool1 = nn.MaxPool1d(kernel_size=2, stride=2)
        
        # Layer 2: Depthwise Conv1D (cin=8, cout=8, k=3, pad=1)
        self.dwconv = nn.Conv1d(8, 8, kernel_size=3, padding=1, groups=8, bias=True)
        self.bn2 = nn.BatchNorm1d(8)
        self.relu2 = nn.ReLU()

    def forward(self, x):
        x = self.relu1(self.bn1(self.conv1(x)))
        x = self.pool1(x)
        x = self.relu2(self.bn2(self.dwconv(x)))
        return x  # Đầu ra: (Batch, 8, 32)

class Full_System(nn.Module):
    """ Tổng thể hệ thống: FPGA trích xuất + CPU phân loại """
    def __init__(self):
        super().__init__()
        self.fpga_core = CNN1D_FPGA_Core()
        
        # Nửa này chạy trên CPU (Chỉ tốn vài chục phép MAC)
        self.cpu_head = nn.Sequential(
            nn.AdaptiveAvgPool1d(1), # Ép ma trận (8, 32) thành vector (8, 1)
            nn.Flatten(),            # San phẳng thành mảng 1D (8,)
            nn.Linear(8, 2)          # Phân loại: 0 (Noise) và 1 (P-Wave)
        )

    def forward(self, x):
        features = self.fpga_core(x)
        logits = self.cpu_head(features)
        return logits

# ==========================================================
# 2. HÀM LOAD DATASET ĐÃ BÀO
# ==========================================================
def load_data(split, batch_size=128, shuffle=True):
    print(f"📦 Đang tải dữ liệu {split.upper()}...")
    X_path = os.path.join("dataset", f"X_{split}.npy")
    Y_path = os.path.join("dataset", f"Y_{split}.npy")
    
    if not os.path.exists(X_path):
        raise FileNotFoundError(f"Không tìm thấy {X_path}. Đã chạy build_dataset.py chưa?")
        
    X = np.load(X_path)
    Y = np.load(Y_path)
    
    # PyTorch yêu cầu input dạng FloatTensor và Label dạng LongTensor
    tensor_x = torch.FloatTensor(X)
    tensor_y = torch.LongTensor(Y)
    
    dataset = TensorDataset(tensor_x, tensor_y)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)

# ==========================================================
# 3. VÒNG LẶP HUẤN LUYỆN (TRAINING LOOP)
# ==========================================================
if __name__ == '__main__':
    # Khóa hạt giống để đảm bảo chạy lại luôn ra cùng 1 kết quả
    torch.manual_seed(42)
    
    # Tải dữ liệu thật vào Loader
    train_loader = load_data('train', batch_size=256, shuffle=True)
    val_loader   = load_data('val', batch_size=256, shuffle=False)
    
    model = Full_System()
    
    # Hàm Loss và Optimizer chuẩn mực
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.005)
    
    EPOCHS = 15
    best_acc = 0.0
    best_model_path = 'best_model_weights.pth'
    
    print("\n🚀 BẮT ĐẦU HUẤN LUYỆN TRÊN DỮ LIỆU SÓNG DAS THẬT...")
    for epoch in range(EPOCHS):
        model.train() # Kích hoạt chế độ học
        total_loss = 0
        correct = 0
        total = 0
        
        for inputs, labels in train_loader:
            optimizer.zero_grad()       # Xóa gradient cũ
            outputs = model(inputs)     # Phán đoán
            loss = criterion(outputs, labels) # Tính lỗi
            loss.backward()             # Lan truyền ngược
            optimizer.step()            # Cập nhật trọng số
            
            total_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
        train_acc = 100 * correct / total
        
        # Đánh giá trên tập Validation
        model.eval() # KHÓA HỌC (Quan trọng để chốt BatchNorm)
        val_correct = 0
        val_total = 0
        with torch.no_grad():
            for inputs, labels in val_loader:
                outputs = model(inputs)
                _, predicted = torch.max(outputs.data, 1)
                val_total += labels.size(0)
                val_correct += (predicted == labels).sum().item()
                
        val_acc = 100 * val_correct / val_total
        
        print(f"Epoch {epoch+1:02d}/{EPOCHS} | Train Loss: {total_loss/len(train_loader):.4f} - Train Acc: {train_acc:.2f}% | Val Acc: {val_acc:.2f}%")
        
        # Lưu lại file trọng số tốt nhất
        if val_acc > best_acc:
            best_acc = val_acc
            torch.save(model.state_dict(), best_model_path)
            print(f"  ⭐ Đã lưu kỷ lục mới: {best_acc:.2f}%")
            
    print(f"\n🎉 HOÀN TẤT HUẤN LUYỆN! Trọng số xịn nhất ({best_acc:.2f}%) nằm ở file '{best_model_path}'")