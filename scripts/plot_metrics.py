import matplotlib
matplotlib.use('Agg')  # Thêm dòng này để tránh lỗi _tkinter trên laptop
import matplotlib.pyplot as plt

# 1. Dữ liệu trích xuất từ quá trình huấn luyện
epochs = [1, 2, 3, 4, 5]
train_loss = [0.6028, 0.3471, 0.2805, 0.2420, 0.2278]
val_loss = [0.3063, 0.2848, 0.2639, 0.2450, 0.2485]
val_acc = [0.8872, 0.9068, 0.9143, 0.9218, 0.9218]

# Cấu hình font chữ và độ phân giải
plt.rcParams.update({'font.size': 12})

# ==========================================
# BIỂU ĐỒ 1: TRAINING & VALIDATION LOSS
# ==========================================
plt.figure(figsize=(8, 5), dpi=150)
plt.plot(epochs, train_loss, marker='o', linewidth=2, color='#1f77b4', label='Train Loss')
plt.plot(epochs, val_loss, marker='s', linewidth=2, color='#ff7f0e', label='Validation Loss')

plt.title('Training and Validation Loss', fontweight='bold', fontsize=14)
plt.xlabel('Epoch', fontweight='bold')
plt.ylabel('Loss', fontweight='bold')
plt.xticks(epochs)
plt.grid(True, linestyle='--', alpha=0.6)
plt.legend(loc='upper right')
plt.tight_layout()

# Lưu ảnh để chèn vào slide
plt.savefig('loss_chart.png')
print("Đã lưu biểu đồ Loss thành công: loss_chart.png")


# ==========================================
# BIỂU ĐỒ 2: VALIDATION ACCURACY
# ==========================================
plt.figure(figsize=(8, 5), dpi=150)
plt.plot(epochs, val_acc, marker='D', linewidth=2, color='#2ca02c', label='Validation Accuracy')

# Khoanh vùng điểm cao nhất tại Epoch 4
best_epoch = 4
best_acc = 0.9218
plt.plot(best_epoch, best_acc, marker='o', markersize=10, color='red', fillstyle='none', markeredgewidth=2)
plt.annotate(f'Best: {best_acc*100:.2f}%', 
             xy=(best_epoch, best_acc), 
             xytext=(best_epoch - 0.8, best_acc - 0.01),
             arrowprops=dict(facecolor='black', shrink=0.05, width=1.5, headwidth=6),
             fontweight='bold', color='red')

plt.title('Validation Accuracy qua từng Epoch', fontweight='bold', fontsize=14)
plt.xlabel('Epoch', fontweight='bold')
plt.ylabel('Accuracy', fontweight='bold')
plt.xticks(epochs)
plt.grid(True, linestyle='--', alpha=0.6)
plt.legend(loc='lower right')
plt.tight_layout()

# Lưu ảnh để chèn vào slide
plt.savefig('accuracy_chart.png')
print("Đã lưu biểu đồ Accuracy thành công: accuracy_chart.png")