import torch
import json
import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, f1_score, top_k_accuracy_score
from torchvision import models, transforms
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from tqdm import tqdm

# Tái định nghĩa lại Dataset để chạy độc lập
class FungiDataset(Dataset):
    def __init__(self, csv_file, transform=None):
        self.data = pd.read_csv(csv_file)
        self.transform = transform
    def __len__(self):
        return len(self.data)
    def __getitem__(self, idx):
        img_path = self.data.iloc[idx]['full_image_path']
        try:
            image = Image.open(img_path).convert('RGB')
        except:
            image = Image.new('RGB', (224, 224))
        label = int(self.data.iloc[idx]['label'])
        if self.transform:
            image = self.transform(image)
        return image, label

def evaluate_model():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Đang tiến hành đánh giá trên: {device}")
    
    # 1. Tải bộ ánh xạ nhãn
    with open('data/processed/label_mapping.json', 'r', encoding='utf-8') as f:
        species_to_idx = json.load(f)
    idx_to_species = {v: k for k, v in species_to_idx.items()}
    class_names = [idx_to_species[i] for i in range(len(species_to_idx))]
    
    # 2. Chuẩn bị tập Test
    test_transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    test_dataset = FungiDataset('data/processed/splits/test.csv', transform=test_transform)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
    
    # 3. Nạp mô hình đã train (best_model.pth)
    model = models.resnet18(weights=None)
    model.fc = torch.nn.Linear(model.fc.in_features, len(class_names))
    model.load_state_dict(torch.load('models/best_model.pth', map_location=device))
    model = model.to(device)
    model.eval()
    
    all_preds, all_labels, all_probs = [], [], []
    
    # 4. Suy luận
    with torch.no_grad():
        for inputs, labels in tqdm(test_loader, desc="Đang quét tập Test"):
            inputs = inputs.to(device)
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            probs = torch.softmax(outputs, dim=1)
            all_probs.extend(probs.cpu().numpy())
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.numpy())
            
    # 5. Tính toán chỉ số báo cáo
    acc = accuracy_score(all_labels, all_preds)
    top3_acc = top_k_accuracy_score(all_labels, all_probs, k=3)
    macro_f1 = f1_score(all_labels, all_preds, average='macro')
    
    print("\n" + "="*40)
    print("KẾT QUẢ TRÊN TẬP TEST")
    print("="*40)
    print(f"Accuracy (Độ chính xác tổng): {acc:.4f}")
    print(f"Top-3 Accuracy: {top3_acc:.4f}")
    print(f"Macro-F1 (Trọng số công bằng):  {macro_f1:.4f}")
    print("\nBáo cáo chi tiết:")
    print(classification_report(all_labels, all_preds, target_names=class_names))
    
    # 6. Vẽ Ma trận nhầm lẫn (Confusion Matrix)
    os.makedirs('results/figures', exist_ok=True)
    cm = confusion_matrix(all_labels, all_preds)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
    plt.ylabel('Nhãn Thực tế (True)')
    plt.xlabel('Nhãn Dự đoán (Predicted)')
    plt.title('Ma Trận Nhầm Lẫn - Tập Test')
    plt.tight_layout()
    plt.savefig('results/figures/confusion_matrix.png')
    print("-> Đã xuất hình ảnh Ma trận nhầm lẫn tại: results/figures/confusion_matrix.png")

if __name__ == "__main__":
    evaluate_model()