import gradio as gr
import torch
import json
import torch.nn.functional as F
from torchvision import models, transforms
from PIL import Image

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Nạp dữ liệu taxonomy và labels
with open('data/processed/label_mapping.json', 'r', encoding='utf-8') as f:
    species_to_idx = json.load(f)
idx_to_species = {v: k for k, v in species_to_idx.items()}

with open('data/processed/taxonomy_mapping.json', 'r', encoding='utf-8') as f:
    taxonomy_mapping = json.load(f)

# Nạp model
model = models.resnet18(weights=None)
model.fc = torch.nn.Linear(model.fc.in_features, len(species_to_idx))
model.load_state_dict(torch.load('models/best_model.pth', map_location=device))
model = model.to(device)
model.eval()

preprocess = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

def predict(image):
    if image is None: return "Lỗi", "", "", {}
    
    img_tensor = preprocess(image).unsqueeze(0).to(device)
    
    with torch.no_grad():
        outputs = model(img_tensor)
        probabilities = F.softmax(outputs, dim=1)[0]
        
    top_prob, top_indices = torch.topk(probabilities, 3)
    
    confidences = {idx_to_species[top_indices[i].item()]: top_prob[i].item() for i in range(3)}
    
    top1_species = idx_to_species[top_indices[0].item()]
    genus = taxonomy_mapping.get(top1_species, {}).get('genus', 'N/A')
    family = taxonomy_mapping.get(top1_species, {}).get('family', 'N/A')
    
    return top1_species, genus, family, confidences

# Giao diện Web
with gr.Blocks() as demo:
    gr.Markdown("# 🍄 Hệ Thống Phân Loại Nấm Thông Minh (DF20-Mini)")
    
    gr.HTML("""
        <div style="background-color: #ffe6e6; padding: 15px; border-left: 5px solid #ff3333; margin-bottom: 20px;">
            <h3 style="margin-top: 0; color: #cc0000;">⚠️ CẢNH BÁO AN TOÀN TỪ NHÓM PHÁT TRIỂN</h3>
            Hệ thống này chỉ phục vụ mục đích <b>học thuật và demo</b>. TUYỆT ĐỐI KHÔNG sử dụng kết quả dự đoán để quyết định ăn bất kỳ loại nấm nào ngoài tự nhiên.
        </div>
    """)
    
    with gr.Row():
        with gr.Column():
            input_image = gr.Image(type="pil", label="Tải ảnh nấm lên đây")
            btn = gr.Button("Phân loại ngay", variant="primary")
        with gr.Column():
            out_species = gr.Textbox(label="Tên Loài (Species) - Dự đoán từ Model")
            out_genus = gr.Textbox(label="Chi (Genus) - Suy ra từ Taxonomy")
            out_family = gr.Textbox(label="Họ (Family) - Suy ra từ Taxonomy")
            out_conf = gr.Label(label="Top 3 độ tự tin")
            
    btn.click(fn=predict, inputs=input_image, outputs=[out_species, out_genus, out_family, out_conf])

if __name__ == "__main__":
    demo.launch(share=False)