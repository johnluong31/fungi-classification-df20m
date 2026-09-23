# 🍄 Hệ thống phân loại nấm (Fungi Classification - DF20-Mini)

Ứng dụng web nhận diện và phân loại nấm theo loài, chi và họ dựa trên kỹ thuật Transfer Learning với tập dữ liệu ảnh DF20-Mini, tích hợp giao diện demo Gradio.

---

## 📁 Cấu trúc thư mục chính

```text
fungi-classification-df20m/
├── data/
│   ├── processed/          # Chứa file ánh xạ nhãn và phân loại học (.json)
│   │   ├── label_mapping.json
│   │   └── taxonomy_mapping.json
│   └── test_samples/       # Ảnh mẫu ngoài tự nhiên dùng để test nhanh
├── models/                 # Trọng số mô hình đã huấn luyện (.pth)
│   └── best_model.pth
├── results/                # Kết quả đánh giá và biểu đồ ma trận nhầm lẫn
│   └── figures/
├── scripts/                # Mã nguồn tiền xử lý, huấn luyện và đánh giá
│   ├── prepare_data.py
│   ├── train.py
│   └── evaluate.py
├── web/                    # Mã nguồn giao diện web Gradio
│   └── app.py
├── requirements.txt        # Danh sách các thư viện phụ thuộc
└── README.md
```

---

## 🚀 Hướng dẫn cài đặt và chạy ứng dụng

### 1. Clone mã nguồn về máy

```bash
git clone [https://github.com/johnluong31/fungi-classification-df20m.git](https://github.com/johnluong31/fungi-classification-df20m.git)
cd fungi-classification-df20m
```

### 2. Thiết lập môi trường ảo

**Sử dụng Conda (khuyên dùng):**
```bash
conda create -n fungi-df20 python=3.11 -y
conda activate fungi-df20
```

*Hoặc sử dụng Python `venv`:*
```bash
# Windows
python -m venv venv
venv\Scripts\activate

# Linux / macOS
python3 -m venv venv
source venv/bin/activate
```

### 3. Cài đặt các thư viện phụ thuộc

```bash
pip install -r requirements.txt
```

### 4. Khởi chạy ứng dụng Web (Gradio)

```bash
python web/app.py
```

Sau khi khởi chạy thành công, mở trình duyệt web và truy cập vào địa chỉ:
```text
[http://127.0.0.1:7860](http://127.0.0.1:7860)
```

---

## 🧪 Hướng dẫn chạy thử nghiệm & Demo ứng dụng

Dự án cung cấp sẵn bộ ảnh mẫu thực tế từ tự nhiên được trích xuất từ iNaturalist và Mushroom Observer trong thư mục `data/test_samples/` để kiểm thử độ chính xác mà không cần tải dữ liệu huấn luyện nặng.

### Các bước kiểm thử:
1. Tại giao diện **Tải ảnh nấm lên đây**, kéo thả hoặc click chọn một ảnh bất kỳ trong thư mục `data/test_samples/`.
2. Bấm nút **Phân loại ngay**.
3. Quan sát các trường thông tin:
   * **Tên Loài (Species):** Kết quả dự đoán từ mô hình Transfer Learning (kèm tên khoa học).
   * **Chi (Genus) & Họ (Family):** Kết quả suy luận phân loại học tự động từ cơ sở dữ liệu taxonomy.
   * **Top 3 độ tự tin:** Biểu đồ xác suất của 3 loài có khả năng cao nhất.

### Danh mục 5 loài nấm trong hệ thống:

| Tên khoa học | Tên tiếng Việt | Đặc tính | File ảnh mẫu kiểm thử |
| :--- | :--- | :--- | :--- |
| *Amanita muscaria* | Nấm tán bay (Nấm tán đỏ) | Độc / Gây ảo giác | `data/test_samples/test_Amanita_muscaria_*.jpg` |
| *Boletus edulis* | Nấm gan bò (Nấm thông) | Ăn được (cao cấp) | `data/test_samples/test_Boletus_edulis_*.jpg` |
| *Amanita rubescens* | Nấm ửng hồng | Ăn được khi nấu chín | `data/test_samples/test_Amanita_rubescens_*.jpg` |
| *Clitocybe nebularis* | Nấm phễu mây | Dễ gây rối loạn tiêu hóa | `data/test_samples/test_Clitocybe_nebularis_*.jpg` |
| *Mycena galericulata* | Nấm mũ nón | Không ăn được | `data/test_samples/test_Mycena_galericulata_*.jpg` |

> **Lưu ý đánh giá (Model Behavior & Error Analysis):**
> * Mô hình đạt độ tin cậy tối ưu khi ảnh chụp rõ nét mũ nấm từ trên xuống hoặc nghiêng nhẹ.
> * Đối với ảnh chụp ngược phiến nấm (under-cap view) hoặc ảnh có bối cảnh phức tạp (thân gỗ mục, tay người cầm), mô hình có thể xuất hiện độ phân tán xác suất giữa Top 1 và Top 2. Người dùng nên tham khảo biểu đồ **Top 3 độ tự tin** để có đánh giá khách quan nhất.
