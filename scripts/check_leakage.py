import csv

def read_gbif_ids(file_path):
    ids = []
    with open(file_path, mode='r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            ids.append(row['gbifID'])
    return ids

# Đọc danh sách gbifID từ 3 file CSV
train_all = read_gbif_ids("data/processed/splits/train.csv")
val_all = read_gbif_ids("data/processed/splits/val.csv")
test_all = read_gbif_ids("data/processed/splits/test.csv")

# Số lượng ảnh
n_train_img = len(train_all)
n_val_img = len(val_all)
n_test_img = len(test_all)

# Tập hợp gbifID duy nhất
train_ids = set(train_all)
val_ids = set(val_all)
test_ids = set(test_all)

print(f"Train - Ảnh: {n_train_img}, gbifID duy nhất: {len(train_ids)}")
print(f"Val   - Ảnh: {n_val_img}, gbifID duy nhất: {len(val_ids)}")
print(f"Test  - Ảnh: {n_test_img}, gbifID duy nhất: {len(test_ids)}")

# Kiểm tra rò rỉ dữ liệu (Giao nhau giữa các tập)
leak_train_val = len(train_ids.intersection(val_ids))
leak_train_test = len(train_ids.intersection(test_ids))
leak_val_test = len(val_ids.intersection(test_ids))

print("-" * 40)
print(f"Giao Train và Val: {leak_train_val}")
print(f"Giao Train và Test: {leak_train_test}")
print(f"Giao Val và Test: {leak_val_test}")