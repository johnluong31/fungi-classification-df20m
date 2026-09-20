import pandas as pd
import json
import os
from sklearn.model_selection import train_test_split

def prepare_data():
    print("1. Đang đọc dữ liệu metadata...")
    train_csv_path = 'data/raw/DF20-Mini/metadata/DF20M-train_metadata_PROD.csv'
    test_csv_path = 'data/raw/DF20-Mini/metadata/DF20M-public_test_metadata_PROD.csv'
    
    df_train_full = pd.read_csv(train_csv_path)
    df_test_full = pd.read_csv(test_csv_path)
    
    # Chốt 5 loài có nhiều ảnh nhất từ kết quả khảo sát
    top_5_species = [
        "Mycena galericulata (Scop.) Gray",
        "Clitocybe nebularis (Batsch) Quél.",
        "Amanita muscaria (L.) Lam., 1783",
        "Boletus edulis Bull.",
        "Amanita rubescens (Pers.) Gray"
    ]
    
    print("2. Đang lọc dữ liệu và tạo đường dẫn ảnh...")
    df_train = df_train_full[df_train_full['scientificName'].isin(top_5_species)].copy()
    df_test = df_test_full[df_test_full['scientificName'].isin(top_5_species)].copy()
    
    # Nối đường dẫn trỏ tới thư mục ảnh
    df_train['full_image_path'] = 'data/raw/DF20-Mini/images/' + df_train['ImageUniqueID'].astype(str) + '.jpg'
    df_test['full_image_path'] = 'data/raw/DF20-Mini/images/' + df_test['ImageUniqueID'].astype(str) + '.jpg'
    
    print("3. Phân chia Train/Validation theo gbifID (Tránh Data Leakage)...")
    # Lấy danh sách ID mẫu vật duy nhất
    unique_obs = df_train['gbifID'].unique()
    # Dành 15% mẫu vật cho tập Validation, 85% cho Train
    train_obs, val_obs = train_test_split(unique_obs, test_size=0.15, random_state=42)
    
    train_df = df_train[df_train['gbifID'].isin(train_obs)].copy()
    val_df = df_train[df_train['gbifID'].isin(val_obs)].copy()
    test_df = df_test.copy() # Tập test dùng luôn file test có sẵn
    
    print("4. Khởi tạo nhãn (Label) và Từ điển Taxonomy...")
    species_to_idx = {name: idx for idx, name in enumerate(top_5_species)}
    
    for df in [train_df, val_df, test_df]:
        df['label'] = df['scientificName'].map(species_to_idx)
        
    taxonomy_mapping = {}
    for _, row in train_df.drop_duplicates(subset=['scientificName']).iterrows():
        taxonomy_mapping[row['scientificName']] = {
            'genus': row['genus'],
            'family': row['family']
        }
        
    print("5. Đang lưu kết quả...")
    os.makedirs('data/processed/splits', exist_ok=True)
    
    train_df.to_csv('data/processed/splits/train.csv', index=False)
    val_df.to_csv('data/processed/splits/val.csv', index=False)
    test_df.to_csv('data/processed/splits/test.csv', index=False)
    
    with open('data/processed/label_mapping.json', 'w', encoding='utf-8') as f:
        json.dump(species_to_idx, f, indent=4)
        
    with open('data/processed/taxonomy_mapping.json', 'w', encoding='utf-8') as f:
        json.dump(taxonomy_mapping, f, indent=4)
        
    print("=== HOÀN TẤT GIAI ĐOẠN TIỀN XỬ LÝ ===")
    print(f"Số lượng ảnh Train: {len(train_df)}")
    print(f"Số lượng ảnh Validation: {len(val_df)}")
    print(f"Số lượng ảnh Test: {len(test_df)}")

if __name__ == "__main__":
    prepare_data()