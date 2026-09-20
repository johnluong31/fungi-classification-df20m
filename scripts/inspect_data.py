import pandas as pd

def find_top_species():
    print("Đang đọc file metadata...")
    # Chỉnh lại đường dẫn nếu file CSV của bạn đang để ở chỗ khác
    csv_path = 'data/raw/DF20-Mini/metadata/DF20M-train_metadata_PROD.csv'
    
    try:
        df = pd.read_csv(csv_path)
        
        # Đếm 10 loài xuất hiện nhiều nhất
        top_10 = df['scientificName'].value_counts().head(10)
        
        print("\n=== TOP 10 LOÀI CÓ NHIỀU ẢNH NHẤT ===")
        print(top_10)
        
        print("\nTổng số lượng dòng trong file:", len(df))
    except FileNotFoundError:
        print(f"LỖI: Không tìm thấy file tại đường dẫn {csv_path}")
        print("Hãy đảm bảo bạn đã tạo thư mục data/raw và bỏ file CSV vào đó!")

if __name__ == "__main__":
    find_top_species()