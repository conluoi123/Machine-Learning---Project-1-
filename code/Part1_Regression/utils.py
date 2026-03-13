
def drop_high_missing_columns(df, threshold=0.5):
    # 1. Tính toán tỷ lệ missing value cho mỗi cột
    missing_ratio = df.isnull().sum() / len(df)
    
    # 2. Xác định danh sách các cột vượt ngưỡng
    cols_to_drop = missing_ratio[missing_ratio > threshold].index.tolist()
    
    if len(cols_to_drop) > 0:
        print(f"Đã loại bỏ các cột sau (> {threshold*100}% missing):")
        for col in cols_to_drop:
            print(f" - {col}: {missing_ratio[col]:.2%}")
            
        # 3. Thực hiện xóa cột
        df_cleaned = df.drop(columns=cols_to_drop)
    else:
        print("Không có cột nào vượt ngưỡng giá trị thiếu.")
        df_cleaned = df.copy()
        
    return df_cleaned

