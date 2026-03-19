import os   
import pandas as pd

def read_data(path, file_name):
    file_path = os.path.join(path, file_name)
    if os.path.exists(file_path):
        print(f"Đang đọc dữ liệu từ {file_name}...")
        return pd.read_csv(file_path)
    else:
        print(f"File {file_name} không tồn tại trong đường dẫn.")
        return None
    
def drop_unnecessary_columns(df, custom_drop_list=None):
    """
    Hàm tự động loại bỏ các cột không có giá trị dự báo cho bài toán vận chuyển.
    
    Args:
        dataframe (pd.DataFrame): DataFrame tổng hợp sau khi merge.
        custom_drop_list (list, optional): Danh sách các cột bổ sung muốn xóa.
        
    Returns:
        pd.DataFrame: DataFrame đã được làm sạch.
    """
    # 1. Danh sách các cột "rác" mặc định cho bài toán Delivery
    default_drop_cols = [
        # Nhóm ID: Chỉ là mã định danh, không có tính quy luật cho máy học
        'order_id', 'customer_id', 'order_item_id', 'product_id', 'seller_id', 
        'review_id', 'customer_unique_id',
        
        # Nhóm Văn bản: Quá dài và cần xử lý NLP riêng
        'review_comment_title', 'review_comment_message', 
        'product_category_name', # Đã có cột dịch tiếng Anh thay thế
        
        # Nhóm Chi tiết sản phẩm: Thường không ảnh hưởng trực tiếp đến tốc độ giao hàng
        'product_name_lenght', 'product_description_lenght', 'product_photos_qty',
        
        # Nhóm Vị trí quá chi tiết: Tránh Overfitting 
        'customer_zip_code_prefix', 'seller_zip_code_prefix'
    ]
    # 2. Kết hợp với danh sách tùy chỉnh (nếu có)
    if custom_drop_list:
        cols_to_drop = list(set(default_drop_cols + custom_drop_list))
    else:
        cols_to_drop = default_drop_cols
        
    # 3. Thực hiện xóa
    initial_cols = df.shape[1]
    df_cleaned = df.drop(columns=cols_to_drop, errors='ignore')
    final_cols = df_cleaned.shape[1]
    
    # 4. Thông báo kết quả
    dropped_count = initial_cols - final_cols
    print(f"--- ĐÃ LÀM SẠCH CỘT ---")
    print(f"Tổng số cột ban đầu: {initial_cols}")
    print(f"Số cột đã bị loại bỏ: {dropped_count}")
    print(f"Số cột còn lại: {final_cols}")
    
    return df_cleaned

def filter_and_clean_order_status(dataframe):
    """
    Loại bỏ những đơn hàng không giao thành công và xóa cột dữ liệu order_status sau khi lọc.
    """
    df = dataframe.copy()
    initial_rows = len(df)

    df = df[df['order_status'] == 'delivered']
    
    df = df.dropna(subset=['order_delivered_customer_date'])

    df = df.drop(columns=['order_status'], errors='ignore')

    final_rows = len(df)
    removed_rows = initial_rows - final_rows
    
    print(f"--- ĐÃ LỌC ĐƠN HÀNG ---")
    print(f"Tổng số đơn hàng ban đầu: {initial_rows}")
    print(f"Số đơn hàng đã bị loại bỏ (chưa giao hoặc lỗi ngày): {removed_rows}")
    print(f"Số đơn hàng còn lại: {final_rows}")

    return df

