import os   
import pandas as pd
import numpy as np
from sklearn.ensemble import IsolationForest

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

def initialize_target_variable(dataframe):
    """
    Tính toán biến mục tiêu (thời gian giao hàng thực tế) và làm sạch các dữ liệu lỗi thời gian.

    """
    df = dataframe.copy()
    
    start_date = 'order_purchase_timestamp'
    end_date = 'order_delivered_customer_date'
    
    # Loại bỏ các dòng không có ngày giao hàng thực tế
    initial_rows = len(df)
    df = df.dropna(subset=[end_date])
    
    # Tính toán biến mục tiêu (Đơn vị: Ngày)
    df['order_delivered_customer_date'] = pd.to_datetime(df['order_delivered_customer_date'])
    df['order_purchase_timestamp'] = pd.to_datetime(df['order_purchase_timestamp'])
    df['target_delivery_days'] = (df[end_date] - df[start_date]).dt.total_seconds() / 86400
    
    # Lọc bỏ các dòng lỗi logic
    # Loại bỏ các đơn hàng có thời gian giao <= 0 (ngày nhận trước hoặc bằng ngày đặt - lỗi hệ thống)
    df = df[df['target_delivery_days'] > 0]
    
    final_rows = len(df)
    print(f"--- KHỞI TẠO BIẾN MỤC TIÊU ---")
    print(f"- Số dòng bị loại bỏ (do thiếu ngày giao hoặc lỗi logic): {initial_rows - final_rows}")
    print(f"- Số lượng mẫu còn lại: {final_rows}")
    print(f"- Thời gian giao hàng trung bình trong tập dữ liệu: {df['target_delivery_days'].mean():.2f} ngày")
    
    return df

def fill_time_gap_with_median(df, ref_col, target_col):
    """
    Điền giá trị thiếu của target_col dựa trên ref_col + median duration.
    Có báo cáo số lượng dòng đã điền.
    """
    duration = (df[target_col] - df[ref_col]).dt.total_seconds()
    median_val = duration.median()
    
    mask = df[target_col].isnull() & df[ref_col].notnull()
    
    filled_count = mask.sum() 
    
    df.loc[mask, target_col] = df.loc[mask, ref_col] + pd.to_timedelta(median_val, unit='s')
    
    print(f"--- Xử lý cột: {target_col} ---")
    print(f"   + Số dòng đã điền: {filled_count:,}")
    print(f"   + Khoảng lệch trung vị (Median): {median_val/3600:.2f} giờ")
    print("-" * 30)
    
    return df
def optimize_logistics_timestamps(dataframe):
    """
    Hàm tổng quát để xử lý toàn bộ chuỗi thời gian logistics.
    """
    df = dataframe.copy()

    time_cols = ['order_purchase_timestamp', 'order_approved_at', 
                 'order_delivered_carrier_date']
    
    for col in time_cols:
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], errors='coerce')
    
    df = fill_time_gap_with_median(df, 'order_purchase_timestamp', 'order_approved_at')
    
    df = fill_time_gap_with_median(df, 'order_approved_at', 'order_delivered_carrier_date')
    
    return df

def fill_product_specs_smart(dataframe):
    """
    Điền các giá trị thiếu cho các cột kích thước sản phẩm
    """
    df = dataframe.copy()
    cols_to_fill = ['product_weight_g', 'product_length_cm', 
                    'product_height_cm', 'product_width_cm']
    print("--- XỬ LÝ KÍCH THƯỚC & TRỌNG LƯỢNG ---")
    for col in cols_to_fill:
        initial_nulls = df[col].isnull().sum()
        if initial_nulls == 0:
            continue

        # Bước 1: Thử điền bằng trung vị của nhóm sản phẩm 
        if 'product_category_name_english' in df.columns:
            df[col] = df[col].fillna(df.groupby('product_category_name_english')[col].transform('median'))
        
        # Bước 2: Nếu vẫn còn trống, điền bằng trung vị toàn cục
        global_median = df[col].median()
        df[col] = df[col].fillna(global_median)
        
        print(f"-> Cột {col}: Đã điền {initial_nulls} dòng. (Global Median dùng: {global_median})")
        
    return df

def handle_category_missing(df):
    """
    Xử lý giá trị thiếu cho cột tên danh mục sản phẩm.
    """
    col = 'product_category_name_english'
    
    if col in df.columns:
        # 1. Chuyển về chữ thường để đồng nhất 
        df[col] = df[col].str.lower()
        
        # 2. Điền giá trị thiếu bằng nhãn 'other'
        null_count = df[col].isnull().sum()
        df[col] = df[col].fillna('other')
        
        print(f"--- XỬ LÝ DANH MỤC ---")
        print(f"-> Đã điền {null_count} dòng trống bằng nhãn 'other'.")
    
    return df

def compare_outlier_methods(df_final, list_columns, contamination=0.05): 
    """
        So sánh 2 phương pháp: 
            - IQR
            - Isolation Forest 
        Args: 
            df_final: Dataframe dữ liệu 
            list_columns: danh sách các cột cần xử lí 
            contamination: Tỷ lệ dữ liệu ngoại lai (mặc định 0.05)
        Returns: 
            pd.Dataframe: Dataframe kèm theo 2 cột mới đánh dấu outliers 
    """
    df = df_final.copy()
    # IQR
    iqr_outliers = pd.Series([False] * len(df), index=df.index)
    for col in list_columns:
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        
        column_outlier = (df_final[col] < lower_bound) | (df_final[col] > upper_bound)
        iqr_outliers = iqr_outliers | column_outlier

    df['is_outlier_iqr'] = iqr_outliers.astype(int)

    # IsolationForest 
    clf = IsolationForest(contamination=contamination, random_state=42)
    predicts = clf.fit_predict(df[list_columns].fillna(0))
    df['is_outlier_if'] = (predicts == -1).astype(int)
    
    return df 

def summarize_outliers(df_processed): 
    """
        In ra bảng tóm tắt giữa hai phương pháp 
    """
    total = len(df_processed)
    iqr_count = df_processed['is_outlier_iqr'].sum()
    if_count = df_processed['is_outlier_if'].sum()
    both = df_processed[(df_processed['is_outlier_iqr'] == 1) & (df_processed['is_outlier_if'] == 1)].shape[0]
    
    print(f"---Tổng kết xử lí Outliers---")
    print(f"Tổng số bản ghi: {total}")
    print(f"Số bản ghi IQR: {iqr_count}")
    print(f"Số bản ghi Isolation Forest: {if_count}")
    print(f"Số bản ghi cả 2: {both}")
    print(f"Số lượng IQR xóa nhưng Isolation Forest giữ lại là {iqr_count-both}")






    
    