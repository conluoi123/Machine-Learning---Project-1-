import os   
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import math
import category_encoders as ce
import time

from statsmodels.stats.outliers_influence import variance_inflation_factor
from statsmodels.tools.tools import add_constant
from sklearn.model_selection import KFold
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, QuantileTransformer, MinMaxScaler
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler

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
    # Danh sách các cột không dùng
    default_drop_cols = [
        'order_id', 'customer_id', 'order_item_id', 'product_id', 'seller_id', 
        'review_id', 'customer_unique_id',
        'review_comment_title', 'review_comment_message', 
        'product_category_name', 
        'product_name_lenght', 'product_description_lenght', 'product_photos_qty',
        
    ]
    if custom_drop_list:
        cols_to_drop = list(set(default_drop_cols + custom_drop_list))
    else:
        cols_to_drop = default_drop_cols
        
    initial_cols = df.shape[1]
    df_cleaned = df.drop(columns=cols_to_drop, errors='ignore')
    final_cols = df_cleaned.shape[1]
    
    dropped_count = initial_cols - final_cols
    print(f"ĐÃ LÀM SẠCH CỘT")
    print(f" + Tổng số cột ban đầu: {initial_cols}")
    print(f" + Số cột đã bị loại bỏ: {dropped_count}")
    print(f" + Số cột còn lại: {final_cols}")
    
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
    
    print(f"ĐÃ LỌC ĐƠN HÀNG")
    print(f" + Tổng số đơn hàng ban đầu: {initial_rows}")
    print(f" + Số đơn hàng đã bị loại bỏ (chưa giao hoặc lỗi ngày): {removed_rows}")
    print(f" + Số đơn hàng còn lại: {final_rows}")

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
    
    df = df[df['target_delivery_days'] > 0]
    
    final_rows = len(df)
    print(f"KHỞI TẠO BIẾN MỤC TIÊU ")
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
    
    print(f"Xử lý cột: {target_col}")
    print(f"   + Số dòng đã điền: {filled_count:,}")
    print(f"   + Khoảng lệch trung vị (Median): {median_val/3600:.2f} giờ")
    
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
    print(" XỬ LÝ KÍCH THƯỚC & TRỌNG LƯỢNG ")
    for col in cols_to_fill:
        initial_nulls = df[col].isnull().sum()
        if initial_nulls == 0:
            continue

        # Điền bằng trung vị của nhóm sản phẩm 
        if 'product_category_name_english' in df.columns:
            df[col] = df[col].fillna(df.groupby('product_category_name_english')[col].transform('median'))
        
        # Nếu vẫn còn trống, điền bằng trung vị toàn cục
        global_median = df[col].median()
        df[col] = df[col].fillna(global_median)
        
        print(f" Cột {col}: Đã điền {initial_nulls} dòng. (Global Median dùng: {global_median})")
        
    return df

def handle_category_missing(df):
    """
    Xử lý giá trị thiếu cho cột tên danh mục sản phẩm.
    """
    col = 'product_category_name_english'
    
    if col in df.columns:
        df[col] = df[col].str.lower()
        
        null_count = df[col].isnull().sum()
        df[col] = df[col].fillna('other')
        
        print(f"XỬ LÝ DANH MỤC ")
        print(f" Đã điền {null_count} dòng trống bằng nhãn 'other'.")
    
    return df

def visualize_outliers(df, cols, n_cols=2, figsize=(20, 12), title='Phân Tích Ngoại Lai (Outliers) Cho Các Đặc Trưng'):
    """
    Vẽ biểu đồ trực quan hóa các điểm ngoại lai (outliers) cho một danh sách các cột số liên tục trong DataFrame.
    """
    n_rows = math.ceil(len(cols) / n_cols)
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize)
    fig.suptitle(title, fontsize=20, fontweight='bold', y=0.98)
    
    if n_rows * n_cols > 1:
        axes_flat = axes.flatten()
    else:
        axes_flat = [axes]
    
    for i, col in enumerate(cols):
        # Vẽ Boxplot
        sns.boxplot(
            x=df[col], 
            ax=axes_flat[i], 
            color='#a8dadc', 
            fliersize=5, 
            flierprops={"marker": "x", "markeredgecolor": "red", "alpha": 0.5}
        )
        
        axes_flat[i].set_title(f'Phân bố của {col}', fontsize=15, fontweight='semibold', pad=10)
        axes_flat[i].set_xlabel(f'Giá trị của {col}', fontsize=12)
        axes_flat[i].set_ylabel('Mật độ / Tần suất', fontsize=12) 
        
        axes_flat[i].grid(axis='x', linestyle='--', alpha=0.6)

    for j in range(i + 1, n_rows * n_cols):
        axes_flat[j].axis('off')

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.show()


def handle_outliers_pipeline(df):
    df_working = df.copy()

    # XỬ LÝ TARGET BẰNG IQR 
    Q1 = df_working['target_delivery_days'].quantile(0.25)
    Q3 = df_working['target_delivery_days'].quantile(0.75)
    IQR = Q3 - Q1
    
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    
    df_working = df_working[(df_working['target_delivery_days'] >= lower_bound) & 
                            (df_working['target_delivery_days'] <= upper_bound)].copy()
    print(f"Đã loại bỏ ngoại lai Target (IQR). Dữ liệu còn lại: {len(df_working)} dòng.")

    # XỬ LÝ FEATURES TÀI CHÍNH BẰNG LOG TRANSFORM 
    df_working.loc[:, 'price'] = np.log1p(df_working['price'])
    df_working.loc[:, 'freight_value'] = np.log1p(df_working['freight_value'])
    print("Đã nén biến Price và Freight Value bằng Log Transform.")

    # LỌC ĐA BIẾN CHO FEATURES KÍCH THƯỚC 
    size_features = ['product_length_cm', 'product_height_cm', 'product_width_cm']
    
    iso_forest = IsolationForest(contamination=0.05, random_state=42)
    outlier_labels = iso_forest.fit_predict(df_working[size_features])
    
    df_final = df_working[outlier_labels == 1].copy()
    
    print(f"Isolation Forest đã loại bỏ thêm {sum(outlier_labels == -1)} dòng ngoại lai đa biến.")
    print(f"Tổng dữ liệu sạch: {len(df_final)} dòng.")
    
    return df_final

def standardize_column_formats(df):
    """
    Chuẩn hóa định dạng dữ liệu
    """
    df_standardized = df.copy()

    # Định dạng NGÀY THÁNG (Datetime)
    date_cols = [
        'order_purchase_timestamp', 'order_approved_at', 
        'order_delivered_carrier_date', 'order_delivered_customer_date', 
        'order_estimated_delivery_date', 'shipping_limit_date'
    ]
    for col in date_cols:
        if col in df_standardized.columns:
            df_standardized[col] = pd.to_datetime(df_standardized[col], errors='coerce')

    # Định dạng SỐ (Numeric/Float)
    num_cols = [
        'price', 'freight_value', 'product_weight_g', 
        'product_length_cm', 'product_height_cm', 'product_width_cm',
        'target_delivery_days'
    ]
    for col in num_cols:
        if col in df_standardized.columns:
            df_standardized[col] = pd.to_numeric(df_standardized[col], errors='coerce')

    # Định dạng CHUỖI VĂN BẢN (String/Object)
    string_cols = [
        'customer_city', 'customer_state', 'seller_city', 
        'seller_state', 'product_category_name_english'
    ]
    for col in string_cols:
        if col in df_standardized.columns:
            df_standardized[col] = df_standardized[col].astype(str).str.lower().str.strip()

    print("CHUẨN HÓA ĐỊNH DẠNG HOÀN TẤT")
    return df_standardized

def handle_duplicates(df, subset=None):
    """
    Phát hiện và loại bỏ các dòng trùng lặp trong DataFrame.
    """
    df_clean = df.copy()
    
    duplicate_count = df_clean.duplicated(subset=subset).sum()
    
    if duplicate_count == 0:
        print("KIỂM TRA TRÙNG LẶP")
        print("Không tìm thấy dòng trùng lặp nào.")
        return df_clean

    df_clean = df_clean.drop_duplicates(subset=subset, keep='first')
    
    print("XỬ LÝ TRÙNG LẶP HOÀN TẤT")
    print(f"- Số lượng dòng trùng lặp đã xóa: {duplicate_count:,}")
    print(f"- Kích thước dữ liệu sau khi xóa: {df_clean.shape}")
    
    return df_clean

def haversine_distance(lat1, lon1, lat2, lon2):
    """
    Tính khoảng cách bề mặt địa cầu (Haversine formula) giữa 2 điểm tọa độ.
    Trọng số bán kính trái đất R = 6371 km.
    """
    R = 6371.0
    
    # Chuyển đổi từ độ sang radian
    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
    
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    
    a = np.sin(dlat / 2)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2)**2
    c = 2 * np.arcsin(np.sqrt(a))
    
    distance = R * c
    return distance

def haversine_distance(lat1, lon1, lat2, lon2):
    """
    Tính khoảng cách Haversine giữa hai điểm trên mặt cầu (đơn vị: km).
    Sử dụng vectorized numpy để tối ưu tốc độ.
    """

    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
    
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    
    a = np.sin(dlat/2.0)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon/2.0)**2
    c = 2 * np.arcsin(np.sqrt(a))
    km = 6371 * c 
    return km

def merge_geolocation_and_calculate_distance(df_main, geo_df):
    df_result = df_main.copy()
    
    geo_clean = geo_df.groupby('geolocation_zip_code_prefix').agg({
        'geolocation_lat': 'mean',
        'geolocation_lng': 'mean'
    }).reset_index()

    df_result = df_result.merge(
        geo_clean, 
        left_on='customer_zip_code_prefix', 
        right_on='geolocation_zip_code_prefix', 
        how='left'
    ).rename(columns={'geolocation_lat': 'cust_lat', 'geolocation_lng': 'cust_lng'}).drop(columns='geolocation_zip_code_prefix')
    
    df_result = df_result.merge(
        geo_clean, 
        left_on='seller_zip_code_prefix', 
        right_on='geolocation_zip_code_prefix', 
        how='left'
    ).rename(columns={'geolocation_lat': 'sell_lat', 'geolocation_lng': 'sell_lng'}).drop(columns='geolocation_zip_code_prefix')
    
    df_result['distance_km'] = haversine_distance(
        df_result['cust_lat'], df_result['cust_lng'],
        df_result['sell_lat'], df_result['sell_lng']
    )
    
    if df_result['distance_km'].isnull().any():
        median_val = df_result['distance_km'].median()
        df_result['distance_km'] = df_result['distance_km'].fillna(median_val)
    
    cols_to_drop = ['cust_lat', 'cust_lng', 'sell_lat', 'sell_lng', 
                    'customer_zip_code_prefix', 'seller_zip_code_prefix']
    df_result.drop(columns=[c for c in cols_to_drop if c in df_result.columns], inplace=True)
    
    print(f" Hoàn thành trích xuất đặc trưng 'distance_km'")
    return df_result

def plot_violin_comparison(df_before, df_after, columns):
    n_cols = len(columns)
    fig, axes = plt.subplots(n_cols, 2, figsize=(15, 5 * n_cols))
    
    for i, col in enumerate(columns):
        # Biểu đồ trước khi chuẩn hóa
        sns.violinplot(data=df_before, y=col, ax=axes[i, 0], color='skyblue')
        axes[i, 0].set_title(f'TRƯỚC: {col}')
        axes[i, 0].set_ylabel('Giá trị gốc')
        
        # Biểu đồ sau khi chuẩn hóa
        sns.violinplot(data=df_after, y=col, ax=axes[i, 1], color='salmon')
        axes[i, 1].set_title(f'SAU: {col}')
        axes[i, 1].set_ylabel('Giá trị đã chuẩn hóa')

    plt.tight_layout()
    plt.show()

def calculate_full_vif(df, target_col='target_delivery_days'):
    X = df.drop(columns=[target_col]) if target_col in df.columns else df.copy()
    X = X.select_dtypes(include=[np.number])
    
    X = X.replace([np.inf, -np.inf], np.nan).fillna(0)
    
    X_with_const = add_constant(X)

    vif_data = pd.DataFrame()
    vif_data["feature"] = X_with_const.columns
    vif_data["VIF"] = [
        variance_inflation_factor(X_with_const.values, i) 
        for i in range(X_with_const.shape[1])
    ]

    return vif_data[vif_data['feature'] != 'const'].sort_values(by="VIF", ascending=False)