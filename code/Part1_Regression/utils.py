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
