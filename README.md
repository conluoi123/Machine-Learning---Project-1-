# Machine Learning Project 1: Olist E-commerce Delivery Time Prediction

## 1. Giới thiệu

Đây là dự án Machine Learning đầu tiên trong học phần, tập trung vào bài toán **Hồi quy (Regression)**. Mục tiêu của dự án là xây dựng mô hình dự đoán thời gian giao hàng của các đơn hàng trên sàn thương mại điện tử Olist.

## 2. Cấu trúc thư mục

```
Machine-Learning---Project-1-/
├── code/
│   ├── Part1_Regression/      # Toàn bộ code xử lý và mô hình hồi quy
│   └── Part2_Classification/  # (Sẽ cập nhật sau) Code phân loại
├── data/
│   ├── raw/                   # Dữ liệu thô (chưa xử lý)
│   └── processed/             # Dữ liệu đã qua xử lý (đã làm sạch, feature engineering)
├── models/                    # Lưu trữ các mô hình đã huấn luyện (.pkl)
├── notebooks/                 # Các file Jupyter Notebook để thử nghiệm
├── requirements.txt           # Danh sách thư viện cần thiết
└── README.md                  # File này
```

## 3. Công nghệ sử dụng

- **Ngôn ngữ**: Python 3.10+
- **Thư viện chính**:
  - `pandas`, `numpy`: Xử lý dữ liệu
  - `scikit-learn`: Xây dựng mô hình, đánh giá, tiền xử lý
  - `matplotlib`, `seaborn`: Trực quan hóa dữ liệu
  - `statsmodels`: Phân tích thống kê
  - `category_encoders`: Mã hóa biến phân loại

## 4. Hướng dẫn cài đặt

1. Clone hoặc tải code về máy
2. Mở terminal/command prompt trong thư mục gốc
3. Tạo môi trường ảo (khuyến nghị):
   ```bash
   python -m venv .venv
   .venv\Scripts\activate
   ```
4. Cài đặt thư viện:
   ```bash
   pip install -r requirements.txt
   ```

## 5. Cách chạy code

### 5.1. Xử lý dữ liệu (Data Processing)

- File chính: `code/Part1_Regression/preprocessing.ipynb`
- Chạy tuần tự các cell để:
  1. Load dữ liệu thô
  2. Làm sạch dữ liệu (loại bỏ đơn không giao, xử lý giá trị thiếu)
  3. Feature Engineering (tạo biến mới)
  4. Lưu dữ liệu đã xử lý vào `data/processed/`

### 5.2. Huấn luyện mô hình (Model Training)

- File chính: `code/Part1_Regression/models.ipynb`
- Chạy tuần tự các cell để:
  1. Load dữ liệu đã xử lý
  2. Huấn luyện các mô hình (Linear Regression, Ridge, Lasso, SVR, Random Forest, XGBoost)
  3. Đánh giá hiệu năng (MAE, MSE, R2, RMSE)
  4. Lưu mô hình tốt nhất vào `models/`

## 6. Kết quả

(Sẽ cập nhật sau khi hoàn thành phần 2)
