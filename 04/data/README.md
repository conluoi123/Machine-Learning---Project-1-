# Thư mục Dữ liệu (Data)

Thư mục này chứa toàn bộ các tập dữ liệu được sử dụng trong dự án. Để thuận tiện cho việc chấm bài và chạy thử nghiệm, dữ liệu được tổ chức theo cấu trúc giúp người dùng có thể chạy ngay các mô hình mà không cần thực hiện lại các bước tiền xử lý phức tạp.

---

## Tải dữ liệu từ Google Drive

Do kích thước dữ liệu lớn và bao gồm các file binary (`.pkl`), chúng tôi đã lưu trữ toàn bộ thư mục này trên Google Drive để đảm bảo tính nhất quán của môi trường.

**[Link tải thư mục DATA tại đây (Google Drive)](https://drive.google.com/drive/folders/1jTmUrWcr3uta2RRPH7qYIhUI9fU2lMjy?usp=sharing)**

---

## Hướng dẫn thiết lập

Dữ liệu của dự án có kích thước khá lớn và bao gồm các file binary (`.pkl`), do đó chúng tôi đã lưu trữ toàn bộ thư mục này trên Google Drive.

**Cách thiết lập nhanh nhất:**

1. Tải toàn bộ thư mục `data` từ Google Drive (Link đính kèm trong báo cáo/README chính).
2. Giải nén và đặt thư mục `data` vào thư mục gốc của project sao cho nó nằm cùng cấp với thư mục `code`.
   ```text
   Machine-Learning---Project-1/
   ├── code/           # Chứa các file Notebook (.ipynb)
   ├── data/           # <--- Giải nén toàn bộ thư mục data vào đây
   ├── report/         # Báo cáo và hình ảnh
   └── requirements.txt
   ```

---

## Cấu trúc thư mục thực tế

```text
data/
├── raw/                    # Dữ liệu gốc chưa qua xử lý
│   ├── classification/     # File train.csv, test.csv của Airline Satisfaction
│   └── regression/         # Các bảng dữ liệu gốc của Olist E-Commerce
├── processed/              # DỮ LIỆU QUAN TRỌNG NHẤT
│                           # Chứa file 'processed_data.pkl' đã được làm sạch,
│                           # mã hóa và chuẩn hóa. Giúp chạy Model ngay lập tức.
├── saved_models/           # Lưu trữ các bộ Scaler, Encoder đã huấn luyện
└── README.md
```

---

## Chi tiết các thành phần chính

### 1. Dữ liệu thô (Raw Data) - `data/raw/`

Chứa các tệp dữ liệu gốc tải về từ Kaggle. Việc giữ lại các tệp này giúp đảm bảo tính minh bạch và cho phép chạy lại toàn bộ quy trình từ bước làm sạch dữ liệu.

### 2. Dữ liệu đã xử lý - `data/processed/`

Đây là thư mục chứa các "Artifacts" cần thiết để các Notebook huấn luyện mô hình hoạt động:

- **`processed_data.pkl`**: Tệp quan trọng nhất cho phần Phân loại. Nó chứa các biến đã được phân tách (Train/Val/Test) và đã qua xử lý. Các notebook như `models.ipynb`, `model_2.ipynb` sẽ load trực tiếp tệp này.
- **`*_models.pkl`**: Lưu trữ kết quả của các mô hình đã huấn luyện (Logistic Regression, LDA, QDA, Perceptron) để phục vụ việc đánh giá và so sánh nhanh.

### 3. Pipeline Artifacts - `data/saved_models/`

Chứa các đối tượng dùng để biến đổi dữ liệu như `TargetEncoder`, `QuantileScaler`. Các đối tượng này được lưu lại để đảm bảo dữ liệu mới (Test set) được xử lý hoàn toàn giống với dữ liệu huấn luyện.

---

## Lưu ý quan trọng

- **Không thay đổi tên file**: Các đường dẫn trong Notebook đã được thiết lập theo dạng tương đối (Relative path). Việc giữ nguyên cấu trúc thư mục như trên là bắt buộc để code có thể chạy thành công.
- **Môi trường chạy**: Nếu chạy trên local, hãy đảm bảo bạn đã cài đặt các thư viện trong `requirements.txt`.
