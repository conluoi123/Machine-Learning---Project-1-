# 🛒 Machine Learning Project 1: Regression & Classification Benchmarking

> **Môn học:** Machine Learning | **Dataset:** [Olist E-Commerce](https://www.kaggle.com/datasets/olistbr/brazilian-ecommerce) & [Airline Passenger Satisfaction](https://www.kaggle.com/datasets/teejmahal20/airline-passenger-satisfaction)

---

## Tổng quan dự án

Dự án ứng dụng các kỹ thuật Machine Learning từ cơ bản đến nâng cao trên hai bộ dữ liệu thực tế, bao gồm **2 phần chính**:

| Phần                        | Bài toán  | Dataset | Mục tiêu                                     |
| --------------------------- | --------- | ------- | -------------------------------------------- |
| **Part 1 – Regression**     | Hồi quy   | Olist   | Dự đoán **thời gian giao hàng** (số ngày)    |
| **Part 2 – Classification** | Phân loại | Airline | Phân loại **mức độ hài lòng** của hành khách |

---

## Dữ liệu (Google Drive)

Do kích thước dữ liệu lớn và bao gồm các file binary (`.pkl`), chúng tôi đã lưu trữ toàn bộ thư mục `data/` trên Google Drive.

**[Link tải thư mục DATA tại đây (Google Drive)](https://drive.google.com/drive/folders/1jTmUrWcr3uta2RRPH7qYIhUI9fU2lMjy?usp=sharing)**

---

## Cấu trúc thư mục

```text
Machine-Learning---Project-1/
│
├── code/
│   ├── Part1_Regression/           # EDA, tiền xử lý & Model hồi quy (Olist)
│   └── Part2_Classification/       # EDA, tiền xử lý & Model phân loại (Airline)
├── data/
│   ├── raw/                        # Backup dữ liệu thô gốc
│   ├── processed/                  # Dữ liệu đã xử lý & artifacts (.pkl)
│   └── saved_models/               # Encoder & Scaler đã fit
├── report/
│   ├── technical_report.md         # Báo cáo kỹ thuật chi tiết (Part 1)
│   └── figures/                    # Toàn bộ biểu đồ xuất ra
└── requirements.txt
```

---

## Hướng dẫn cài đặt & chạy

### 1. Cài đặt môi trường

```bash
git clone https://github.com/conluoi123/Machine-Learning---Project-1-.git
cd Machine-Learning---Project-1-
python -m venv .venv
# Windows
.venv\Scripts\activate
# macOS / Linux
source .venv/bin/activate
pip install -r requirements.txt
```

### 2. Thiết lập dữ liệu

Tải thư mục `data` từ Drive và giải nén vào thư mục gốc của project (cùng cấp với thư mục `code`).

---

## Kết quả — Part 1: Regression (Olist Data)

### Pipeline

```text
Raw Data (Olist) → Merge & Clean → Feature Engineering → Scaling → Models
```

### Hiệu năng mô hình (Test Set)

| Mô hình               | RMSE (ngày) | MAE (ngày) | R²        |
| --------------------- | ----------- | ---------- | --------- |
| **Baseline (OLS)**    | 4.26        | 3.30       | 0.383     |
| **Robust Regression** | **4.15**    | **3.12**   | **0.410** |

---

## Kết quả — Part 2: Classification (Airline Satisfaction)

### Pipeline

```text
Raw Data (Airline) → Handling Missing → Target Encoding → Quantile Scaling → Models
```

### Hiệu năng tổng hợp

| Mô hình                 | Accuracy   | F1-Score   | ROC-AUC    |
| ----------------------- | ---------- | ---------- | ---------- |
| **Logistic Regression** | **0.8752** | **0.8541** | **0.9482** |
| **LDA (Linear)**        | 0.8698     | 0.8465     | 0.9412     |
| **QDA (Quadratic)**     | 0.8512     | 0.8214     | 0.9256     |
| **Perceptron**          | 0.8124     | 0.7910     | N/A        |

---

## Thành viên nhóm (5 Thành viên)

1. **Nguyễn Kim Quốc**
2. **Huỳnh Trọng Viên**
3. **Ngô Thị Thục Quyên**
4. **Lục Hoàng Tuấn**
5. **Cao Quốc Tuấn**

---

## License

Dự án phục vụ mục đích học thuật. Dataset thuộc bản quyền của các tác giả tương ứng trên Kaggle.
