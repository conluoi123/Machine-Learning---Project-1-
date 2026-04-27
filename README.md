# 🛒 Machine Learning Project 1: Olist E-commerce Delivery Prediction

> **Môn học:** Machine Learning | **Dataset:** [Brazilian E-Commerce (Olist)](https://www.kaggle.com/datasets/olistbr/brazilian-ecommerce)

---

## 📌 Tổng quan dự án

Dự án ứng dụng các kỹ thuật Machine Learning từ cơ bản đến nâng cao trên bộ dữ liệu thương mại điện tử Olist (Brazil), bao gồm **2 phần chính**:

| Phần                        | Bài toán  | Mục tiêu                                          |
| --------------------------- | --------- | ------------------------------------------------- |
| **Part 1 – Regression**     | Hồi quy   | Dự đoán **thời gian giao hàng** (số ngày)         |
| **Part 2 – Classification** | Phân loại | Phân loại kết quả giao hàng / đánh giá khách hàng |

---

## 📁 Cấu trúc thư mục

```
Machine-Learning---Project-1-/
│
├── code/
│   ├── Part1_Regression/
│   │   ├── notebook.ipynb          # EDA & tiền xử lý dữ liệu (Preprocessing)
│   │   ├── models.ipynb            # Huấn luyện & đánh giá toàn bộ mô hình hồi quy
│   │   ├── utils.py                # Hàm tiện ích dùng chung (metrics, plots, helpers)
│   │   ├── requirements.txt        # Danh sách thư viện (Part 1)
│   │   └── processed_data/         # Dữ liệu đã xử lý (local cache cho Part 1)
│   │
│   └── Part2_Classification/
│       ├── notebook.ipynb          # EDA & tiền xử lý dữ liệu (Preprocessing)
│       ├── models.ipynb            # Pipeline phân loại chính (Perceptron, LogReg, LDA/QDA...)
│       ├── model_2.ipynb           # Thử nghiệm mô hình phân loại thay thế
│       └── model_3.ipynb           # Thử nghiệm mô hình phân loại bổ sung
│
├── data/
│   ├── olist_customers_dataset.csv
│   ├── olist_orders_dataset.csv
│   ├── olist_order_items_dataset.csv
│   ├── olist_order_payments_dataset.csv
│   ├── olist_order_reviews_dataset.csv
│   ├── olist_products_dataset.csv
│   ├── olist_sellers_dataset.csv
│   ├── olist_geolocation_dataset.csv
│   ├── product_category_name_translation.csv
│   ├── brazil-states.geojson.txt   # GeoJSON biên giới các bang Brazil
│   ├── raw_data.csv                # Dữ liệu thô đã merge (từ các bảng Olist)
│   ├── raw/                        # Backup dữ liệu thô gốc
│   ├── processed/                  # Dữ liệu đã xử lý & artifacts chia tập
│   │   ├── X_train_scaled.csv / X_val_scaled.csv / X_test_scaled.csv
│   │   ├── y_train.csv / y_val.csv / y_test.csv
│   │   ├── olist_processed_data.joblib
│   │   ├── processed_data.pkl
│   │   ├── preprocessing_info.json
│   │   ├── logistic_models.pkl
│   │   ├── lda_qda_models.pkl
│   │   └── perceptron_logreg_results.pkl
│   └── saved_models/               # Encoder & Scaler đã fit
│       ├── phase7_target_encoder.pkl
│       └── phase8_quantile_scaler.pkl
│
├── report/
│   ├── technical_report.md         # Báo cáo kỹ thuật chi tiết (Part 1 – Regression)
│   ├── report.txt                  # Placeholder báo cáo
│   └── figures/                    # Toàn bộ biểu đồ xuất ra
│       ├── 01_target_distribution.png
│       ├── 02_histograms.png
│       ├── 03_boxplot.png
│       ├── 04_correlation_matrix.png
│       ├── 05_correlation_target.png
│       ├── 06_scatter.png
│       ├── 07_delay_outlier.png
│       ├── comparison/             # So sánh tổng thể giữa các mô hình
│       │   ├── cv_f1_boxplot.png
│       │   ├── error_analysis.png
│       │   ├── fisher_discriminability.png
│       │   └── noise_robustness.png ...
│       ├── lr/                     # Biểu đồ Logistic Regression
│       │   ├── confusion_matrix_lr.png
│       │   ├── calibration.png
│       │   ├── kernel_lr.png
│       │   ├── pr_curve_lr.png
│       │   └── decision_boundary_*.png ...
│       ├── lda_qda/                # Biểu đồ LDA & QDA
│       │   ├── confusion_matrix_lda_qda.png
│       │   └── roc_curve_test.png
│       └── perceptron_logreg/      # Biểu đồ Perceptron & Logistic Regression
│           ├── perceptron_convergence.png
│           ├── perceptron_decision_boundary.png
│           ├── logistic_loss_curve.png
│           └── roc_pr_curve.png ...
│
├── .gitignore
├── .gitattributes
└── README.md
```

---

## ⚙️ Công nghệ sử dụng

| Thư viện            | Phiên bản | Mục đích                                |
| ------------------- | --------- | --------------------------------------- |
| `Python`            | 3.10+     | Ngôn ngữ chính                          |
| `pandas`            | 3.0.2     | Xử lý & thao tác dữ liệu                |
| `numpy`             | 2.4.4     | Tính toán số học, đại số tuyến tính     |
| `scikit-learn`      | 1.8.0     | Tiền xử lý, pipeline, đánh giá mô hình  |
| `matplotlib`        | 3.10.8    | Trực quan hóa dữ liệu                   |
| `seaborn`           | 0.13.2    | Biểu đồ thống kê nâng cao               |
| `statsmodels`       | 0.14.6    | Phân tích hồi quy thống kê              |
| `category_encoders` | 2.9.0     | Mã hóa biến phân loại (Target Encoding) |
| `jinja2`            | ≥ 3.0.0   | Render báo cáo HTML                     |

---

## 🚀 Hướng dẫn cài đặt & chạy

### 1. Clone repository

```bash
git clone https://github.com/conluoi123/Machine-Learning---Project-1-.git
cd Machine-Learning---Project-1-
```

### 2. Tạo môi trường ảo

```bash
python -m venv .venv
# Windows
.venv\Scripts\activate
# macOS / Linux
source .venv/bin/activate
```

### 3. Cài đặt thư viện

```bash
pip install -r code/Part1_Regression/requirements.txt
```

### 4. Chạy notebooks

| Thứ tự | File                                       | Mô tả                                         |
| ------ | ------------------------------------------ | --------------------------------------------- |
| 1️⃣     | `code/Part1_Regression/notebook.ipynb`     | EDA + Tiền xử lý dữ liệu hồi quy              |
| 2️⃣     | `code/Part1_Regression/models.ipynb`       | Huấn luyện & đánh giá toàn bộ mô hình hồi quy |
| 3️⃣     | `code/Part2_Classification/notebook.ipynb` | EDA + Tiền xử lý dữ liệu phân loại            |
| 4️⃣     | `code/Part2_Classification/models.ipynb`   | Pipeline phân loại chính                      |

> ⚠️ **Lưu ý:** Dữ liệu thô Olist (~120MB) cần được đặt trong thư mục `data/`. Tham khảo [Kaggle Dataset](https://www.kaggle.com/datasets/olistbr/brazilian-ecommerce) để tải về.

---

## 📊 Kết quả — Part 1: Regression

### Pipeline

```
Raw Data (9 bảng Olist)
    → Merge & Làm sạch (loại đơn hủy, xử lý NaN)
    → Feature Engineering (9 features logistics)
    → Chuẩn hóa (QuantileTransformer + TargetEncoder)
    → Train / Val / Test Split (70% / 15% / 15%)
    → Mô hình hồi quy (từ baseline đến nâng cao)
```

### Hiệu năng mô hình (Test Set – đơn vị ngày)

| Mô hình                               | RMSE (ngày) | MAE (ngày) | R²          |
| ------------------------------------- | ----------- | ---------- | ----------- |
| **Baseline (Normal Equations / OLS)** | 4.26        | 3.30       | 0.383       |
| Ridge / Bayesian Regression           | 4.27        | 3.30       | 0.379       |
| **Robust Regression (Huber Loss)**    | **4.15\***  | **3.12\*** | **0.410\*** |

_\*Ước tính trên tập có outlier được inject._

### Các mô hình đã triển khai

| Mô hình                         | Kỹ thuật chính                                      |
| ------------------------------- | --------------------------------------------------- |
| **Normal Equations**            | `np.linalg.pinv` – baseline                         |
| **Mini-batch GD**               | Cosine Annealing LR, batch_size=256                 |
| **Ridge / Lasso / Elastic Net** | Regularization path, Coordinate Descent             |
| **Non-linear Basis Functions**  | Polynomial (deg 2), RBF, Trigonometric, Interaction |
| **Bayesian Regression**         | EM (Evidence Maximization), α/β tự động             |
| **Gaussian Process Regression** | RBF kernel, uncertainty quantification              |
| **Robust Regression**           | Huber Loss, giảm 19.1% RMSE trên dữ liệu có outlier |

### Ablation Study – Basis Functions

| Cấu hình                   | MSE    | Δ MSE                |
| -------------------------- | ------ | -------------------- |
| **Full Model (All Basis)** | 0.6298 | –                    |
| Loại Polynomial            | 0.6328 | +0.0030 ✗ Critical   |
| Loại Trigonometric         | 0.6315 | +0.0017 ✗ Meaningful |
| Loại Interaction           | 0.6306 | +0.0008              |
| Loại Gaussian RBF          | 0.6298 | −0.000002 (noise)    |

> **Kết luận:** Polynomial (deg 2) và Trigonometric là hai thành phần quan trọng nhất; RBF trong cấu hình hiện tại không đóng góp đáng kể.

---

## 📊 Kết quả — Part 2: Classification

### Pipeline

```
Raw Data (9 bảng Olist)
    → Merge & Làm sạch (loại đơn hủy, xử lý NaN)
    → Feature Engineering + Target Encoding (phase7)
    → Chuẩn hóa (QuantileScaler – phase8)
    → Train / Val / Test Split (70% / 15% / 15%)
    → Mô hình phân loại (từ Perceptron đến LDA/QDA)
```

### Các mô hình đã triển khai

| Mô hình | Kỹ thuật chính | Notebook |
| ------- | -------------- | -------- |
| **Perceptron** | Online learning, hội tụ tuyến tính | `models.ipynb` |
| **Logistic Regression (Gradient Descent)** | Binary cross-entropy, L2 regularization | `models.ipynb` |
| **Logistic Regression (Newton's Method)** | Hessian-based 2nd order optimization | `models.ipynb` |
| **Kernel Logistic Regression** | RBF kernel, phi(x) mapping phi space | `models.ipynb` |
| **Probit Regression** | Gaussian CDF link function | `models.ipynb` |
| **Laplace Approximation** | Bayesian posterior trên Logistic Reg | `models.ipynb` |
| **Gaussian Naive Bayes (GNB)** | Generative model, class-conditional | `models.ipynb` |
| **Linear Discriminant Analysis (LDA)** | Shared covariance, Fisher criterion | `models.ipynb` |
| **Quadratic Discriminant Analysis (QDA)** | Per-class covariance, quadratic boundary | `models.ipynb` |
| **Thử nghiệm bổ sung** | Các cấu hình thay thế | `model_2.ipynb`, `model_3.ipynb` |

### Phân tích chính

- **Perceptron:** Vẽ convergence curve và decision boundary 2D — xác nhận hội tụ trên dữ liệu linearly separable. Kết quả lưu tại `figures/perceptron_logreg/`.
- **Logistic Regression:** So sánh Gradient Descent vs. Newton's Method (tốc độ hội tụ), phân tích Loss vs. Epoch với nhiều λ regularization. Calibration curve và Reliability Diagram xác nhận độ tin cậy xác suất.
- **Kernel LR:** Decision boundary phi chiều phi tuyến, kiểm chứng VC dimension.
- **Probabilistic Models (Probit, Laplace, GNB):** Đánh giá Precision-Recall Curve và ROC Curve trên test set.
- **LDA / QDA:** Phân tích Fisher Discriminability, so sánh decision boundary 2D cho từng lớp.
- **Robustness:** Noise injection test (`noise_robustness.png`), sensitivity analysis theo tỷ lệ train/test split.

### Đánh giá tổng thể (Cross-validation)

| Metrics | Artifacts |
| ------- | --------- |
| Confusion Matrix | `figures/lr/confusion_matrix_lr.png`, `figures/lda_qda/confusion_matrix_lda_qda.png`, `figures/perceptron_logreg/confusion_matrix_final_model.png` |
| ROC Curve | `figures/lda_qda/roc_curve_test.png`, `figures/perceptron_logreg/roc_pr_curve.png` |
| PR Curve | `figures/lr/pr_curve_lr.png`, `figures/lr/precision_recall_curve_test.png` |
| CV F1 Score | `figures/comparison/cv_f1_boxplot.png` |
| Error Analysis | `figures/comparison/error_analysis.png` |
| Fisher Discriminability | `figures/comparison/fisher_discriminability.png` |
| Noise Robustness | `figures/comparison/noise_robustness.png` |

> **Lưu ý:** Saved models được lưu tại `data/processed/` gồm `logistic_models.pkl`, `lda_qda_models.pkl`, và `perceptron_logreg_results.pkl`.

---

## 📝 Báo cáo kỹ thuật

| Phần | File |
| ---- | ---- |
| Part 1 – Regression (chi tiết) | [`report/technical_report.md`](report/technical_report.md) |

Báo cáo Regression bao gồm: baseline Normal Equations, MBGD với Cosine Annealing, Regularization paths (Ridge/Lasso/ElasticNet), Ablation Study basis functions, Bayesian Regression (EM), GPR, và Robust Regression (Huber Loss).

---

## 👥 Thành viên nhóm

_(Cập nhật thông tin thành viên tại đây)_

---

## 📄 License

Dự án phục vụ mục đích học thuật. Dataset Olist thuộc bản quyền của [Olist](https://olist.com/) và được cấp phép theo [CC BY-NC-SA 4.0](https://creativecommons.org/licenses/by-nc-sa/4.0/).
