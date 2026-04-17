# 🏗️ EDA & Preprocessing Pipeline — Senior Review Edition
### Olist E-Commerce · Delivery Time Prediction · Linear Regression Project

> **Nguyên tắc vàng xuyên suốt notebook:**
> 1. **Không bao giờ fit bất cứ thứ gì (Encoder, Scaler) trên data trước khi Split.**
> 2. **Chỉ giữ lại feature có tại thời điểm khách bấm "Đặt hàng"** — mọi cột phát sinh sau đó đều là leakage.
> 3. **Mỗi bước biến đổi phải có chart validate** — biến đổi không có bằng chứng bằng 0.

---

## 📑 Table of Contents

| # | Phase | Nội dung |
|---|---|---|
| 0 | Setup | Môi trường & Load Raw Data |
| 1 | Integration | Merge & Aggregate Relational Tables |
| 2 | Cleaning | Filter, Target Engineering & Outlier Audit |
| 3 | Feature Engineering | 8 nhóm feature từ domain knowledge |
| 4 | Leakage Audit | Kiểm tra toàn diện trước khi Split |
| 5 | Train/Val/Test Split | Temporal Split — mô phỏng thực tế |
| 6 | Missing & Encoding | Fit chỉ trên Train |
| 7 | Scaling | RobustScaler — Fit chỉ trên Train |
| 8 | Multicollinearity | VIF Check & Feature Pruning |
| 9 | Baseline Sanity Check | OLS + RandomForest Feature Importance |
| 10 | Export | Lưu artifacts cho Model Notebook |

---

## Phase 0 — Setup & Load Raw Data

**Mục tiêu:** Khởi tạo môi trường, load tất cả 9 bảng, in bản đồ kích thước tổng quan.

- [ ] Import thư viện: `numpy`, `pandas`, `matplotlib`, `seaborn`, `scipy`, `statsmodels`, `sklearn`, `category_encoders`
- [ ] Load 9 file CSV vào dict `{tên_bảng: DataFrame}` — in `.shape` + `.dtypes` từng bảng
- [ ] Vẽ bảng tổng quan Missing Value heatmap (`seaborn.heatmap(df.isnull())`) cho từng bảng raw — chụp ảnh trạng thái gốc trước khi xử lý

**📌 Validate:** Table summary `(table_name | rows | cols | null_count | null_pct)` — in dạng DataFrame đẹp.

---

## Phase 1 — Integration: Merge & Aggregate

### 1.1 Merge Relational Tables

**Thứ tự merge (quan trọng — sai thứ tự sẽ phình data):**

```
orders
  └─ LEFT JOIN order_items       ON order_id
       └─ LEFT JOIN products     ON product_id
            └─ LEFT JOIN translation ON product_category_name
                 └─ LEFT JOIN sellers    ON seller_id
                      └─ LEFT JOIN customers ON customer_id
                           └─ LEFT JOIN payments  ON order_id  ← (aggregated trước)
                                └─ LEFT JOIN reviews   ON order_id  ← (aggregated trước)
```

> ⚠️ **Bảng `payments` và `reviews`** phải được **aggregate trước khi merge** vì 1 order có nhiều payment installments và nhiều reviews — merge thẳng sẽ tạo duplicate rows làm phình data.

- [ ] Aggregate `payments`: `groupby('order_id')` → `sum(payment_value)`, `max(payment_installments)`, `mode(payment_type)`
- [ ] Aggregate `reviews`: `groupby('order_id')` → `mean(review_score)`, lấy review mới nhất nếu nhiều review
- [ ] Merge toàn bộ theo thứ tự trên bằng `how='left'`

### 1.2 Aggregate theo Order (Groupby)

> **Vấn đề:** Sau khi merge với `order_items`, mỗi item trong order tạo 1 dòng riêng → 1 order có 3 items = 3 dòng. Phải gom lại.

- [ ] `groupby('order_id')` với các agg function:
  - `price`: `sum` → `total_price`
  - `freight_value`: `sum` → `total_freight`
  - `product_id`: `count` → `item_count`
  - `product_category_name_english`: `lambda x: x.mode()[0]` → lấy category của item có giá trị cao nhất (dominant category)
  - `product_weight_g`, `product_length_cm`, `product_height_cm`, `product_width_cm`: `mean` → đặc trưng trung bình kiện hàng trong đơn

- [ ] **📌 Validate:**
  - Assert: `len(df_agg) == df['order_id'].nunique()` → phải bằng nhau
  - Bar chart: "Số dòng Trước Groupby vs Sau Groupby"

---

## Phase 2 — Cleaning: Filter, Target & Outlier Audit

### 2.1 Filter Order Status

- [ ] Chỉ giữ `order_status == 'delivered'` — drop cột sau khi filter
- [ ] Drop các dòng `order_delivered_customer_date` là null
- [ ] **📌 Validate:** Bar chart phân bổ `order_status` **trước khi drop** — chứng minh tỷ lệ non-delivered < 5%

### 2.2 Convert Timestamp

- [ ] Convert toàn bộ cột thời gian sang `pd.to_datetime`:
  - `order_purchase_timestamp`
  - `order_approved_at`
  - `order_delivered_carrier_date`
  - `order_delivered_customer_date`
  - `order_estimated_delivery_date`

### 2.3 Khởi tạo Biến Mục Tiêu (Target Engineering)

- [ ] Tạo `delivery_time_days = (order_delivered_customer_date − order_purchase_timestamp).dt.total_seconds() / 86400`
- [ ] Filter lỗi logic: drop `delivery_time_days <= 0`
- [ ] **📌 Validate — Bộ 3 chart bắt buộc:**
  1. **Histogram + KDE** của `delivery_time_days` raw — ghi rõ Mean, Median, Skewness
  2. **Boxplot** xác định ngưỡng outlier cực đoan (`Q3 + 1.5*IQR`) — thường rơi vào ~45-60 ngày
  3. **Shapiro-Wilk test** (nếu sample > 5000 thì dùng D'Agostino K² test) → kết luận phân phối
  4. **Side-by-side Histogram:** `delivery_time_raw` vs `delivery_time_log1p` → justify quyết định log transform

> ⚠️ **Quyết định tại đây:** Cutoff outlier ở đâu (90-day rule? IQR rule?). Ghi rõ lý do vào Markdown cell. Đơn > 90 ngày khả năng cao là lỗi hệ thống, không phải logistics thực tế.

---

## Phase 3 — Feature Engineering (8 nhóm)

> **Quy tắc:** Tất cả feature engineering ở đây chỉ dùng thông tin **có tại thời điểm đặt hàng**. Kiểm tra kỹ mục Phase 4 - Leakage Audit sau khi làm xong.

### FE-1: Geospatial Distance (Haversine)

- [ ] Merge tọa độ từ `geolocation` (gộp duplicate zip codes bằng `mean lat/lon`)
- [ ] Tính `distance_km` bằng công thức Haversine
- [ ] Drop `customer_lat`, `customer_lng`, `seller_lat`, `seller_lng` sau khi tính xong
- [ ] **📌 Validate:** Scatter `distance_km` vs `delivery_time_days` — tô màu theo `customer_state`

### FE-2: Spatial Flags

- [ ] `is_intra_state`: 1 nếu `customer_state == seller_state`
- [ ] `is_sp_seller`: 1 nếu seller ở São Paulo (trung tâm logistics lớn nhất) → thường giao nhanh hơn
- [ ] `is_remote_state`: 1 nếu `customer_state` thuộc danh sách bang vùng sâu (`['AM', 'RR', 'AP', 'AC', 'RO', 'PA', 'TO']`) — infrastructure yếu, delay không tuyến tính
- [ ] **📌 Validate:** 3 Boxplot: `delivery_time` phân theo từng flag (0 vs 1) — chứng minh sự khác biệt có ý nghĩa thống kê (Mann-Whitney U test, ghi p-value)

### FE-3: Temporal Cyclical Encoding

- [ ] Trích xuất từ `order_purchase_timestamp`:
  - `order_hour`, `order_weekday`, `order_month`, `order_year`
  - `is_weekend`: 1 nếu weekday ∈ {5, 6}
- [ ] Cyclical encoding: `hour_sin = sin(2π * hour/24)`, `hour_cos = cos(...)` — tương tự cho `month`
- [ ] **📌 Validate:** Polar scatter plot `order_hour` theo vòng tròn 24h — chứng minh tại sao phải dùng sin/cos thay vì integer raw

### FE-4: Holiday & Seasonal Flags

- [ ] `is_black_friday`: 1 nếu order rơi vào tuần cuối tháng 11
- [ ] `is_carnival`: 1 nếu order rơi vào tuần Carnival Brazil (thường T2 tháng 2-3)
- [ ] `days_to_next_holiday`: số ngày từ ngày đặt đến kỳ nghỉ gần nhất
- [ ] **📌 Validate:** Line chart `avg_delivery_time` theo tháng (12 tháng) — highlight T11 spike và T2 Carnival

### FE-5: Product Physical Attributes

- [ ] `product_volume_cm3 = length * width * height`
- [ ] `density = weight_g / volume_cm3` (nếu volume = 0 thì fill NaN)
- [ ] `is_heavy`: 1 nếu `weight_g > percentile 75` (hàng nặng cần xe tải riêng)
- [ ] Drop `length`, `width`, `height` sau khi tính xong (tránh multicollinearity)

### FE-6: Seller Performance Features

> ⚠️ **Leakage Alert:** Feature này phải được tính **chỉ trên Training set** rồi map sang Val/Test. Tính trên toàn bộ data sẽ bị leakage vì dùng thông tin từ tương lai.

- [ ] Tính trên train sau khi split: `seller_avg_delivery_days`, `seller_order_count`, `seller_on_time_rate` (% đơn giao trước estimated date)
- [ ] Map vào Val/Test bằng `seller_id` → điền median toàn train cho seller mới chưa gặp

### FE-7: System Estimate Features (Rất quan trọng)

- [ ] `estimated_delivery_duration = (order_estimated_delivery_date − order_purchase_timestamp).dt.days` → Đây là **dự báo của chính hệ thống Olist**. Feature power cực mạnh.
- [ ] `promise_buffer = estimated_delivery_duration − distance_km / avg_km_per_day` → Olist đang "hứa" thêm bao nhiêu ngày buffer so với khoảng cách thực tế?
- [ ] **📌 Validate:** Scatter `estimated_delivery_duration` vs `delivery_time_days` — kỳ vọng correlation r > 0.6

### FE-8: Order Complexity & Payment Features

- [ ] `order_complexity = item_count * n_unique_categories` (đơn nhiều loại hàng = gom từ nhiều kho = delay cao hơn)
- [ ] `payment_type_encoded`: boleto (thanh toán phiếu in) thường mất 1-3 ngày được confirm → encode riêng
- [ ] `payment_installments`: số kỳ trả góp (có thể liên quan đến loại hàng luxury → đặc trưng gián tiếp)
- [ ] **📌 Validate:** Bar chart `avg_delivery_time` theo `payment_type` — boleto có delay cao hơn credit_card không?

---

## Phase 4 — Leakage Audit (Bước Không Thể Bỏ Qua)

> Mục tiêu: Đảm bảo mô hình chỉ "biết" những gì có thể biết tại thời điểm khách đặt hàng.

**Danh sách cột PHẢI DROP trước khi đưa vào model:**

| Cột | Lý do |
|---|---|
| `order_delivered_customer_date` | Chính là target, không được dùng làm feature |
| `order_delivered_carrier_date` | Xảy ra sau khi đặt hàng — leakage |
| `order_approved_at` | Có thể giữ nếu tính làm `approval_delay`, sau đó drop timestamp gốc |
| `order_estimated_delivery_date` | Giữ lại SAU KHI đã tính `estimated_delivery_duration` — rồi drop |
| `review_score` | Review được viết SAU khi nhận hàng — leakage nặng |
| `review_creation_date` | Tương tự |

- [ ] Kiểm tra từng cột còn lại: "Cột này có tồn tại vào lúc khách bấm Order không?"
- [ ] In ra danh sách cột cuối cùng trước khi split — confirm với team

---

## Phase 5 — Train / Val / Test Split (Temporal Split)

> ⚠️ **Không dùng Random Split** cho bài toán này. Dữ liệu có tính mùa vụ và trend theo thời gian. Dùng random split để train trên data tháng 12 và test trên data tháng 3 là **mô phỏng sai thực tế**.

- [ ] Sort toàn bộ dataset theo `order_purchase_timestamp` tăng dần
- [ ] Cutoff:
  - **Train:** 70% đầu (dữ liệu cũ nhất)
  - **Val:** 10% tiếp theo
  - **Test:** 20% cuối (dữ liệu mới nhất)
- [ ] Tách `X` và `y` (`delivery_time_log = np.log1p(delivery_time_days)`)
- [ ] **📌 Validate:**
  - Print phân bổ thời gian của 3 tập: `(min_date, max_date)` → phải không chồng lên nhau
  - Histogram `y_train` vs `y_test` — phân phối target phải tương đồng (không bị distribution shift cực đoan)

---

## Phase 6 — Missing Value Handling & Encoding (Fit trên Train Only)

### 6.1 Xử lý Missing Values

- [ ] **Numeric columns** (`weight`, `length`, `width`, `height`):
  - Fit `groupby('product_category').median()` **chỉ trên `X_train`**
  - Transform lên cả 3 tập
  - Fallback: global median của train nếu category không có trong train
- [ ] **Categorical columns** (`product_category_name_english`): fill `'unknown'`
- [ ] **📌 Validate:** `assert X_train.isnull().sum().sum() == 0`

### 6.2 Group Sparse Categories

- [ ] Tính frequency của từng category **chỉ trên train** — nhóm category có count < threshold (ví dụ: < 100 đơn) vào `'other'`
- [ ] Apply cùng mapping lên Val/Test

### 6.3 Target Encoding (K-Fold — Chống Leakage)

> ⚠️ **Không dùng TargetEncoder thông thường fit trên toàn train** — vẫn có in-sample leakage. Phải dùng K-Fold Target Encoding (out-of-fold mean).

- [ ] Dùng `category_encoders.TargetEncoder(smoothing=10)` với cross_val strategy:
  - Cho `product_category_name_english`
  - Fit **chỉ trên train** bằng k-fold out-of-fold scheme để tránh target leakage
  - Transform Val/Test bằng global mean của train (no leakage)
- [ ] **Không** dùng Label Encoding cho `customer_state` / `seller_state` — đây là nominal variable, dùng Target Encoding tương tự

---

## Phase 7 — Scaling (Fit trên Train Only)

- [ ] Dùng `RobustScaler` (ít nhạy cảm với outlier hơn StandardScaler — phù hợp với data logistics có đuôi dài)
- [ ] Fit **ONLY** trên `X_train`
- [ ] Transform `X_train`, `X_val`, `X_test`
- [ ] **📌 Validate:** Histogram số cột trước và sau scale — phân phối shape phải tương đồng (chỉ shift về 0)

---

## Phase 8 — Multicollinearity Check (VIF)

> Pearson heatmap **không đủ** để phát hiện đa cộng tuyến trong regression. Cần VIF.

- [ ] Tính VIF cho toàn bộ feature của `X_train`:
  ```
  from statsmodels.stats.outliers_influence import variance_inflation_factor
  ```
- [ ] **Ngưỡng quyết định:** VIF > 10 → cần xử lý
- [ ] Các cặp nguy hiểm cần kiểm tra đặc biệt:
  - `distance_km` ↔ `is_intra_state` ↔ `total_freight` → thường tương quan cao
  - `product_volume_cm3` ↔ `density` (nếu weight gần hằng số)
  - `estimated_delivery_duration` ↔ `distance_km` (Olist tính estimate dựa vào distance)
- [ ] Với từng cặp VIF > 10: giữ 1 trong 2 cột (ưu tiên giữ cột có correlation cao hơn với target)
- [ ] **📌 Validate:** 
  - Heatmap Pearson của final feature set (sau khi drop)
  - Bar chart VIF score của từng feature — highlight ngưỡng 10

---

## Phase 9 — Baseline Sanity Check

### 9.1 Feature Importance (RandomForest — 30 giây)

- [ ] Fit `RandomForestRegressor(n_estimators=100, random_state=42)` trên `X_train`
- [ ] Plot `feature_importances_` → Bar chart Top 15 features
- [ ] **Kỳ vọng:** `estimated_delivery_duration`, `distance_km`, `seller_lead_time` phải trong Top 5 — nếu không, cần review lại pipeline

### 9.2 OLS Residual Check

- [ ] Fit OLS đơn giản trên `X_train`
- [ ] Plot Residuals vs Fitted:
  - Hình phễu (V-shape) → Heteroscedasticity → ghi chú vào Notebook: "Cần dùng WLS hoặc MAE loss trong model"
  - Phân tán đều quanh 0 → Homoscedasticity → Linear Regression là phù hợp
- [ ] **Đây là bằng chứng justify toàn bộ Model phase tiếp theo**

---

## Phase 10 — Export Artifacts

- [ ] Lưu tất cả bằng `.pkl` (load nhanh hơn CSV):
  - `X_train.pkl`, `X_val.pkl`, `X_test.pkl`
  - `y_train.pkl`, `y_val.pkl`, `y_test.pkl`
  - `scaler.pkl` (để inverse transform prediction về đơn vị ngày)
  - `target_encoder.pkl` (để encode data mới trong production)
  - `feature_names.txt` (list tên cột theo đúng thứ tự)
- [ ] In ra **Final Feature Summary Table:**

| Feature | Type | Source | VIF | Pearson r với target |
|---|---|---|---|---|
| `estimated_delivery_duration` | Numeric | System estimate | ... | ... |
| `distance_km` | Numeric | Haversine | ... | ... |
| `seller_lead_time` | Numeric | Derived | ... | ... |
| ... | | | | |

---

## ⚠️ Leakage Prevention Checklist — Ký tên trước khi nộp

- [ ] Không có cột nào phát sinh sau `order_purchase_timestamp` còn trong X
- [ ] Target Encoding và Scaler đều được fit **chỉ** trên `X_train`
- [ ] `seller_avg_delivery_days` được tính chỉ từ train, rồi map sang val/test
- [ ] `order_delivered_customer_date` đã được drop khỏi X
- [ ] `review_score` đã được drop khỏi X
- [ ] Split là Temporal Split, không phải Random Split
- [ ] `random_state` được chốt cứng ở mọi bước có randomness

---

## 📊 EDA Insight Requirements (Checklist cho Báo cáo)

Các biểu đồ bắt buộc phải có trong notebook (phục vụ phần báo cáo):

| # | Biểu đồ | Insight kỳ vọng |
|---|---|---|
| 1 | Distribution of `delivery_time_days` (Histogram + KDE) | Right-skewed, cần log transform |
| 2 | `delivery_time` vs `distance_km` (Scatter) | Tương quan dương rõ ràng |
| 3 | `delivery_time` theo `customer_state` (Boxplot) | Bang miền Bắc delay cao hơn |
| 4 | `delivery_time` theo `product_category` (Top 15 Bar) | Furniture/Heavy items trễ hơn |
| 5 | `delivery_time` theo `payment_type` (Boxplot) | Boleto trễ hơn credit card |
| 6 | `delivery_time` theo tháng (Line chart) | Spike T11 (Black Friday) |
| 7 | Correlation Heatmap của final features | Không có cặp > 0.9 |
| 8 | VIF Bar chart | Không có feature > 10 |
| 9 | `estimated_delivery_duration` vs `delivery_time` (Scatter) | r > 0.6 |
| 10 | Residual plot của OLS baseline | Justify model choice |
