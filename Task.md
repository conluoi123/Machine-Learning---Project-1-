# Kế Hoạch Tuần 1: Data Preprocessing & Exploratory Data Analysis (EDA)

**Mục tiêu tuần 1:**

- Hoàn thành Data Preprocessing và Integration (Nối bảng).
- Rút trích và tạo feature cơ bản (Feature Engineering), đặc biệt là target `delivery_time`.
- Thực hiện EDA chi tiết để lấy insight làm báo cáo.
- **Dữ liệu sử dụng:** Brazilian E-Commerce Public Dataset by Olist.
- **Mục tiêu của Model (Regression):** Dự đoán `delivery_time = order_delivered_customer_date - order_purchase_timestamp`.

**Phân chia vai trò:**

- **Người A:** Phụ trách Data Preprocessing & Data Integration.
- **Người B:** Phụ trách EDA & Data Visualization.
  _(Cả hai hỗ trợ nhau trong quá trình thực hiện và phân tích)._

---

## 📅 Bảng Theo Dõi Task Tuần 1

|  Task   | Nội dung                                       | Phụ trách | Trạng thái |
| :-----: | :--------------------------------------------- | :-------: | :--------: |
| **T1**  | Download dataset + load CSV                    |     A     |     ⬜     |
| **T2**  | Data overview (shape, columns)                 |     A     |     ⬜     |
| **T3**  | Check missing values                           |     A     |     ⬜     |
| **T4**  | Check duplicate rows                           |     A     |     ⬜     |
| **T5**  | Convert timestamp format                       |     A     |     ⬜     |
| **T6**  | Join các bảng dataset                          |     A     |     ⬜     |
| **T7**  | Tạo feature `delivery_time`                    |     A     |     ⬜     |
| **T8**  | Remove invalid rows                            |     A     |     ⬜     |
| **T9**  | Tạo dataset ML ban đầu                         |     A     |     ⬜     |
| **T10** | Distribution analysis (Phân tích phân phối)    |     B     |     ⬜     |
| **T11** | Time-based analysis (Phân tích theo thời gian) |     B     |     ⬜     |
| **T12** | Distance analysis (Phân tích khoảng cách)      |     B     |     ⬜     |
| **T13** | Category analysis (Phân tích theo danh mục)    |     B     |     ⬜     |
| **T14** | Geography analysis (Phân tích địa lý)          |     B     |     ⬜     |
| **T15** | Correlation analysis (Phân tích tương quan)    |     B     |     ⬜     |
| **T16** | Outlier analysis (Kiểm tra ngoại lai)          |   A + B   |     ⬜     |
| **T17** | EDA insights summary (Tổng hợp insight)        |   A + B   |     ⬜     |

---

## 🛠 Chi Tiết Công Việc (Hướng Dẫn Cụ Thể)

### Dành cho Người A (Data Preprocessing & Integration)

#### **T1 – Load Dataset**

- **Yêu cầu:** Load các file csv cần thiết bằng pandas. Tên file thực tế trong folder `data`:
- **Files chính:**
  - `olist_orders_dataset.csv`
  - `olist_order_items_dataset.csv`
  - `olist_products_dataset.csv`
  - `olist_customers_dataset.csv`
  - `olist_sellers_dataset.csv`
  - `olist_order_reviews_dataset.csv`
  - `olist_geolocation_dataset.csv`
  - `olist_order_payments_dataset.csv` (Có thể dùng thêm để lấy thông tin thanh toán nếu cần thiết cho Model)
  - `product_category_name_translation.csv` (Dùng để dịch tên nhóm hàng hóa từ tiếng Bồ Đào Nha sang tiếng Anh)
- **Output:** Kiểm tra thử bằng cách in ra `.shape` và `.columns` của từng DataFrame.

#### **T2 – Data Overview**

- **Yêu cầu:** Lập một bảng mô tả tổng quan xem mỗi bảng có bao nhiêu dòng, cột.
- **Ví dụ:**
  | Table | Rows | Columns |
  | :--- | :--- | :--- |
  | orders | ~100k | 8 |
  | order_items | ~112k | 7 |

#### **T3 – Missing Values**

- **Yêu cầu:** Kiểm tra số lượng null trong từng bảng.
- **Code:** `df.isnull().sum()`
- **Output:** Danh sách các cột bị thiếu dữ liệu và số lượng thiếu. Từ đó đưa ra quyết định drop hay fill.

#### **T4 – Duplicate Rows**

- **Yêu cầu:** Kiểm tra các dòng trùng lặp hoàn toàn.
- **Code:** `df.duplicated().sum()`
- **Hành động:** Loại bỏ duplicates nếu có.

#### **T5 – Convert Timestamp**

- **Yêu cầu:** Chuyển đổi các cột thời gian dạng object/string sang `datetime`.
- **Các cột cần xử lý:**
  - `order_purchase_timestamp`
  - `order_delivered_customer_date`
  - `order_estimated_delivery_date`
- **Code:** `pd.to_datetime(df['col_name'])`

#### **T6 – Join Dataset (Task Quan Trọng Nhất)**

- **Yêu cầu:** Nối (Merge) các bảng lại thành một bảng master dữ liệu.
- **Thứ tự Join tham khảo:**
  1. `olist_orders_dataset` ➔ `olist_order_items_dataset` (thông qua `order_id`)
  2. ➔ `olist_products_dataset` (thông qua `product_id`)
  3. ➔ `product_category_name_translation` (thông qua `product_category_name` để lấy cột `product_category_name_english`)
  4. ➔ `olist_sellers_dataset` (thông qua `seller_id`)
  5. ➔ `olist_customers_dataset` (thông qua `customer_id`)
  6. ➔ `olist_order_reviews_dataset` (thông qua `order_id`, lưu ý check duplicate vì 1 order có thể có nhiều review)
- **Output:** Tạo ra biến `master_dataset` chứa đầy đủ thông tin (Khoảng ~100k rows và \>20 columns). Chú ý sử dụng `how='left'` hoặc `how='inner'` cho phù hợp.

#### **T7 – Target Engineering**

- **Yêu cầu:** Tạo cột biến mục tiêu `delivery_time`.
- **Công thức:** `delivery_time = order_delivered_customer_date - order_purchase_timestamp`
- **Hành động:** Convert kết quả timedelta này sang số ngày (days) thực.

#### **T8 – Remove Invalid Data**

- **Yêu cầu:** Lọc bỏ các dữ liệu lỗi trong biến mục tiêu.
- **Hành động:** Remove các dòng có `delivery_time < 0` (lỗi logic thời gian) và `delivery_time > 60` (quá lâu, có thể là outlier cực đoan cản trở regression ban đầu).

#### **T9 – Create ML Dataset**

- **Yêu cầu:** Chốt lại bộ dataset cuối cùng cho mô hình học máy.
- **Các tính năng lý tưởng nên có:**
  | Feature | Type | Ghi chú |
  | :--- | :--- | :--- |
  | `delivery_time` | Target | Số ngày giao (numeric) |
  | `price` | Numeric | Giá trị đơn hàng |
  | `freight_value` | Numeric | Phí vận chuyển |
  | `items_per_order` | Numeric | Số lượng item trong 1 đơn |
  | `product_category` | Categorical | Tên danh mục (đã dịch sang tiếng Anh) |
  | `customer_state` | Categorical | Bang của khách |
  | `seller_state` | Categorical | Bang của seller |
  | `distance_km` | Numeric | Khoảng cách theo lat/long |
  | `order_hour` | Numeric | Giờ đặt hàng |
  | `order_weekday` | Numeric | Thứ đặt hàng |

---

### Dành cho Người B (EDA & Data Visualization)

_Note: Người B sử dụng `master_dataset` mà Người A đã xử lý để vẽ chart._

#### **T10 – Target Distribution**

- **Yêu cầu:** Vẽ biểu đồ Histogram để xem phân phối của biến `delivery_time`.
- **Phân tích bổ sung:** Tính Mean, Median, Skewness (Độ lệch) của biến target để xem có cần transform (ví dụ Log Transform) ở các tuần sau không.

#### **T11 – Time-based Analysis**

- **Yêu cầu:** Phân tích sự ảnh hưởng của yếu tố thời gian đến thời gian giao hàng.
- **Tạo Feature:** Trích xuất giờ (`order_hour`), thứ (`order_weekday`), tháng (`order_month`) từ `order_purchase_timestamp`.
- **Vẽ Chart:**
  - Bar plot: `delivery_time` trung bình theo `order_weekday`.
  - Line chart: Số lượng đơn hàng hoặc thời gian giao trung bình theo các tháng.

#### **T12 – Distance Analysis**

- **Yêu cầu:** Khảo sát ảnh hưởng của khoảng cách địa lý.
- **Tính toán:** Dựa trên cột vĩ độ/kinh độ (`latitude`, `longitude`) của seller và customer, tính khoảng cách `distance_km` (có thể dùng công thức Haversine).
- **Vẽ Chart:** Scatter plot giữa `distance_km` và `delivery_time`.

#### **T13 – Product Category Analysis**

- **Yêu cầu:** Phân tích ảnh hưởng của loại sản phẩm.
- **Vẽ Chart:** Bar plot hiển thị `avg_delivery_time` theo từng `product_category_name_english`. (Chỉ nên plot top 10 - top 15 danh mục nhiều đơn nhất hoặc thời gian giao lâu nhất).

#### **T14 – Geography Analysis**

- **Yêu cầu:** Thời gian giao hàng phụ thuộc vào địa lý.
- **Vẽ Chart:** Bar chart cho `delivery_time` trung bình theo `customer_state` (Bang của khách).
- **Kỳ vọng Insight:** Các vùng xa trung tâm hoặc khác bang với seller sẽ có thời gian vận chuyển dài hơn.

#### **T15 – Correlation Analysis**

- **Yêu cầu:** Xem các biến số (numeric) có quan hệ như thế nào với mục tiêu.
- **Tính toán:** Lấy các cột số (`price`, `freight_value`, `distance_km`, `items_per_order`, `delivery_time`,...).
- **Vẽ Chart:** Correlation Heatmap (Dùng Seaborn). Xem hệ số Pearson.

#### **T16 – Outlier Analysis (A + B cùng làm)**

- **Yêu cầu:** Chẩn đoán thử những đơn hàng giao quá chậm.
- **Hành động:** Lọc tập dữ liệu con có `delivery_time > 40` ngày.
- **Phân tích:** Xem nguyên nhân do đâu (Do category đặc thù? Khoảng cách quá xa? Hay do một số sellers cụ thể gửi hàng chậm bưu cục?).

#### **T17 – EDA Insight Summary (A + B cùng làm)**

- **Yêu cầu:** Tổng hợp lại 5-10 insight thiết thực nhất từ các biểu đồ trên, đưa vào Markdown của báo cáo.
- **Ví dụ Insight:**
  1. _Distance có tương quan tuyến tính cao nhất với thời gian giao nhận._
  2. _Một số chủng loại hàng nặng/cồng kềnh có thời gian giao lâu hơn hẳn._
  3. _Đơn đặt vào cuối tuần có xu hướng bị delay lâu hơn trong việc đóng gói ban đầu._
  4. _Phân phối delivery time bị lệch phải (right-skewed), đa số nhận từ 5-10 ngày nhưng cái đuôi outlier kéo dài đến cả tháng._

---

## 📅 Lịch Trình Thực Hiện Gợi Ý (7 Ngày)

|   Ngày    | Công việc trọng tâm                                                                                                                |
| :-------: | :--------------------------------------------------------------------------------------------------------------------------------- |
| **Day 1** | T1, T2 - Load dataset, đọc hiểu cấu trúc data, lập bảng overview summary.                                                          |
| **Day 2** | T3, T4, T5 - Chuyển kiểu dữ liệu thời gian, kiểm tra missing, duplicates.                                                          |
| **Day 3** | T6 - Tập trung viết query/code Join toàn bộ các bảng vào một Master DataFrame chuẩn.                                               |
| **Day 4** | T7, T8, T9, T11, T12 - Tạo feature thời gian, tính `distance_km`, tính `delivery_time`, lọc rác dữ liệu. Chốt bộ ML dataset chuẩn. |
| **Day 5** | T10, T13, T14 - Plot các phân phối Target, Địa lý, Category.                                                                       |
| **Day 6** | T15, T16 - Chạy Correlation, phân tích Outliers chuyên sâu. Build hoàn thiện các file hình ảnh.                                    |
| **Day 7** | T17 - Họp nhóm thống nhất Insight. Clean up code trong Notebook để chuẩn bị nộp tuần 1.                                            |

---

## 🎯 Deliverables Cuối Tuần

Hoàn thành tuần 1, nhóm phải có đẩy đủ các tài nguyên sau trong source code:

1. **`EDA.ipynb`**: Notebook chứa toàn bộ script nối bảng, kiểm tra dữ liệu, và các đoạn code vẽ biểu đồ (Sạch, có comment rõ ràng).
2. **`processed_dataset.csv`**: File dataset đã join và lọc hoàn chỉnh, sẵn sàng đem vào model ở tuần 2.
3. **Thư mục Images/Plots**: Export ra được `10-15` biểu đồ rõ nét, có tiêu đề, chú thích đầy đủ.
4. **Báo cáo sơ bộ**: Có liệt kê sẵn các insight chính rút ra chuẩn bị chèn vào Slide/Word báo cáo của Project.

---

## 🌟 Ý Tưởng Nâng Cao (Bonus)

Nếu 2 bạn hoàn thành lịch trình sớm hoặc muốn đầu tư mạnh cho Portfolio/CV/Phỏng vấn sau này, hãy cân nhắc chọn 1-2 ý tưởng dưới đây:

### 1. Advanced Geospatial Analytics (Mapping)

- **Kỹ thuật:** Thay vì chỉ tính distance bằng toán học, hãy dùng thư viện `Folium` hoặc `GeoPandas` vẽ luôn bản đồ Brazil thực tế. Tạo Heatmap (bản đồ nhiệt) những các tuyến ship hàng chậm nhất.
- **Tại sao ghi điểm:** Kỹ năng phân tích Geospatial (dữ liệu không gian, lat/long) cực kỳ hiếm và rất được lòng các công ty Logistics (như Shopee, Lazada, GHTK).

### 2. Feature: Lịch sử và Ngày Lễ (Temporal Dynamics)

- **Kỹ thuật:** Cào thêm dữ liệu các ngày lễ của Brazil (Carnival, Black Friday, Christmas) năm 2017-2018. Tạo cột `is_holiday_season`. Hoặc tự tạo biến `seller_avg_delivery_past_30_days` (Tốc độ TB của seller đó trong 1 tháng vừa qua).
- **Tại sao ghi điểm:** Rất thực tế! Khối lượng hàng ở các kho vào ngày Black Friday bị kẹt là chuyện hiển nhiên, model bạn học được điều này sẽ ăn điểm tuyệt đối về "Domain Knowledge" (hiểu biết kinh doanh).

### 3. Phân tích Dữ liệu Hàng Hóa Dạng Khối (Dimensionality)

- **Kỹ thuật:** Bảng `olist_products_dataset` có các cột `product_weight_g` (khối lượng), `product_length_cm`, `width`... Hãy tính Thể Tích Của Món Hàng (`volume = L x W x H`).
- **Tại sao ghi điểm:** Hàng "siêu trường siêu trọng" tốn thời gian tìm xe tải riêng để ship. Đây là một Insight tuyệt vời nếu bạn chứng minh được bằng biểu đồ trong quá trình EDA.

### 4. Giải Thích AI - Explainable AI (Dành cho Tuần Build Model)

- **Kỹ thuật:** Cài thư viện `SHAP`. Sau khi huấn luyện mô hình Regression xong, dùng SHAP vẽ các biểu đồ phân tích Feature Importance cá nhân được.
- **Tại sao ghi điểm:** Nhà tuyển dụng thích những ứng viên có khả năng giải thích "Tại sao AI nghĩ đơn hàng này mất 25 ngày để giao? À, vì SHAP chỉ ra rằng Seller ở quá xa + Lại đặt vào T7, CN".
