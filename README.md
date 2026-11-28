<a id="top"></a>

# Dự đoán Phân khúc Giá Thiết bị Di động (Mobile Price Classification)

![Python](https://img.shields.io/badge/Python-3.12-blue?logo=python&logoColor=white)
![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-Linear%20SVM-F7931E?logo=scikitlearn&logoColor=white)
![Streamlit](https://img.shields.io/badge/Streamlit-Web%20App-FF4B4B?logo=streamlit&logoColor=white)
![Accuracy](https://img.shields.io/badge/Accuracy-97.00%25-brightgreen)

Hệ thống Học máy phân loại phân khúc giá điện thoại di động thành 4 nhóm từ phổ thông đến cao cấp dựa trên các thông số cấu hình phần cứng. Dự án được tổ chức theo kiến trúc pipeline module hóa và triển khai giao diện tương tác trực tiếp bằng **Streamlit**.

---

## Mục lục

- [1. Giới thiệu đề tài](#1-giới-thiệu-đề-tài)
- [2. Cấu trúc thư mục dự án](#2-cấu-trúc-thư-mục-dự-án)
- [3. Quy trình xử lý và mô hình hóa](#3-quy-trình-xử-lý-và-mô-hình-hóa)
- [4. Kết quả thực nghiệm](#4-kết-quả-thực-nghiệm)
- [5. Hướng dẫn cài đặt và chạy ứng dụng](#5-hướng-dẫn-cài-đặt-và-chạy-ứng-dụng)
- [6. Triển khai ứng dụng web](#6-triển-khai-ứng-dụng-web)
- [7. Thông tin đồ án](#7-thông-tin-đồ-án)

---

## 1. Giới thiệu đề tài

Trong thị trường thiết bị di động cạnh tranh cao, việc định vị phân khúc giá dựa trên thông số kỹ thuật phần cứng giúp nhà sản xuất và đơn vị bán lẻ tối ưu hóa chiến lược sản phẩm. Bài toán sử dụng tập dữ liệu gồm **2.000 mẫu thiết bị** với **20 thuộc tính phần cứng** đầu vào (RAM, dung lượng pin, bộ nhớ trong, độ phân giải màn hình, camera, kết nối...) để phân loại vào 4 nhãn mục tiêu (`price_range`):

* **Mức 0 (`Low`):** Phân khúc phổ thông.
* **Mức 1 (`Medium`):** Phân khúc tầm trung.
* **Mức 2 (`High`):** Phân khúc cận cao cấp.
* **Mức 3 (`Very High`):** Phân khúc cao cấp.

[↑ Quay lại đầu trang](#top)

---

## 2. Cấu trúc thư mục dự án

```text
Mobile-Price-Classification/
├── config/
│   └── params.yaml                      # Tham số cấu hình pipeline và siêu tham số mô hình
├── data/
│   ├── external/                        # Dữ liệu bổ sung bên ngoài (nếu có)
│   ├── processed/                       # Dữ liệu sạch sau tiền xử lý (train_cleaned.csv)
│   └── raw/                             # Dữ liệu thô ban đầu (train.csv)
├── models/
│   ├── best_svm_model.joblib            # Trọng số mô hình Linear SVM đã huấn luyện
│   └── scaler.joblib                    # Bộ chuẩn hóa dữ liệu StandardScaler
├── notebooks/
│   ├── 01_exploratory_data_analysis.ipynb  # Phân tích khám phá dữ liệu (EDA)
│   └── 02_model_experiments.ipynb          # Thử nghiệm và so sánh các mô hình
├── reports/
│   ├── figures/                         # Biểu đồ phân tích, ma trận nhầm lẫn và sơ đồ luồng
│   └── metrics.json                     # Báo cáo chỉ số đánh giá chi tiết (JSON)
├── src/
│   └── mobile_price/
│       ├── __init__.py                  # Khởi tạo package mobile_price
│       ├── data_preprocessing.py        # Làm sạch dữ liệu và tạo đặc trưng phái sinh
│       ├── model_training.py            # Huấn luyện Linear SVM và lưu artifacts
│       ├── model_evaluation.py          # Đánh giá trên tập kiểm thử và xuất biểu đồ
│       └── utils.py                     # Các hàm tiện ích quản lý đường dẫn và đọc/ghi file
├── .gitignore                           # Cấu hình loại trừ file tạm khỏi Git
├── app.py                               # Ứng dụng web dự đoán trực quan (Streamlit)
├── draw_pipeline.py                     # Script vẽ sơ đồ luồng xử lý hệ thống
├── main.py                              # Điểm khởi chạy toàn bộ pipeline huấn luyện
├── README.md                            # Tài liệu hướng dẫn dự án
├── requirements.txt                     # Danh sách thư viện phụ thuộc
└── setup.py                             # Cấu hình đóng gói package
```

[↑ Quay lại đầu trang](#top)

---

## 3. Quy trình xử lý và mô hình hóa

1. **Làm sạch dữ liệu (`data_preprocessing.py`):**
   * Xử lý các giá trị `0` bất hợp lý về mặt vật lý ở hai thuộc tính chiều rộng màn hình (`sc_w`) và chiều cao điểm ảnh (`px_height`) bằng giá trị trung vị (`median`) của các mẫu hợp lệ (`> 0`).
2. **Kỹ nghệ đặc trưng (Feature Engineering):**
   * Tạo thêm 2 đặc trưng tương tác phản ánh không gian hiển thị thực tế:
     * `px_area = px_height * px_width` (Tổng số điểm ảnh màn hình)
     * `sc_area = sc_h * sc_w` (Diện tích bề mặt màn hình)
   * Nâng tổng số đặc trưng đầu vào từ 20 lên **22 thuộc tính**.
3. **Phân chia dữ liệu và Chuẩn hóa:**
   * Chia tập dữ liệu theo tỷ lệ **80% huấn luyện (1.600 mẫu) / 20% kiểm thử (400 mẫu)** có phân tầng (`stratify=y`, `random_state=42`).
   * Áp dụng `StandardScaler` (`fit` trên tập huấn luyện và `transform` trên tập kiểm thử) để tránh rò rỉ dữ liệu (Data Leakage).
4. **Huấn luyện mô hình (`model_training.py`):**
   * Sử dụng thuật toán **Support Vector Machine với nhân tuyến tính (`SVC(kernel='linear', C=1.0, random_state=42)`)**.

[↑ Quay lại đầu trang](#top)

---

## 4. Kết quả thực nghiệm

Đánh giá mô hình **Linear SVM** trên tập kiểm thử độc lập gồm **400 mẫu** (100 mẫu mỗi phân khúc):

| Phân khúc giá (`price_range`) | Precision | Recall | F1-score | Support |
| :--- | :---: | :---: | :---: | :---: |
| **0 (Phổ thông - Low)** | 0.99 | 0.99 | 0.99 | 100 |
| **1 (Tầm trung - Medium)** | 0.97 | 0.95 | 0.96 | 100 |
| **2 (Cận cao cấp - High)** | 0.94 | 0.96 | 0.95 | 100 |
| **3 (Cao cấp - Very High)** | 0.98 | 0.98 | 0.98 | 100 |
| **Accuracy (Độ chính xác tổng thể)** | — | — | **0.97** | **400** |
| **Macro Avg (Trung bình cộng)** | **0.97** | **0.97** | **0.97** | **400** |
| **Weighted Avg (Trung bình trọng số)** | **0.97** | **0.97** | **0.97** | **400** |

[↑ Quay lại đầu trang](#top)

---

## 5. Hướng dẫn cài đặt và chạy ứng dụng

### Bước 1: Clone repository về máy
```bash
git clone https://github.com/hongthanh179/Mobile-Price-Classification.git
cd Mobile-Price-Classification
```

### Bước 2: Tạo môi trường ảo và cài đặt thư viện
```bash
python -m venv .venv
# Kích hoạt trên Windows:
.venv\Scripts\activate
# Kích hoạt trên macOS/Linux:
source .venv/bin/activate

pip install -r requirements.txt
```

### Bước 3: Chạy lại pipeline huấn luyện và đánh giá SVM (tùy chọn)
*(Bỏ qua bước này nếu sử dụng trực tiếp trọng số đã huấn luyện sẵn trong thư mục `models/`)*
```bash
python main.py
```
Lưu ý: `main.py` chạy lại pipeline SVM và ghi đè các artifact SVM hiện có. Artifact `best_svm_model.joblib` hiện dùng `C=10`, còn `model_training.py` và `params.yaml` vẫn đặt `C=1.0`; không chạy lại pipeline nếu muốn giữ nguyên artifact và kết quả hiện tại. Để chỉ chạy phần Random Forest của Nguyễn Hoàng Thái, dùng lệnh ở mục bên dưới.

### Bước 4: Khởi chạy API và giao diện Streamlit
Mở hai terminal tại thư mục dự án. Terminal thứ nhất chạy API:
```bash
uvicorn src.mobile_price.api:app --reload
```

Terminal thứ hai chạy giao diện:
```bash
streamlit run app.py
```
Ứng dụng mở tại `http://localhost:8501`; tài liệu tương tác của API tại `http://127.0.0.1:8000/docs`.

### Thí nghiệm Decision Tree và Random Forest
Đặt dataset Kaggle tại `data/raw/train.csv`, sau đó chạy:
```bash
python -m src.mobile_price.rf_training
```
Lệnh này dùng lại `models/scaler.joblib`, thực hiện Stratified 5-Fold CV và đánh giá mô hình tốt nhất trên cùng tập test 80/20. Kết quả được lưu riêng vào `models/rf_model.joblib`, `reports/metrics_rf.json`, các bảng CSV và hình trong `reports/figures/`; nó cũng so sánh thứ hạng đặc trưng với Linear SVM. Lệnh không ghi đè model hay metrics của SVM.

Lưu ý: `.gitignore` loại trừ các tệp dữ liệu trong `data/raw/` và `data/processed/`. Nếu cần người chấm tái chạy thực nghiệm từ file ZIP, hãy kèm dataset hoặc hướng dẫn tải nguồn Kaggle.

[↑ Quay lại đầu trang](#top)

---

## 6. Triển khai ứng dụng web

Streamlit gọi FastAPI qua biến môi trường `MOBILE_PRICE_API_URL` (mặc định `http://127.0.0.1:8000/predict`). Khi triển khai lên đám mây, cần triển khai API riêng và cấu hình URL công khai của API trong phần Secrets/Environment Variables của Streamlit; địa chỉ localhost trên máy cá nhân không dùng được từ Streamlit Cloud.

[↑ Quay lại đầu trang](#top)

---

## 7. Thông tin đồ án

* **Đề tài:** Dự đoán phân khúc giá thiết bị di động (Mobile Price Classification)
* **Sinh viên thực hiện:** Nguyễn Hồng Thanh, Nguyễn Hoàng Thái
* **Công nghệ sử dụng:** Python, Scikit-learn, Pandas, NumPy, Matplotlib, Seaborn, Streamlit

[↑ Quay lại đầu trang](#top)
