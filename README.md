# BTL Nhóm 5 – Đề 16: Phân tích lỗi sản xuất & Dự đoán lỗi máy móc

**Học phần:** Dữ liệu lớn, Khai phá dữ liệu  
**Nhóm:** 5 – Lớp CNTT1707  
**Dataset:** [AI4I 2020 Predictive Maintenance Dataset (UCI)](https://archive.ics.uci.edu/ml/datasets/AI4I+2020+Predictive+Maintenance+Dataset)

---

## Mục tiêu

Xây dựng pipeline khai phá dữ liệu hoàn chỉnh trên tập dữ liệu bảo trì dự đoán AI4I 2020, bao gồm:

- **Luật kết hợp (Association Rules):** Tìm tổ hợp điều kiện cảm biến dẫn đến lỗi máy.
- **Phân cụm (Clustering):** Phân nhóm máy theo hành vi cảm biến, profiling rủi ro từng cụm.
- **Phân lớp có giám sát (Supervised):** Dự đoán lỗi máy móc (Machine failure), xử lý mất cân bằng lớp (SMOTE).
- **Bán giám sát (Semi-supervised):** Giả lập thiếu nhãn, so sánh Self-Training vs Supervised-only, phân tích rủi ro pseudo-label.
- **Hồi quy / Chuỗi thời gian (Regression):** Dự báo độ mòn dao cụ (Tool wear) theo thứ tự UDI ≈ thời gian.

---

## Cấu trúc thư mục

```
BTL_KPDL/
├── README.md
├── requirements.txt
├── .gitignore
├── configs/
│   └── params.yaml           # Tham số: seed, paths, hyperparams
├── data/
│   ├── raw/
│   │   └── ai4i2020.csv      # Dữ liệu gốc (không commit nếu quá lớn)
│   └── processed/            # Dữ liệu sau tiền xử lý
├── notebooks/
│   ├── 01_eda.ipynb                  # EDA & Tiền xử lý
│   ├── 02_data_preprocessing.ipynb   # Chi tiết tiền xử lý + feature
│   ├── 03_mining_or_clustering.ipynb # Association Rules + Clustering
│   ├── 04_modeling.ipynb             # Supervised Learning
│   ├── 04b_semi_supervised.ipynb     # Semi-supervised Learning
│   └── 05_evaluation_report.ipynb    # Tổng hợp kết quả & insight
├── src/
│   ├── data/
│   │   ├── loader.py         # Đọc dữ liệu, kiểm tra schema
│   │   └── cleaner.py        # Xử lý thiếu, outlier, binning, encoding
│   ├── mining/
│   │   ├── association.py    # Apriori – luật kết hợp
│   │   └── clustering.py     # K-Means + Elbow + Silhouette + Profiling
│   ├── models/
│   │   ├── supervised.py     # RF, XGBoost + SMOTE + PR-AUC
│   │   ├── semi_supervised.py# Self-Training + pseudo-label risk analysis
│   │   └── forecasting.py    # Ridge, XGBoost hồi quy Tool wear
│   └── visualization/
│       └── plots.py          # Hàm vẽ dùng chung
├── scripts/
│   └── run_pipeline.py       # Chạy toàn bộ pipeline
└── outputs/
    ├── figures/              # Biểu đồ
    ├── tables/               # Bảng kết quả
    ├── models/               # File model (.pkl)
    └── reports/
        └── metrics_summary.md
```

---

## Data Dictionary – AI4I 2020

| Cột | Kiểu | Mô tả |
|-----|------|-------|
| `UDI` | int | Unique Device Identifier – dùng làm time index (UDI ≈ thứ tự thời gian) |
| `Product ID` | str | Mã sản phẩm (L/M/H prefix = chất lượng thấp/trung/cao) |
| `Type` | str | Loại sản phẩm: L (Low), M (Medium), H (High quality) |
| `Air temperature [K]` | float | Nhiệt độ không khí (Kelvin) |
| `Process temperature [K]` | float | Nhiệt độ quy trình (Kelvin) |
| `Rotational speed [rpm]` | int | Tốc độ quay (vòng/phút) |
| `Torque [Nm]` | float | Mô-men xoắn (Newton-meter) |
| `Tool wear [min]` | int | Độ mòn dao cụ (phút) – **Target cho hồi quy** |
| `Machine failure` | int (0/1) | Nhãn lỗi máy tổng hợp – **Target cho phân lớp** |
| `TWF` | int (0/1) | Tool Wear Failure (lỗi do mòn dao) |
| `HDF` | int (0/1) | Heat Dissipation Failure (lỗi tản nhiệt) |
| `PWF` | int (0/1) | Power Failure (lỗi điện) |
| `OSF` | int (0/1) | Overstrain Failure (lỗi quá tải) |
| `RNF` | int (0/1) | Random Failures (lỗi ngẫu nhiên) |

> ⚠️ **Lưu ý data leakage:** Các cột TWF, HDF, PWF, OSF, RNF là nguyên nhân trực tiếp gây ra `Machine failure`. Không được dùng chúng làm feature khi huấn luyện mô hình phân lớp `Machine failure`.

> ⚠️ **Mất cân bằng lớp:** ~96.6% mẫu là bình thường (label 0), chỉ ~3.4% là lỗi (label 1). Cần xử lý bằng SMOTE và dùng PR-AUC thay vì Accuracy.

---

## Hướng dẫn cài đặt

### 1. Clone repo và cài dependencies

```bash
git clone <repo-url>
cd BTL_KPDL
pip install -r requirements.txt
```

### 2. Tải dataset

Tải file `ai4i2020.csv` từ:
- **UCI ML Repository:** https://archive.ics.uci.edu/ml/datasets/AI4I+2020+Predictive+Maintenance+Dataset
- **Kaggle:** https://www.kaggle.com/datasets/stephanmatzka/predictive-maintenance-dataset-ai4i-2020

Đặt file vào: `data/raw/ai4i2020.csv`

### 3. Cập nhật đường dẫn (nếu cần)

Chỉnh sửa `configs/params.yaml`:

```yaml
data_paths:
  raw_data: "data/raw/ai4i2020.csv"
  processed_data: "data/processed/ai4i2020_processed.csv"
```

---

## Chạy pipeline

### Cách 1: Chạy toàn bộ pipeline tự động

```bash
python scripts/run_pipeline.py
```

Pipeline sẽ thực hiện tuần tự:
1. Load & làm sạch dữ liệu
2. Khai phá luật kết hợp (Apriori)
3. Phân cụm K-Means
4. Supervised Learning (RF + XGBoost + SMOTE)
5. Semi-supervised Learning (Self-Training)
6. Forecasting (Tool wear prediction)

Kết quả được lưu tại `outputs/`.

### Cách 2: Chạy từng notebook theo thứ tự

```
notebooks/01_eda.ipynb              → EDA & Tiền xử lý
notebooks/02_data_preprocessing.ipynb
notebooks/03_mining_or_clustering.ipynb → Association + Clustering
notebooks/04_modeling.ipynb         → Supervised Learning
notebooks/04b_semi_supervised.ipynb → Semi-supervised
notebooks/05_evaluation_report.ipynb → Tổng hợp kết quả
```

---

## Thiết kế thực nghiệm

| Hạng mục | Chi tiết |
|----------|----------|
| **Train/Test split** | 80/20, stratified (giữ tỉ lệ lớp), `random_seed=42` |
| **Xử lý mất cân bằng** | SMOTE chỉ trên tập Train (tránh data leakage) |
| **Metric phân lớp** | PR-AUC, F1-Score (minority class), Recall |
| **Metric hồi quy** | MAE, RMSE (đơn vị: phút) |
| **Metric clustering** | Silhouette Score, Davies-Bouldin Index (DBI) |
| **Semi-supervised** | Self-Training với ngưỡng tin cậy 0.85, giả lập 5%/10%/20% nhãn |
| **Time-series split** | Không xáo trộn (shuffle=False), UDI ≈ time index |

---

## Kết quả chính (tóm tắt)

Sau khi chạy pipeline, xem chi tiết tại:
- `outputs/reports/metrics_summary.md`
- `outputs/figures/` (biểu đồ Confusion Matrix, Feature Importance, Elbow, v.v.)

---

## Dependencies

```
pandas>=1.5.0
numpy>=1.23.0
scikit-learn>=1.2.0
xgboost>=1.7.0
imbalanced-learn>=0.10.0
mlxtend>=0.22.0
matplotlib>=3.6.0
seaborn>=0.12.0
pyyaml>=6.0
joblib>=1.2.0
```
