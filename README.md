# 🌬️ AQI Forecast Web Demo — Miền Trung Việt Nam

Web demo dự báo chất lượng không khí (AQI) cho 4 tỉnh miền Trung:  
**Thanh Hóa · Nghệ An · Hà Tĩnh · Huế**

---

## 📁 Cấu trúc thư mục

```
app.py                       ← File chính
requirements.txt
best_pca_models/             ← Copy từ ĐATN/05_models/best_pca_models/
    thanh_hoa_best_model.pkl
    thanh_hoa_scaler_pca.pkl
    thanh_hoa_pca.pkl
    thanh_hoa_strong_vars.pkl
    thanh_hoa_inference_info.pkl
    nghe_an_best_model.pkl
    ... (tương tự cho nghe_an, ha_tinh, hue)
```

---

## 🚀 Cài đặt & Chạy

```bash
# 1. Tạo môi trường ảo
python -m venv .venv
source .venv/bin/activate       # Linux/Mac
# hoặc
.venv\Scripts\activate          # Windows

# 2. Cài thư viện
pip install -r requirements.txt

# 3. Copy artifacts
cp -r /path/to/ĐATN/05_models/best_pca_models ./best_pca_models

# 4. Chạy app
streamlit run app.py
```

---

## 🗂️ Các Tab

### Tab 1 — Dự báo Realtime 📡
- Tự động lấy dữ liệu từ **Open-Meteo Air Quality API** + **Archive API**
- Xây dựng **81 features** (lags, rolling stats, thời gian, tương tác)
- Pipeline: `strong_vars filter → StandardScaler → PCA 95% → Best Model`
- Hiển thị **7 chân trời**: t+1h, t+3h, t+6h, t+12h, t+24h, t+48h, t+72h
- Gauge chart + bar chart màu theo mức AQI

### Tab 2 — Phân loại & Khuyến nghị 📊
- **Rule-based** theo 6 mức VN-AQI (US AQI scale)
- 3 chế độ: dùng AQI dự báo / nhập tay / xem tất cả mức
- Khuyến nghị: hoạt động phù hợp, khung giờ an toàn, nhóm nhạy cảm

### Tab 3 — Lịch sử 3 ngày 📅
- Biểu đồ AQI + PM2.5 theo thời gian
- Thống kê tóm tắt (TB, min, max, std, % giờ Tốt)
- Phân bố theo mức AQI (bảng + pie chart)
- AQI trung bình theo giờ trong ngày
- Tải CSV

---

## ⚙️ Cấu hình

Chỉnh đường dẫn artifacts trong `app.py` nếu cần:

```python
BEST_MODEL_DIR = os.path.join(os.path.dirname(__file__), "best_pca_models")
```

---

## 📊 Kết quả mô hình (PCA 95%)

| Tỉnh      | Model    | RMSE   | WLA    |
|-----------|----------|--------|--------|
| Thanh Hóa | CatBoost | 13.974 | 77.5%  |
| Nghệ An   | CatBoost | 10.465 | 83.3%  |
| Hà Tĩnh   | Lasso    | 10.524 | 82.9%  |
| Huế       | CatBoost | 9.380  | 88.6%  |

---

## 🔌 API sử dụng

| API | Endpoint | Mục đích |
|-----|----------|----------|
| Open-Meteo Air Quality | `air-quality-api.open-meteo.com/v1/air-quality` | AQI, PM2.5, PM10, ... |
| Open-Meteo Archive | `archive-api.open-meteo.com/v1/archive` | Thời tiết lịch sử |
| Open-Meteo Forecast | `api.open-meteo.com/v1/forecast` | Thời tiết dự báo |

Tất cả API đều **miễn phí**, không cần API key.
