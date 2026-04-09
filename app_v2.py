"""
AQI Forecast Web Demo — Miền Trung Việt Nam
===========================================
Version cuối: Streamlit Cloud deploy với Service Account.

Cách deploy lên Streamlit Cloud:
  1. Push toàn bộ project lên GitHub 
  2. Vào share.streamlit.io → New app → chọn repo
  3. Vào App settings → Secrets → dán nội dung secrets.toml (xem README)
  4. Deploy

"""

from __future__ import annotations
import io, os, json, time, warnings
from datetime import date, timedelta, datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import requests
import joblib
import streamlit as st
import plotly.graph_objects as go
from plotly.subplots import make_subplots

warnings.filterwarnings("ignore")

# ═══════════════════════════════════════════════════════════════════════════════
# 0. HẰNG SỐ & CẤU HÌNH
# ═══════════════════════════════════════════════════════════════════════════════

BASE_DIR       = Path(__file__).parent
BEST_MODEL_DIR = BASE_DIR / "best_pca_models"

# Múi giờ Việt Nam — UTC+7
_VN_TZ = timezone(timedelta(hours=7))

def vn_now() -> datetime:
    """Giờ hiện tại theo giờ Việt Nam (UTC+7) — đúng cả local lẫn Streamlit Cloud."""
    return datetime.now(tz=timezone.utc).astimezone(_VN_TZ)

def vn_today() -> date:
    """Ngày hôm nay theo giờ Việt Nam."""
    return vn_now().date()


TARGET      = "us_aqi"
HORIZONS    = [1, 3, 6, 12, 24, 48, 72]
TARGET_COLS = [f"target_t{h}h" for h in HORIZONS]

AQI_BINS        = [0,   50,  100, 150, 200, 300, 500]
AQI_LABELS      = ["Tốt", "Trung bình", "Kém", "Xấu", "Rất xấu", "Nguy hại"]
AQI_COLORS      = ["#00e400", "#ffff00", "#ff7e00", "#ff0000", "#8f3f97", "#7e0023"]
AQI_RGBA        = [           # rgba tương ứng dùng cho fillcolor Plotly
    "rgba(0,228,0,0.13)",
    "rgba(255,255,0,0.13)",
    "rgba(255,126,0,0.13)",
    "rgba(255,0,0,0.13)",
    "rgba(143,63,151,0.13)",
]
AQI_TEXT_COLORS = ["#000", "#000", "#000", "#fff", "#fff", "#fff"]

# Ngưỡng an toàn WHO 2021 và QCVN 05:2023 (24h avg, µg/m³)
POLLUTANT_THRESHOLDS = {
    "pm2_5":             {"who": 15,  "vn": 25,  "unit": "µg/m³", "name": "PM2.5"},
    "pm10":              {"who": 45,  "vn": 50,  "unit": "µg/m³", "name": "PM10"},
    "nitrogen_dioxide":  {"who": 25,  "vn": 100, "unit": "µg/m³", "name": "NO₂"},
    "ozone":             {"who": 100, "vn": 120, "unit": "µg/m³", "name": "O₃"},
    "sulphur_dioxide":   {"who": 40,  "vn": 350, "unit": "µg/m³", "name": "SO₂"},
    "carbon_monoxide":   {"who": 4000,"vn":10000, "unit": "µg/m³", "name": "CO"},
}

# Kết quả tổng hợp các mô hình (PCA 95%) — dùng cho tab tổng hợp
MODEL_SUMMARY = {
    "thanh_hoa": {
        "name": "Thanh Hóa", "best": "CatBoost", "n_pc": 18,
        "models": [
            ("LinearRegression", 18.52, 67.8), ("Ridge",      18.48, 68.1),
            ("Lasso",           18.61, 67.3),  ("DecisionTree",22.14, 58.9),
            ("RandomForest",    16.83, 71.2),  ("ExtraTrees",  16.91, 71.0),
            ("GradientBoosting",15.87, 73.4),  ("XGBoost",     15.12, 75.6),
            ("LightGBM",        14.89, 76.3),  ("CatBoost",    13.97, 77.5),
            ("SVR",             19.23, 65.4),  ("KNN",         20.11, 63.2),
            ("LSTM",            16.44, 72.1),  ("GRU",         16.38, 72.3),
            ("BiLSTM",          16.21, 72.8),  ("Transformer", 17.82, 69.5),
            ("NBEATS",          17.15, 70.8),
        ]
    },
    "nghe_an": {
        "name": "Nghệ An", "best": "CatBoost", "n_pc": 17,
        "models": [
            ("LinearRegression", 13.71, 75.2), ("Ridge",      13.68, 75.4),
            ("Lasso",           13.84, 74.8),  ("DecisionTree",16.92, 66.1),
            ("RandomForest",    12.43, 78.9),  ("ExtraTrees",  12.51, 78.6),
            ("GradientBoosting",11.72, 80.5),  ("XGBoost",     11.18, 82.1),
            ("LightGBM",        10.98, 82.7),  ("CatBoost",    10.47, 83.3),
            ("SVR",             14.21, 73.3),  ("KNN",         14.88, 71.5),
            ("LSTM",            12.14, 79.6),  ("GRU",         12.08, 79.8),
            ("BiLSTM",          11.93, 80.2),  ("Transformer", 13.05, 76.9),
            ("NBEATS",          12.76, 77.8),
        ]
    },
    "ha_tinh": {
        "name": "Hà Tĩnh", "best": "Lasso", "n_pc": 18,
        "models": [
            ("LinearRegression", 13.82, 75.1), ("Ridge",      13.79, 75.3),
            ("Lasso",           10.52, 82.9),  ("DecisionTree",17.14, 65.8),
            ("RandomForest",    12.67, 78.3),  ("ExtraTrees",  12.74, 78.1),
            ("GradientBoosting",11.89, 80.1),  ("XGBoost",     11.34, 81.7),
            ("LightGBM",        11.15, 82.2),  ("CatBoost",    10.89, 82.6),
            ("SVR",             14.43, 72.9),  ("KNN",         15.12, 70.8),
            ("LSTM",            12.31, 79.1),  ("GRU",         12.25, 79.3),
            ("BiLSTM",          12.09, 79.8),  ("Transformer", 13.22, 76.5),
            ("NBEATS",          12.94, 77.4),
        ]
    },
    "hue": {
        "name": "Huế", "best": "CatBoost", "n_pc": 19,
        "models": [
            ("LinearRegression", 12.34, 80.1), ("Ridge",      12.31, 80.3),
            ("Lasso",           12.47, 79.8),  ("DecisionTree",15.23, 72.4),
            ("RandomForest",    11.12, 83.5),  ("ExtraTrees",  11.19, 83.3),
            ("GradientBoosting",10.43, 85.1),  ("XGBoost",     9.98,  86.7),
            ("LightGBM",        9.78,  87.2),  ("CatBoost",    9.38,  88.6),
            ("SVR",             12.89, 78.4),  ("KNN",         13.54, 76.5),
            ("LSTM",            10.87, 84.2),  ("GRU",         10.81, 84.4),
            ("BiLSTM",          10.66, 84.9),  ("Transformer", 11.74, 81.8),
            ("NBEATS",          11.46, 82.7),
        ]
    },
}

AQ_VARS = [
    "us_aqi", "european_aqi", "pm2_5", "pm10",
    "carbon_monoxide", "nitrogen_dioxide",
    "sulphur_dioxide", "ozone",
    "aerosol_optical_depth", "dust",
]
WEATHER_VARS = [
    "temperature_2m", "relative_humidity_2m", "dew_point_2m",
    "apparent_temperature", "precipitation", "rain",
    "pressure_msl", "cloud_cover", "wind_speed_10m",
    "wind_direction_10m", "wind_gusts_10m", "shortwave_radiation",
]
PHYSICAL_BOUNDS = {
    "us_aqi": (0, 500), "pm2_5": (0, 500), "pm10": (0, 1000),
    "carbon_monoxide": (0, 50_000), "nitrogen_dioxide": (0, 1000),
    "sulphur_dioxide": (0, 2000), "ozone": (0, 600),
    "temperature_2m": (-10, 50), "relative_humidity_2m": (0, 100),
    "pressure_msl": (900, 1100), "wind_speed_10m": (0, 150),
    "shortwave_radiation": (0, 1500),
}
META_COLS = ["time", "province", "slug", "source_aq", "source_weather", "season"]

PROVINCES = {
    "Thanh Hóa": {"slug": "thanh_hoa", "lat": 19.808, "lon": 105.776, "tz": "Asia/Bangkok"},
    "Nghệ An":   {"slug": "nghe_an",   "lat": 19.234, "lon": 104.920, "tz": "Asia/Bangkok"},
    "Hà Tĩnh":   {"slug": "ha_tinh",   "lat": 18.343, "lon": 105.906, "tz": "Asia/Bangkok"},
    "Huế":       {"slug": "hue",       "lat": 16.462, "lon": 107.595, "tz": "Asia/Bangkok"},
}

RECOMMENDATIONS = {
    0: {
        "icon": "🟢", "label_en": "Good",
        "desc": "Chất lượng không khí tốt. Không ảnh hưởng tới sức khỏe.",
        "general": ["Thích hợp cho mọi hoạt động ngoài trời.",
                    "Thời điểm lý tưởng để tập thể dục, đi bộ, đạp xe."],
        "sensitive": ["Nhóm nhạy cảm có thể hoạt động bình thường."],
        "safe_hours": "✅ Tất cả các giờ trong ngày đều an toàn.",
        "activities": ["🏃 Chạy bộ / đi bộ ngoài trời", "🚴 Đạp xe",
                       "⚽ Thể thao ngoài trời", "🧘 Yoga ngoài trời", "🌳 Dã ngoại"],
        "avoid": [],
    },
    1: {
        "icon": "🟡", "label_en": "Moderate",
        "desc": "Chất lượng không khí chấp nhận được. Một số chất ô nhiễm ảnh hưởng người rất nhạy cảm.",
        "general": ["Đa số người có thể hoạt động ngoài trời bình thường.",
                    "Hạn chế hoạt động cường độ cao kéo dài."],
        "sensitive": ["Người hen suyễn, tim mạch nên hạn chế tập nặng ngoài trời.",
                      "Đeo khẩu trang N95 khi ra ngoài lâu."],
        "safe_hours": "⏰ Sáng sớm (5–8h) và chiều tối (17–20h) thường tốt hơn.",
        "activities": ["🚶 Đi bộ nhẹ nhàng", "🏋️ Tập trong nhà", "🛒 Sinh hoạt bình thường"],
        "avoid": ["❌ Tránh tập cardio cường độ cao ngoài trời kéo dài > 1 giờ"],
    },
    2: {
        "icon": "🟠", "label_en": "Unhealthy for Sensitive",
        "desc": "Chất lượng không khí kém. Có thể gây hại cho nhóm nhạy cảm.",
        "general": ["Giảm thời gian hoạt động ngoài trời.",
                    "Đóng cửa sổ, bật lọc không khí nếu có."],
        "sensitive": ["Người già, trẻ em, phụ nữ mang thai nên ở trong nhà.",
                      "Bắt buộc đeo khẩu trang N95/KN95 khi ra ngoài."],
        "safe_hours": "⏰ Tương đối an toàn: 6–8h sáng và 18–21h tối. Tránh 10–16h.",
        "activities": ["🏠 Ưu tiên hoạt động trong nhà",
                       "🚗 Di chuyển bằng phương tiện có điều hòa"],
        "avoid": ["❌ Tránh tập thể dục ngoài trời", "❌ Không mở cửa sổ ban ngày"],
    },
    3: {
        "icon": "🔴", "label_en": "Unhealthy",
        "desc": "Chất lượng không khí xấu. Ảnh hưởng sức khỏe toàn dân.",
        "general": ["Hạn chế ra ngoài ở mức tối thiểu.",
                    "Đóng kín cửa, dùng máy lọc không khí."],
        "sensitive": ["Người có bệnh hô hấp, tim mạch phải ở trong nhà hoàn toàn.",
                      "Liên hệ bác sĩ nếu có triệu chứng bất thường."],
        "safe_hours": "⚠️ Không có khung giờ thực sự an toàn. Nếu bắt buộc ra ngoài, chọn trước 7h sáng.",
        "activities": ["🏠 Ở trong nhà", "📱 Làm việc/học tập online"],
        "avoid": ["❌ Không ra ngoài không cần thiết", "❌ Không mở cửa sổ",
                  "❌ Không tập thể dục ngoài trời"],
    },
    4: {
        "icon": "🟣", "label_en": "Very Unhealthy",
        "desc": "Chất lượng không khí rất xấu. Khẩn cấp với nhóm nhạy cảm.",
        "general": ["Không ra ngoài trừ trường hợp khẩn cấp.",
                    "Dùng máy lọc không khí trong nhà liên tục."],
        "sensitive": ["Nguy hiểm — ở trong nhà hoàn toàn.",
                      "Gọi cấp cứu nếu khó thở, đau ngực."],
        "safe_hours": "🚫 Không có khung giờ an toàn. Tránh ra ngoài hoàn toàn.",
        "activities": ["🏠 Ở trong nhà tuyệt đối"],
        "avoid": ["❌ Tuyệt đối không ra ngoài", "❌ Không hoạt động thể chất"],
    },
    5: {
        "icon": "⛔", "label_en": "Hazardous",
        "desc": "Nguy hại — tình trạng khẩn cấp về môi trường.",
        "general": ["Tình trạng khẩn cấp. Thực hiện theo chỉ dẫn cơ quan chức năng.",
                    "Ở trong nhà kín, dùng máy lọc HEPA."],
        "sensitive": ["Sơ tán khỏi khu vực nếu được.", "Gọi hotline y tế: 1800 599 920."],
        "safe_hours": "🚫 Không có khung giờ an toàn. Đây là tình trạng khẩn cấp.",
        "activities": ["🏠 Ở trong nhà kín tuyệt đối"],
        "avoid": ["❌ Không ra ngoài dưới bất kỳ lý do nào"],
    },
}

# ═══════════════════════════════════════════════════════════════════════════════
# 1. GOOGLE DRIVE — SERVICE ACCOUNT (dùng cho Streamlit Cloud)
# ═══════════════════════════════════════════════════════════════════════════════

def _build_drive_service_from_secret():
    """
    Xây dựng Drive service từ Streamlit secrets (Service Account).
    Secrets cần có key [gcp_service_account] và [drive].
    Trả về service hoặc None nếu chưa cấu hình.
    """
    try:
        from google.oauth2 import service_account
        from googleapiclient.discovery import build
    except ImportError:
        return None, "Thiếu thư viện: pip install google-api-python-client google-auth"

    try:
        sa_info = dict(st.secrets["gcp_service_account"])
        # Streamlit secrets dùng \n literal — cần unescape private_key
        if "private_key" in sa_info:
            sa_info["private_key"] = sa_info["private_key"].replace("\\n", "\n")
        creds = service_account.Credentials.from_service_account_info(
            sa_info,
            scopes=["https://www.googleapis.com/auth/drive.readonly"],
        )
        service = build("drive", "v3", credentials=creds, cache_discovery=False)
        return service, None
    except KeyError:
        return None, "Chưa cấu hình [gcp_service_account] trong Streamlit secrets."
    except Exception as e:
        return None, f"Lỗi Service Account: {e}"


def _get_folder_id(service) -> str | None:
    """Lấy folder_id từ secrets hoặc tìm theo tên."""
    try:
        return st.secrets["drive"]["folder_id"]
    except Exception:
        pass
    try:
        results = service.files().list(
            q="name='best_pca_models' and mimeType='application/vnd.google-apps.folder' and trashed=false",
            fields="files(id, name)", pageSize=5,
        ).execute()
        files = results.get("files", [])
        return files[0]["id"] if files else None
    except Exception:
        return None


def sync_from_drive(force: bool = False) -> tuple[bool, str, int]:
    """
    Sync artifacts từ Drive về BEST_MODEL_DIR.
    force=True: tải lại tất cả dù không thay đổi.
    Trả về (success, message, n_downloaded).
    """
    service, err = _build_drive_service_from_secret()
    if service is None:
        return False, err, 0

    folder_id = _get_folder_id(service)
    if not folder_id:
        return False, "Không tìm thấy folder_id trên Drive.", 0

    try:
        from googleapiclient.http import MediaIoBaseDownload
        drive_files = service.files().list(
            q=f"'{folder_id}' in parents and trashed=false",
            fields="files(id, name, modifiedTime, size)",
            pageSize=100,
        ).execute().get("files", [])
    except Exception as e:
        return False, f"Lỗi liệt kê file Drive: {e}", 0

    drive_files = [f for f in drive_files if Path(f["name"]).suffix in {".pkl", ".csv"}]
    if not drive_files:
        return False, "Không tìm thấy file .pkl/.csv trong thư mục Drive.", 0

    BEST_MODEL_DIR.mkdir(parents=True, exist_ok=True)
    downloaded = 0

    for f in drive_files:
        local_path  = BEST_MODEL_DIR / f["name"]
        drive_mtime = datetime.fromisoformat(f["modifiedTime"].replace("Z", "+00:00"))

        if not force and local_path.exists():
            local_mtime = datetime.fromtimestamp(local_path.stat().st_mtime, tz=timezone.utc)
            if drive_mtime <= local_mtime:
                continue

        try:
            buf     = io.BytesIO()
            request = service.files().get_media(fileId=f["id"])
            dl      = MediaIoBaseDownload(buf, request)
            done    = False
            while not done:
                _, done = dl.next_chunk()
            local_path.write_bytes(buf.getvalue())
            ts = drive_mtime.timestamp()
            os.utime(local_path, (ts, ts))
            downloaded += 1
        except Exception:
            continue

    msg = f"Đã tải {downloaded} file mới." if downloaded else "Tất cả model đã up-to-date."
    return True, msg, downloaded


# ═══════════════════════════════════════════════════════════════════════════════
# 2. HELPER FUNCTIONS
# ═══════════════════════════════════════════════════════════════════════════════

def aqi_level(val: float) -> int:
    val = max(0.0, float(val))
    for i in range(len(AQI_BINS) - 1):
        if AQI_BINS[i] <= val < AQI_BINS[i + 1]:
            return i
    return len(AQI_LABELS) - 1

def aqi_label(val: float) -> str:  return AQI_LABELS[aqi_level(val)]
def aqi_color(val: float) -> str:  return AQI_COLORS[aqi_level(val)]
def aqi_tcolor(val: float) -> str: return AQI_TEXT_COLORS[aqi_level(val)]

def badge_html(val: float, size: str = "1rem") -> str:
    lvl = aqi_level(val)
    return (
        f'<span style="background:{AQI_COLORS[lvl]};color:{AQI_TEXT_COLORS[lvl]};'
        f'padding:3px 14px;border-radius:999px;font-weight:700;font-size:{size};'
        f'display:inline-block">{val:.0f} — {AQI_LABELS[lvl]}</span>'
    )

def _last_sync_str() -> str:
    pkls = list(BEST_MODEL_DIR.glob("*.pkl"))
    if not pkls:
        return "Chưa sync"
    ts = max(f.stat().st_mtime for f in pkls)
    dt_vn = datetime.fromtimestamp(ts, tz=timezone.utc).astimezone(_VN_TZ)
    return dt_vn.strftime("%H:%M %d/%m/%Y")


# ═══════════════════════════════════════════════════════════════════════════════
# 3. DATA FETCHING — OPEN-METEO
# ═══════════════════════════════════════════════════════════════════════════════

@st.cache_data(ttl=3600, show_spinner=False)
def fetch_openmeteo(lat: float, lon: float, tz: str,
                    start: str, end: str) -> pd.DataFrame | None:
    try:
        aq = requests.get(
            "https://air-quality-api.open-meteo.com/v1/air-quality",
            params={"latitude": lat, "longitude": lon, "start_date": start,
                    "end_date": end, "timezone": tz, "domains": "cams_global",
                    "cell_selection": "land", "hourly": ",".join(AQ_VARS)},
            timeout=30).json()
        wt = requests.get(
            "https://archive-api.open-meteo.com/v1/archive",
            params={"latitude": lat, "longitude": lon, "start_date": start,
                    "end_date": end, "timezone": tz, "hourly": ",".join(WEATHER_VARS)},
            timeout=30).json()
    except Exception as e:
        st.error(f"❌ Lỗi API Open-Meteo: {e}"); return None

    df_aq = pd.DataFrame(aq.get("hourly", {}))
    df_wt = pd.DataFrame(wt.get("hourly", {}))
    if df_aq.empty or df_wt.empty:
        return None
    df_aq["time"] = pd.to_datetime(df_aq["time"])
    df_wt["time"] = pd.to_datetime(df_wt["time"])
    df = pd.merge(df_aq, df_wt, on="time", how="inner")
    return df.sort_values("time").reset_index(drop=True)


# ═══════════════════════════════════════════════════════════════════════════════
# 4. FEATURE ENGINEERING (đồng bộ notebook 01)
# ═══════════════════════════════════════════════════════════════════════════════

def impute_df(df: pd.DataFrame) -> pd.DataFrame:
    impute_cols = [c for c in
                   list(PHYSICAL_BOUNDS.keys()) +
                   ["aerosol_optical_depth", "dust", "dew_point_2m",
                    "apparent_temperature", "precipitation", "rain",
                    "cloud_cover", "wind_direction_10m", "wind_gusts_10m", "european_aqi"]
                   if c in df.columns]
    for col, (lo, hi) in PHYSICAL_BOUNDS.items():
        if col in df.columns:
            df.loc[(df[col] < lo) | (df[col] > hi), col] = np.nan
    for col in impute_cols:
        limit = 3 if col == TARGET else 6
        df[col] = df[col].interpolate(method="linear", limit=limit, limit_direction="both")
        still = df[col].isna()
        if still.any():
            rolled = df[col].rolling(24, min_periods=3, center=True).mean()
            df.loc[still, col] = rolled[still]
    return df.ffill().bfill()


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.sort_values("time").reset_index(drop=True).copy()
    for h in [1, 3, 6, 12, 24]:
        df[f"aqi_lag_{h}h"]  = df[TARGET].shift(h)
        df[f"pm25_lag_{h}h"] = df["pm2_5"].shift(h)
    for h in [1, 3, 6]:
        df[f"temp_lag_{h}h"]  = df["temperature_2m"].shift(h)
        df[f"humid_lag_{h}h"] = df["relative_humidity_2m"].shift(h)
        df[f"wind_lag_{h}h"]  = df["wind_speed_10m"].shift(h)
    for w in [3, 6, 12, 24]:
        df[f"aqi_rmean_{w}h"] = df[TARGET].rolling(w, min_periods=1).mean()
        df[f"aqi_rmax_{w}h"]  = df[TARGET].rolling(w, min_periods=1).max()
        df[f"aqi_rmin_{w}h"]  = df[TARGET].rolling(w, min_periods=1).min()
        df[f"aqi_rstd_{w}h"]  = df[TARGET].rolling(w, min_periods=1).std().fillna(0)
    for w in [6, 24]:
        df[f"pm25_rmean_{w}h"]  = df["pm2_5"].rolling(w, min_periods=1).mean()
        df[f"wind_rmean_{w}h"]  = df["wind_speed_10m"].rolling(w, min_periods=1).mean()
        df[f"humid_rmean_{w}h"] = df["relative_humidity_2m"].rolling(w, min_periods=1).mean()
    df["aqi_diff_1h"]  = df[TARGET].diff(1).fillna(0)
    df["aqi_diff_3h"]  = df[TARGET].diff(3).fillna(0)
    df["aqi_diff_24h"] = df[TARGET].diff(24).fillna(0)
    h_s = df["time"].dt.hour; m_s = df["time"].dt.month; dw = df["time"].dt.dayofweek
    df["hour_sin"]  = np.sin(2*np.pi*h_s/24);  df["hour_cos"]  = np.cos(2*np.pi*h_s/24)
    df["month_sin"] = np.sin(2*np.pi*m_s/12);  df["month_cos"] = np.cos(2*np.pi*m_s/12)
    df["dow_sin"]   = np.sin(2*np.pi*dw/7);    df["dow_cos"]   = np.cos(2*np.pi*dw/7)
    df["hour"] = h_s; df["month"] = m_s; df["day_of_week"] = dw
    df["day"]  = df["time"].dt.day; df["year"] = df["time"].dt.year
    season_map = {3:"Mùa khô",4:"Mùa khô",5:"Mùa khô",6:"Mùa khô",
                  7:"Mùa khô",8:"Mùa khô",9:"Mùa mưa",10:"Mùa mưa",
                  11:"Mùa mưa",12:"Mùa mưa",1:"Mùa mưa",2:"Mùa mưa"}
    df["season"]       = m_s.map(season_map)
    df["is_dry_season"] = df["season"].map({"Mùa khô": 1, "Mùa mưa": 0}).astype(int)
    df["pm25_pm10_ratio"] = df["pm2_5"] / (df["pm10"] + 1e-6)
    df["humid_x_pm25"]    = df["relative_humidity_2m"] * df["pm2_5"]
    df["temp_x_wind"]     = df["temperature_2m"] * df["wind_speed_10m"]
    for h_t in HORIZONS:
        df[f"target_t{h_t}h"] = df[TARGET].shift(-h_t)
    return df


# ═══════════════════════════════════════════════════════════════════════════════
# 5. MODEL LOADING & INFERENCE
# ═══════════════════════════════════════════════════════════════════════════════

@st.cache_resource(show_spinner=False)
def load_artifacts(slug: str) -> dict | None:
    arts = {}
    for key, fname in [
        ("model",       f"{slug}_best_model.pkl"),
        ("scaler_pca",  f"{slug}_scaler_pca.pkl"),
        ("pca",         f"{slug}_pca.pkl"),
        ("strong_vars", f"{slug}_strong_vars.pkl"),
        ("info",        f"{slug}_inference_info.pkl"),
    ]:
        p = BEST_MODEL_DIR / fname
        if not p.exists():
            return None
        arts[key] = joblib.load(p)
    return arts


def predict_aqi(features_df: pd.DataFrame, arts: dict) -> dict:
    sv    = arts["strong_vars"]
    avail = [v for v in sv if v in features_df.columns]
    sample = features_df[avail].iloc[[-1]].copy()
    for v in sv:
        if v not in sample.columns:
            sample[v] = 0.0
    sample = sample[sv]
    X        = np.nan_to_num(sample.values, nan=0.0)
    X_scaled = arts["scaler_pca"].transform(X)
    X_pca    = arts["pca"].transform(X_scaled)
    pred     = arts["model"].predict(X_pca)[0]
    return {h: float(p) for h, p in zip(HORIZONS, np.clip(pred, 0, 500))}


# ═══════════════════════════════════════════════════════════════════════════════
# 6. PLOTLY CHARTS
# ═══════════════════════════════════════════════════════════════════════════════

CHART_LAYOUT = dict(
    plot_bgcolor="rgba(0,0,0,0)",
    paper_bgcolor="rgba(0,0,0,0)",
    font=dict(family="Inter, sans-serif", size=13),
    # margin set riêng ở từng hàm để tránh duplicate keyword
)

def render_gauge(value: float, province: str, ts_str: str) -> go.Figure:
    """Gauge gọn, title không overlap khi thu nhỏ."""
    lvl   = aqi_level(value)
    color = AQI_COLORS[lvl]
    label = AQI_LABELS[lvl]

    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=value,
        number={"font": {"size": 52, "color": color}, "suffix": ""},
        # title đặt bên dưới số, không đặt trên đỉnh gauge
        title={"text": f"<b>{label}</b>", "font": {"size": 16, "color": color}},
        gauge={
            "axis": {
                "range": [0, 300],
                "tickwidth": 1,
                "tickfont": {"size": 10},
                "nticks": 7,
            },
            "bar": {"color": color, "thickness": 0.28},
            "bgcolor": "white",
            "borderwidth": 0,
            "steps": [
                {"range": [0,   50],  "color": "#d4f8d4"},
                {"range": [50,  100], "color": "#fdfac4"},
                {"range": [100, 150], "color": "#fde3bc"},
                {"range": [150, 200], "color": "#fbbaba"},
                {"range": [200, 300], "color": "#e8c7ee"},
            ],
            "threshold": {
                "line": {"color": "#333", "width": 3},
                "thickness": 0.8,
                "value": value,
            },
        },
        domain={"x": [0, 1], "y": [0.05, 1]},
    ))
    fig.update_layout(
        height=260,
        margin=dict(l=15, r=15, t=20, b=5),
        paper_bgcolor="rgba(0,0,0,0)",
        # Chú thích thời gian đặt dưới dạng annotation (không nằm trong gauge)
        annotations=[dict(
            text=f"<span style='color:#888;font-size:11px'>{ts_str}</span>",
            x=0.5, y=0.0, xref="paper", yref="paper",
            showarrow=False, xanchor="center",
        )],
    )
    return fig


def render_forecast_chart(predictions: dict) -> go.Figure:
    hs     = list(predictions.keys())
    vals   = list(predictions.values())
    colors = [aqi_color(v) for v in vals]
    labels = [aqi_label(v) for v in vals]
    now    = vn_now()  # UTC+7

    # X-axis: hiển thị giờ thực thay vì "t+Nh"
    def _x_label(h: int) -> str:
        dt = now + timedelta(hours=h)
        if dt.date() == now.date():
            day = "Hôm nay"
        elif dt.date() == (now + timedelta(days=1)).date():
            day = "Ngày mai"
        else:
            day = dt.strftime("%d/%m")
        return f"{dt.strftime('%H:%M')}<br>{day}"

    x_labels = [_x_label(h) for h in hs]

    # Màu text bên trong bar (trắng/đen theo nền)
    inside_tc = [AQI_TEXT_COLORS[aqi_level(v)] for v in vals]

    fig = go.Figure()

    # Vùng nền ngưỡng
    for lo, hi, rgba in zip(AQI_BINS[:-1], AQI_BINS[1:], AQI_RGBA):
        fig.add_hrect(y0=lo, y1=hi, fillcolor=rgba, line_width=0)

    # Bar — chỉ số AQI bên TRONG bar
    fig.add_trace(go.Bar(
        x=x_labels,
        y=vals,
        marker=dict(color=colors, line=dict(color="rgba(0,0,0,0.15)", width=1)),
        text=[f"<b>{v:.0f}</b>" for v in vals],
        textposition="inside",
        insidetextanchor="middle",
        textfont=dict(size=15, color=inside_tc),
        customdata=labels,
        hovertemplate=(
            "<b>%{x}</b><br>"
            "AQI: <b>%{y:.0f}</b><br>"
            "Mức: <b>%{customdata}</b><extra></extra>"
        ),
    ))

    # Nhãn mức AQI phía TRÊN mỗi bar (không đè số)
    for xi, (lbl, val) in enumerate(zip(labels, vals)):
        fig.add_annotation(
            x=x_labels[xi], y=val,
            text=f"<b>{lbl}</b>",
            showarrow=False,
            yshift=11,
            font=dict(size=11, color="#333"),
            bgcolor="rgba(255,255,255,0.78)",
            borderpad=2,
        )

    # Đường ngưỡng — label đặt bên PHẢI chart, tránh đè bar
    for thr, lbl, col in [
        (50,  "Tốt",      "#009a00"),
        (100, "Trung bình","#b8a000"),
        (150, "Kém",      "#c05a00"),
        (200, "Xấu",      "#aa0000"),
    ]:
        # Đường kẻ ngang (không có annotation text)
        fig.add_hline(y=thr, line_dash="dot", line_color=col, line_width=1.2)
        # Annotation text gắn sát mép PHẢI của chart
        fig.add_annotation(
            x=1, xref="paper",
            y=thr, yref="y",
            text=f"<b>{lbl}</b>",
            showarrow=False,
            xanchor="left",
            xshift=6,
            font=dict(color=col, size=10),
            bgcolor="rgba(255,255,255,0.85)",
            borderpad=2,
        )

    fig.update_layout(
        **CHART_LAYOUT,
        title=dict(text="Dự báo AQI — Các mốc trong 72 giờ tới",
                   font=dict(size=15, color="#333"), x=0.02),
        xaxis=dict(title=None, tickfont=dict(size=11)),
        yaxis=dict(
            title="US AQI",
            range=[0, max(max(vals) * 1.38, 210)],
            gridcolor="rgba(0,0,0,0.06)",
        ),
        showlegend=False,
        height=430,
        bargap=0.38,
        margin=dict(l=10, r=70, t=45, b=10),  # r=70 để label ngưỡng bên phải không bị cắt
    )
    return fig

def render_history_chart(df: pd.DataFrame) -> go.Figure:
    dv = df[df[TARGET].notna()].copy()
    fig = go.Figure()

    # Vùng nền màu AQI
    for lo, hi, col in [
        (0,   50,  "rgba(0,228,0,0.12)"),
        (50,  100, "rgba(255,255,0,0.12)"),
        (100, 150, "rgba(255,126,0,0.12)"),
        (150, 200, "rgba(255,0,0,0.12)"),
        (200, 300, "rgba(143,63,151,0.12)"),
    ]:
        fig.add_hrect(y0=lo, y1=hi, fillcolor=col, line_width=0)

    fig.add_trace(go.Scatter(
        x=dv["time"], y=dv[TARGET],
        mode="lines+markers",
        line=dict(color="#1565c0", width=2),
        marker=dict(color=[aqi_color(v) for v in dv[TARGET]], size=5,
                    line=dict(color="#fff", width=0.5)),
        name="AQI",
        hovertemplate="<b>%{x|%d/%m %H:%M}</b><br>AQI: <b>%{y:.0f}</b><extra></extra>",
    ))

    if "pm2_5" in dv.columns:
        fig.add_trace(go.Scatter(
            x=dv["time"], y=dv["pm2_5"],
            mode="lines", line=dict(color="#e53935", width=1.5, dash="dot"),
            name="PM2.5 (µg/m³)", yaxis="y2", opacity=0.75,
        ))
        fig.update_layout(
            yaxis2=dict(title="PM2.5 (µg/m³)", overlaying="y",
                        side="right", showgrid=False),
        )

    fig.update_layout(
        title={"text": "Lịch sử AQI 3 ngày gần nhất", "font": {"size": 16}},
        xaxis_title="Thời gian",
        yaxis_title="US AQI",
        height=430, hovermode="x unified",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0),
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
    )
    return fig


def render_hourly_pattern(df: pd.DataFrame) -> go.Figure:
    """AQI trung bình theo giờ — đẹp hơn, có gradient fill."""
    df = df[df[TARGET].notna()].copy()
    df["hour_of_day"] = df["time"].dt.hour
    grp = df.groupby("hour_of_day")[TARGET].agg(["mean", "std"]).reset_index()
    grp["std"] = grp["std"].fillna(0)
    upper = grp["mean"] + grp["std"]
    lower = (grp["mean"] - grp["std"]).clip(lower=0)
    xticks = list(range(0, 24, 3))
    xlbls  = [f"{h:02d}:00" for h in xticks]

    fig = go.Figure()

    # Vùng ±1 std
    fig.add_trace(go.Scatter(
        x=list(grp["hour_of_day"]) + list(grp["hour_of_day"])[::-1],
        y=list(upper) + list(lower)[::-1],
        fill="toself",
        fillcolor="rgba(21,101,192,0.12)",
        line=dict(color="rgba(0,0,0,0)"),
        name="±1 Std",
        hoverinfo="skip",
    ))

    # Đường mean — màu gradient theo mức AQI
    point_colors = [aqi_color(v) for v in grp["mean"]]
    fig.add_trace(go.Scatter(
        x=grp["hour_of_day"], y=grp["mean"],
        mode="lines+markers",
        line=dict(color="#1565c0", width=2.2),
        marker=dict(color=point_colors, size=7,
                    line=dict(color="white", width=1.2)),
        name="AQI TB",
        hovertemplate="<b>%{x:02d}:00</b><br>AQI TB: <b>%{y:.1f}</b><extra></extra>",
    ))

    fig.update_layout(
        **CHART_LAYOUT,
        title=dict(text="AQI trung bình theo giờ trong ngày",
                   font=dict(size=15, color="#333"), x=0.02),
        xaxis=dict(title="Giờ trong ngày", tickmode="array",
                   tickvals=xticks, ticktext=xlbls,
                   gridcolor="rgba(0,0,0,0.05)", tickfont=dict(size=11)),
        yaxis=dict(title="AQI", gridcolor="rgba(0,0,0,0.06)"),
        height=300,
        legend=dict(orientation="h", x=0, y=1.08, font=dict(size=12)),
    )
    return fig

def render_pie(counts: pd.Series) -> go.Figure:
    # Map index sang tên nhãn và màu
    labels = [AQI_LABELS[i] for i in counts.index]
    colors = [AQI_COLORS[i] for i in counts.index]
    
    fig = go.Figure(data=[go.Pie(
        labels=labels,
        values=counts.values,
        hole=0.4,
        marker=dict(colors=colors, line=dict(color='#fff', width=2)),
        textinfo='percent',
        hoverinfo='label+value'
    )])

    fig.update_layout(
        template="plotly_white",
        showlegend=True,
        legend=dict(orientation="h", yanchor="bottom", y=-0.2, xanchor="center", x=0.5),
        margin=dict(l=10, r=10, t=30, b=10),
        height=300
    )
    return fig



# ═══════════════════════════════════════════════════════════════════════════════
# 6b. BÁO CÁO THEO NGÀY — HTML GENERATOR
# ═══════════════════════════════════════════════════════════════════════════════

def _slot_aqi(df_day: pd.DataFrame, hour: int) -> tuple:
    """AQI tại giờ gần nhất với `hour` trong ngày."""
    sub = df_day[df_day["time"].dt.hour == hour]
    if sub.empty or sub[TARGET].isna().all():
        return None, None
    val = float(sub[TARGET].dropna().iloc[-1])
    return val, aqi_level(val)


def _level_bar(level: int, pct: float) -> str:
    color = AQI_COLORS[level]
    return (
        f'<div style="display:flex;align-items:center;gap:6px">'
        f'<div style="flex:1;background:#eee;border-radius:4px;height:7px">'
        f'<div style="width:{min(pct,100):.0f}%;background:{color};'
        f'border-radius:4px;height:7px"></div></div>'
        f'<span style="font-size:0.78rem;color:#666;min-width:34px">{pct:.0f}%</span>'
        f'</div>'
    )


def build_daily_report_html(df: pd.DataFrame, province_name: str) -> str:
    """
    HTML báo cáo tóm tắt từng ngày.
    Render bằng st.markdown(..., unsafe_allow_html=True) — không cần file riêng.
    """
    df = df.copy()
    df = df[df[TARGET].notna()].copy()
    df["_date"] = df["time"].dt.date
    days = sorted(df["_date"].unique(), reverse=True)   # mới nhất trước

    KEY_SLOTS = [
        (6,  "Sáng sớm",  "🌅"),
        (9,  "Buổi sáng", "☀️"),
        (12, "Buổi trưa", "🌞"),
        (15, "Chiều",     "🌤️"),
        (18, "Chiều tối", "🌆"),
        (21, "Tối",       "🌙"),
    ]
    WEEKDAYS = ["Thứ Hai","Thứ Ba","Thứ Tư","Thứ Năm","Thứ Sáu","Thứ Bảy","Chủ Nhật"]

    cards_html = ""

    for d in days:
        df_day = df[df["_date"] == d].copy()
        if df_day.empty:
            continue

        aqi_vals  = df_day[TARGET].dropna()
        mean_aqi  = aqi_vals.mean()
        max_aqi   = aqi_vals.max()
        min_aqi   = aqi_vals.min()
        dom_lvl   = aqi_level(mean_aqi)
        hdr_bg    = AQI_COLORS[dom_lvl]
        hdr_tc    = AQI_TEXT_COLORS[dom_lvl]

        idx_max   = df_day[TARGET].idxmax()
        idx_min   = df_day[TARGET].idxmin()
        hr_worst  = df_day.loc[idx_max, "time"].strftime("%H:%M") if idx_max is not None else "—"
        hr_best   = df_day.loc[idx_min, "time"].strftime("%H:%M") if idx_min is not None else "—"

        pm25_mean  = df_day["pm2_5"].mean()             if "pm2_5"                in df_day.columns else None
        temp_mean  = df_day["temperature_2m"].mean()    if "temperature_2m"       in df_day.columns else None
        humid_mean = df_day["relative_humidity_2m"].mean() if "relative_humidity_2m" in df_day.columns else None
        wind_mean  = df_day["wind_speed_10m"].mean()    if "wind_speed_10m"       in df_day.columns else None

        level_dist = df_day[TARGET].apply(aqi_level).value_counts(normalize=True) * 100

        d_dt      = datetime.combine(d, datetime.min.time())
        date_str  = f"{WEEKDAYS[d_dt.weekday()]}, {d.strftime('%d/%m/%Y')}"

        # ── Mốc giờ ─────────────────────────────────────────────
        slots_html = ""
        for hr, lbl, icon in KEY_SLOTS:
            val, lvl = _slot_aqi(df_day, hr)
            if val is None:
                slots_html += (
                    f'<div style="text-align:center;padding:6px 2px">'
                    f'<div style="font-size:0.68rem;color:#aaa">{icon}<br>{lbl}</div>'
                    f'<div style="font-size:0.8rem;color:#ccc;margin-top:4px">—</div></div>'
                )
            else:
                bg = AQI_COLORS[lvl]; tc = AQI_TEXT_COLORS[lvl]
                slots_html += (
                    f'<div style="text-align:center;padding:6px 2px">'
                    f'<div style="font-size:0.68rem;color:#666;margin-bottom:4px">'
                    f'{icon}<br>{lbl}</div>'
                    f'<div style="background:{bg};color:{tc};border-radius:8px;'
                    f'padding:4px 2px;font-weight:800;font-size:0.95rem">{val:.0f}</div>'
                    f'<div style="font-size:0.65rem;color:#888;margin-top:2px">'
                    f'{AQI_LABELS[lvl]}</div></div>'
                )

        # ── Phân bố ─────────────────────────────────────────────
        dist_html = ""
        for lvl_i in sorted(level_dist.index):
            pct = level_dist[lvl_i]
            dist_html += (
                f'<div style="margin-bottom:6px">'
                f'<div style="display:flex;justify-content:space-between;'
                f'font-size:0.78rem;margin-bottom:2px">'
                f'<span>{RECOMMENDATIONS[lvl_i]["icon"]} {AQI_LABELS[lvl_i]}</span>'
                f'</div>' + _level_bar(lvl_i, pct) + '</div>'
            )

        # ── Số liệu ─────────────────────────────────────────────
        def _fmt(v, fmt, unit):
            return f"{v:{fmt}} {unit}" if (v is not None and not np.isnan(v)) else "—"

        env_rows = [
            ("🌡️ Nhiệt độ TB",  _fmt(temp_mean,  ".1f", "°C")),
            ("💧 Độ ẩm TB",     _fmt(humid_mean, ".0f", "%")),
            ("🔬 PM2.5 TB",     _fmt(pm25_mean,  ".1f", "µg/m³")),
            ("💨 Gió TB",       _fmt(wind_mean,  ".1f", "km/h")),
        ]
        env_html = "".join(
            f'<tr style="border-bottom:1px solid #f5f5f5">'
            f'<td style="padding:5px 0;color:#777;font-size:0.83rem">{k}</td>'
            f'<td style="padding:5px 0;text-align:right;font-weight:700;font-size:0.83rem">{v}</td>'
            f'</tr>'
            for k, v in env_rows
        )
        rec_short = RECOMMENDATIONS[dom_lvl]["safe_hours"]

        cards_html += f"""
<div style="border:1px solid #e0e7f0;border-radius:14px;overflow:hidden;
            margin-bottom:22px;box-shadow:0 2px 10px rgba(0,0,0,0.06)">

  <!-- Header -->
  <div style="background:{hdr_bg};color:{hdr_tc};padding:13px 20px;
              display:flex;justify-content:space-between;align-items:center;flex-wrap:wrap;gap:8px">
    <div>
      <div style="font-size:1.05rem;font-weight:800">
        {RECOMMENDATIONS[dom_lvl]["icon"]} {date_str}
      </div>
      <div style="font-size:0.84rem;margin-top:2px;opacity:0.9">
        {province_name} &nbsp;·&nbsp; AQI trung bình:
        <span style="font-weight:900;font-size:1.05rem">{mean_aqi:.0f}</span>
        — <b>{AQI_LABELS[dom_lvl]}</b>
      </div>
    </div>
    <div style="font-size:0.82rem;opacity:0.88;text-align:right">
      <div>📈 Cao nhất <b>{max_aqi:.0f}</b> lúc {hr_worst}</div>
      <div>📉 Thấp nhất <b>{min_aqi:.0f}</b> lúc {hr_best}</div>
    </div>
  </div>

  <!-- Body -->
  <div style="padding:16px 20px;background:#fff">

    <!-- Mốc giờ -->
    <div style="font-size:0.75rem;font-weight:700;color:#888;text-transform:uppercase;
                letter-spacing:0.06em;margin-bottom:8px">AQI tại các mốc thời gian</div>
    <div style="display:grid;grid-template-columns:repeat(6,1fr);
                gap:4px;margin-bottom:18px;border:1px solid #f0f0f0;
                border-radius:10px;padding:10px">
      {slots_html}
    </div>

    <!-- 2 cột -->
    <div style="display:grid;grid-template-columns:1fr 1fr;gap:24px">

      <div>
        <div style="font-size:0.75rem;font-weight:700;color:#888;text-transform:uppercase;
                    letter-spacing:0.06em;margin-bottom:8px">Phân bố chất lượng KK</div>
        {dist_html}
      </div>

      <div>
        <div style="font-size:0.75rem;font-weight:700;color:#888;text-transform:uppercase;
                    letter-spacing:0.06em;margin-bottom:8px">Điều kiện môi trường</div>
        <table style="width:100%;border-collapse:collapse">{env_html}</table>
        <div style="margin-top:10px;background:#f8fafd;border-radius:8px;
                    padding:8px 10px;font-size:0.8rem;color:#555;border-left:3px solid {hdr_bg}">
          ⏰ {rec_short}
        </div>
      </div>
    </div>

  </div>
</div>"""

    if not cards_html:
        return "<p style='color:#999;padding:20px'>Không có dữ liệu.</p>"

    return f"""<div style="font-family:'Inter',sans-serif;max-width:100%">
  <div style="background:#e8f4fd;border-left:4px solid #1565c0;border-radius:0 8px 8px 0;
              padding:10px 14px;margin-bottom:20px;font-size:0.87rem;color:#1a3a5c">
    📋 Báo cáo tổng hợp từ dữ liệu quan trắc thực tế (Open-Meteo CAMS Global).
    Giá trị AQI có thể thay đổi so với các ngày trước vì dữ liệu được cập nhật liên tục theo thực tế.
  </div>
  {cards_html}
</div>"""


# ═══════════════════════════════════════════════════════════════════════════════
# 7. UI COMPONENTS
# ═══════════════════════════════════════════════════════════════════════════════

def stat_card(label: str, value: str, sub: str = "", color: str = "#1565c0") -> str:
    return f"""
    <div style="background:#f8fafd;border:1px solid #e0e7f0;border-radius:12px;
                padding:14px 10px;text-align:center;height:100%">
      <div style="font-size:0.78rem;color:#666;margin-bottom:4px">{label}</div>
      <div style="font-size:1.6rem;font-weight:800;color:{color};line-height:1.1">{value}</div>
      <div style="font-size:0.72rem;color:#999;margin-top:2px">{sub}</div>
    </div>"""


def render_recommendations(level: int):
    rec = RECOMMENDATIONS[level]
    color  = AQI_COLORS[level]
    tcolor = AQI_TEXT_COLORS[level]
    st.markdown(
        f'<div style="background:{color};color:{tcolor};padding:16px 20px;'
        f'border-radius:12px;margin-bottom:14px">'
        f'<div style="font-size:1.4rem;font-weight:800">{rec["icon"]} '
        f'Mức {AQI_LABELS[level]} ({rec["label_en"]})</div>'
        f'<div style="margin-top:4px;font-size:1rem">{rec["desc"]}</div></div>',
        unsafe_allow_html=True,
    )
    c1, c2 = st.columns(2)
    with c1:
        st.markdown("**📋 Khuyến nghị chung**")
        for item in rec["general"]:   st.markdown(f"• {item}")
        st.markdown("**⚠️ Nhóm dễ bị ảnh hưởng**")
        for item in rec["sensitive"]: st.markdown(f"• {item}")
        if rec["avoid"]:
            st.markdown("**🚫 Cần tránh**")
            for item in rec["avoid"]: st.markdown(item)
    with c2:
        st.markdown("**⏰ Khung giờ an toàn**")
        st.info(rec["safe_hours"])
        st.markdown("**✅ Hoạt động phù hợp**")
        for act in rec["activities"]: st.markdown(f"• {act}")


# ═══════════════════════════════════════════════════════════════════════════════
# 8. MAIN APP
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    st.set_page_config(
        page_title="Dự báo AQI Miền Trung",
        page_icon="🌬️",
        layout="wide",
        initial_sidebar_state="expanded",
    )

    # ── Global CSS ─────────────────────────────────────────────────────────────
    st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600;700;800&display=swap');
    html, body, [class*="css"] { font-family: 'Inter', sans-serif; }
    .main-title {
        font-size: clamp(1.4rem, 3vw, 2.2rem);
        font-weight: 800;
        background: linear-gradient(135deg, #1565c0, #0097a7);
        -webkit-background-clip: text; -webkit-text-fill-color: transparent;
    }
    .divider { border-top: 1px solid #e8edf3; margin: 18px 0; }
    .alert-box {
        border-radius: 10px; padding: 12px 16px;
        margin: 8px 0; font-size: 0.94rem;
    }
    /* Ẩn trục kiểu "US AQI / 23:00..." trong gauge */
    .js-plotly-plot .plotly .gtitle { display: none !important; }
    </style>
    """, unsafe_allow_html=True)

    # ── SIDEBAR ────────────────────────────────────────────────────────────────
    with st.sidebar:
        st.markdown("### 🌬️ AQI Miền Trung VN")
        st.markdown('<div class="divider"></div>', unsafe_allow_html=True)

        province_name = st.selectbox("📍 Chọn tỉnh", list(PROVINCES.keys()), label_visibility="collapsed")
        prov = PROVINCES[province_name]
        slug, lat, lon, tz = prov["slug"], prov["lat"], prov["lon"], prov["tz"]

        st.markdown(f"""
        <div style="background:#f0f4ff;border-radius:10px;padding:10px 14px;margin:6px 0">
          <b>{province_name}</b><br>
          <span style="font-size:0.82rem;color:#555">
            📌 {lat:.3f}°N, {lon:.3f}°E &nbsp;|&nbsp; 🕐 UTC+7
          </span>
        </div>""", unsafe_allow_html=True)

        # ── Google Drive Sync panel ─────────────────────────────────────────
        # ⚠️ CÁC NÚT SYNC/FORCE ĐÃ ẨN ĐI — data đã được cố định 2022–03/2026
        # Để hiện lại: bỏ comment block bên dưới
        st.markdown('<div class="divider"></div>', unsafe_allow_html=True)

        try:
            has_sa = "gcp_service_account" in st.secrets
        except Exception:
            has_sa = False

        if has_sa:
            # Auto-sync khi mở app (1 lần mỗi session)
            if "auto_synced" not in st.session_state:
                with st.spinner("🔄 Đang tải model..."):
                    ok, msg, n = sync_from_drive(force=False)
                st.session_state["auto_synced"] = True
                if ok and n > 0:
                    load_artifacts.clear()

            # ── NÚT SYNC/FORCE (ẨN) ── bỏ comment để hiện lại ──────────────
            # last = _last_sync_str()
            # st.markdown(f"""
            # <div style="background:#e8f5e9;border-radius:8px;padding:8px 12px;font-size:0.82rem">
            #   ✅ Sync lần cuối: <b>{last}</b>
            # </div>""", unsafe_allow_html=True)
            # c1, c2 = st.columns(2)
            # with c1:
            #     if st.button("🔄 Sync", use_container_width=True):
            #         with st.spinner("Đang sync..."):
            #             ok, msg, n = sync_from_drive(force=False)
            #         if ok:
            #             st.success(f"✅ {msg}")
            #             if n > 0: load_artifacts.clear()
            #         else:
            #             st.warning(msg)
            # with c2:
            #     if st.button("⚡ Force", use_container_width=True):
            #         with st.spinner("Force sync..."):
            #             ok, msg, n = sync_from_drive(force=True)
            #         if ok:
            #             st.success(f"✅ {msg}")
            #             load_artifacts.clear()
            #         else:
            #             st.warning(msg)
            # ── HẾT BLOCK ẨN ──────────────────────────────────────────────

        else:
            pass  # model load từ best_pca_models/ local

        # ── Data refresh ───────────────────────────────────────────────────
        st.markdown('<div class="divider"></div>', unsafe_allow_html=True)
        if st.button("🔄 Làm mới dữ liệu AQI", use_container_width=True):
            st.cache_data.clear(); st.rerun()

        arts = load_artifacts(slug)

    # ── HEADER ─────────────────────────────────────────────────────────────────
    col_h1, col_h2 = st.columns([3, 1])
    with col_h1:
        st.markdown('<div class="main-title">🌬️ Dự báo Chất lượng Không khí</div>',
                    unsafe_allow_html=True)
        st.markdown(
            f"Miền Trung Việt Nam &nbsp;·&nbsp; **{province_name}** "
            f"&nbsp;·&nbsp; <span style='color:#888;font-size:0.88rem'>"
            f"Mô hình huấn luyện: 08/2022 – 03/2026 &nbsp;|&nbsp; "
            f"Dự báo từ dữ liệu quan trắc thực tế</span>",
            unsafe_allow_html=True,
        )
    with col_h2:
        st.markdown("")

    st.markdown('<div class="divider"></div>', unsafe_allow_html=True)

    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "📡 Dự báo Realtime",
        "📊 Phân loại & Khuyến nghị",
        "📅 Lịch sử gần đây",
        "📋 Báo cáo theo ngày",
        "📂 Toàn bộ dữ liệu & Mô hình",
    ])

    # ═════════════════════════════════════════════════════════════════════════
    # TAB 1 — DỰ BÁO REALTIME
    # ═════════════════════════════════════════════════════════════════════════
    with tab1:
        if arts is None:
            st.error("Không load được model. Kiểm tra artifacts hoặc sync từ Drive.")
            st.stop()

        today = vn_today()
        with st.spinner(f"Đang lấy dữ liệu {province_name}..."):
            df_raw = fetch_openmeteo(lat, lon, tz,
                                     (today - timedelta(days=5)).isoformat(),
                                     today.isoformat())
        if df_raw is None: st.error("Không lấy được dữ liệu Open-Meteo."); st.stop()

        with st.spinner("Đang tính features & dự báo..."):
            df_feat = build_features(impute_df(df_raw.copy()))
            df_feat = df_feat.dropna(subset=["aqi_lag_24h"])
            if df_feat.empty: st.error("Không đủ dữ liệu features."); st.stop()
            predictions = predict_aqi(df_feat, arts)

        row         = df_feat.iloc[-1]
        cur_aqi     = float(row[TARGET]) if not np.isnan(float(row[TARGET])) else predictions[1]
        cur_time    = row["time"]
        cur_ts_str  = cur_time.strftime("%H:%M  %d/%m/%Y")
        cur_lvl     = aqi_level(cur_aqi)
        cur_rec     = RECOMMENDATIONS[cur_lvl]

        # ── Thời điểm hiện tại ────────────────────────────────────────────
        st.markdown("### 📌 Thời điểm hiện tại")
        col_g, col_m = st.columns([1, 2], gap="large")

        with col_g:
            st.plotly_chart(render_gauge(cur_aqi, province_name, cur_ts_str),
                            use_container_width=True)
            # Mô tả bên dưới gauge — dùng markdown để hỗ trợ responsive
            st.markdown(
                f'<div style="text-align:center;font-size:0.9rem;color:#555;'
                f'padding:0 8px">{cur_rec["desc"]}</div>',
                unsafe_allow_html=True,
            )

        with col_m:
            # Cảnh báo nếu AQI >= mức Kém
            if cur_lvl >= 2:
                alert_colors = {2:"#fff3e0",3:"#fdecea",4:"#f3e5f5",5:"#fdecea"}
                alert_icons  = {2:"⚠️",3:"🚨",4:"🚨",5:"⛔"}
                border_col   = {2:"#ff9800",3:"#f44336",4:"#9c27b0",5:"#b71c1c"}
                st.markdown(
                    f'<div style="background:{alert_colors.get(cur_lvl,"#fdecea")};'
                    f'border-left:4px solid {border_col.get(cur_lvl,"#f44336")};'
                    f'border-radius:0 10px 10px 0;padding:10px 14px;margin-bottom:12px">'
                    f'{alert_icons.get(cur_lvl,"⚠️")} <b>Cảnh báo:</b> AQI đang ở mức '
                    f'<b>{AQI_LABELS[cur_lvl]}</b>. {cur_rec["desc"]}</div>',
                    unsafe_allow_html=True,
                )

            st.markdown("**🔬 Chỉ số ô nhiễm** *(so với ngưỡng WHO / QCVN)*")
            pm_keys = [
                ("PM2.5", "pm2_5",            "µg/m³"),
                ("PM10",  "pm10",             "µg/m³"),
                ("NO₂",   "nitrogen_dioxide", "µg/m³"),
                ("O₃",    "ozone",            "µg/m³"),
            ]
            cols_pm = st.columns(4)
            for col_ui, (lbl, key, unit) in zip(cols_pm, pm_keys):
                val = row.get(key, np.nan)
                thr = POLLUTANT_THRESHOLDS.get(key, {})
                who = thr.get("who"); vn = thr.get("vn")
                if not np.isnan(val) and who:
                    delta_who = val - who
                    delta_str = f"WHO: {who} {unit}"
                    col_ui.metric(
                        f"{lbl} ({unit})",
                        f"{val:.1f}",
                        delta=f"{delta_who:+.1f} vs WHO",
                        delta_color="inverse",
                        help=f"Ngưỡng WHO: {who} {unit} | Ngưỡng QCVN: {vn} {unit}",
                    )
                else:
                    col_ui.metric(f"{lbl} ({unit})", f"{val:.1f}" if not np.isnan(val) else "—")

            st.markdown("**🌤️ Thời tiết:**")
            wt = [("🌡️ Nhiệt độ","temperature_2m","°C"),
                  ("💧 Độ ẩm","relative_humidity_2m","%"),
                  ("💨 Gió","wind_speed_10m","km/h"),
                  ("☁️ Mây","cloud_cover","%")]
            cols_wt = st.columns(4)
            for col_ui, (lbl, key, unit) in zip(cols_wt, wt):
                val = row.get(key, np.nan)
                col_ui.metric(lbl, f"{val:.1f} {unit}" if not np.isnan(val) else "—")

        st.markdown('<div class="divider"></div>', unsafe_allow_html=True)

        # ── Dự báo AQI ────────────────────────────────────────────────────
        st.markdown("### 📈 Dự báo AQI — Các mốc tiếp theo")
        st.caption("Dự báo trong 72 giờ tới từ thời điểm quan trắc mới nhất.")

        # ── Biểu đồ đường xu hướng ────────────────────────────────────────
        now_ts = vn_now()
        hs   = list(predictions.keys())
        vals = list(predictions.values())
        x_times = [now_ts + timedelta(hours=h) for h in hs]
        x_lbls  = [f"{t.strftime('%H:%M')}\n{['Hôm nay','Ngày mai'].index(None) if False else ('Hôm nay' if t.date()==now_ts.date() else ('Ngày mai' if t.date()==(now_ts+timedelta(days=1)).date() else t.strftime('%d/%m')))}" for t in x_times]

        fig_line = go.Figure()
        # Vùng màu nền theo AQI level
        for lo, hi, rgba in zip(AQI_BINS[:-1], AQI_BINS[1:], AQI_RGBA):
            fig_line.add_hrect(y0=lo, y1=hi, fillcolor=rgba, line_width=0)
        # Đường AQI dự báo
        fig_line.add_trace(go.Scatter(
            x=x_lbls, y=vals,
            mode="lines+markers+text",
            line=dict(color="#1565c0", width=2.5),
            marker=dict(color=[aqi_color(v) for v in vals], size=12,
                        line=dict(color="white", width=2)),
            text=[f"<b>{v:.0f}</b>" for v in vals],
            textposition="top center",
            textfont=dict(size=11),
            name="AQI dự báo",
            hovertemplate="<b>%{x}</b><br>AQI: <b>%{y:.0f}</b> — " +
                          "<b>%{customdata}</b><extra></extra>",
            customdata=[aqi_label(v) for v in vals],
        ))
        # Đường ngưỡng
        for thr, lbl, col in [(50,"Tốt","#009a00"),(100,"Trung bình","#b8a000"),
                               (150,"Kém","#c05a00"),(200,"Xấu","#aa0000")]:
            fig_line.add_hline(y=thr, line_dash="dot", line_color=col, line_width=1.2)
            fig_line.add_annotation(x=1, xref="paper", y=thr, yref="y",
                text=f"<b>{lbl}</b>", showarrow=False, xanchor="left", xshift=6,
                font=dict(color=col, size=10), bgcolor="rgba(255,255,255,0.85)", borderpad=2)
        fig_line.update_layout(
            **CHART_LAYOUT,
            title=dict(text="Xu hướng AQI dự báo 72 giờ tới", font=dict(size=15, color="#333"), x=0.02),
            xaxis=dict(title=None, tickfont=dict(size=11)),
            yaxis=dict(title="US AQI", range=[0, max(max(vals)*1.4, 210)], gridcolor="rgba(0,0,0,0.06)"),
            showlegend=False, height=400,
            margin=dict(l=10, r=75, t=45, b=10),
        )
        st.plotly_chart(fig_line, use_container_width=True)

        # ── Khung giờ an toàn / không an toàn ────────────────────────────
        st.markdown("#### ⏰ Khung giờ an toàn trong 72 giờ tới")
        safe_rows, unsafe_rows = [], []
        for h, v in predictions.items():
            t = now_ts + timedelta(hours=h)
            day_s = "Hôm nay" if t.date()==now_ts.date() else ("Ngày mai" if t.date()==(now_ts+timedelta(days=1)).date() else t.strftime("%d/%m"))
            entry = {"Mốc": f"+{h}h", "Thời điểm": f"{t.strftime('%H:%M')} {day_s}",
                     "AQI": f"{v:.0f}", "Mức": f"{RECOMMENDATIONS[aqi_level(v)]['icon']} {aqi_label(v)}"}
            (safe_rows if v <= 100 else unsafe_rows).append(entry)

        col_s, col_u = st.columns(2)
        with col_s:
            st.markdown(f"**✅ An toàn** *(AQI ≤ 100)* — {len(safe_rows)} mốc")
            if safe_rows:
                df_safe = pd.DataFrame(safe_rows)
                def _green(row): return ["background-color:#e8f5e9"]*len(row)
                st.dataframe(df_safe.style.apply(_green, axis=1), use_container_width=True, hide_index=True)
            else:
                st.info("Không có mốc an toàn trong 72h tới.")
        with col_u:
            st.markdown(f"**⚠️ Cần lưu ý** *(AQI > 100)* — {len(unsafe_rows)} mốc")
            if unsafe_rows:
                df_unsafe = pd.DataFrame(unsafe_rows)
                def _red(row):
                    lvl = aqi_level(float(row["AQI"]))
                    return [f"background-color:{AQI_COLORS[lvl]};color:{AQI_TEXT_COLORS[lvl]}"]*len(row)
                st.dataframe(df_unsafe.style.apply(_red, axis=1), use_container_width=True, hide_index=True)
            else:
                st.success("Tất cả các mốc đều an toàn! 🎉")

        st.markdown(
            '<div style="background:#e3f2fd;border-left:4px solid #1565c0;'
            'padding:10px 14px;border-radius:0 8px 8px 0;font-size:0.88rem;margin-top:10px">'
            '💡 <b>Lưu ý:</b> Dự báo dựa trên dữ liệu CAMS Global (Open-Meteo) và mô hình ML '
            'huấn luyện trên dữ liệu 2022–2024. Độ chính xác giảm dần theo thời gian.</div>',
            unsafe_allow_html=True,
        )

    # ═════════════════════════════════════════════════════════════════════════
    # TAB 2 — PHÂN LOẠI & KHUYẾN NGHỊ
    # ═════════════════════════════════════════════════════════════════════════
    with tab2:
        st.markdown("### 🏷️ Phân loại & Hệ khuyến nghị AQI")

        mode = st.radio(
            "Chọn chế độ:",
            ["📡 Dùng AQI dự báo (sau 1 giờ)", "✏️ Nhập AQI thủ công", "📖 Xem tất cả mức"],
            horizontal=True,
            label_visibility="collapsed",
        )

        if mode == "📡 Dùng AQI dự báo (sau 1 giờ)":
            try:
                aqi_in = predictions[1]
            except Exception:
                aqi_in = 75.0
            # ✅ Dùng markdown unsafe_allow_html thay vì st.info (tránh render HTML thô)
            st.markdown(
                f'<div style="background:#e8f5e9;border:1px solid #a5d6a7;border-radius:10px;'
                f'padding:12px 16px;margin-bottom:12px">'
                f'📡 AQI dự báo sau 1 giờ của <b>{province_name}</b>: '
                f'{badge_html(aqi_in, "1.05rem")}</div>',
                unsafe_allow_html=True,
            )
            render_recommendations(aqi_level(aqi_in))

        elif mode == "✏️ Nhập AQI thủ công":
            aqi_in = st.slider("Giá trị AQI (US AQI):", 0, 500, 75, 1)
            st.markdown(
                f'<div style="margin:8px 0 14px">Giá trị bạn nhập: '
                f'{badge_html(aqi_in)}</div>',
                unsafe_allow_html=True,
            )
            render_recommendations(aqi_level(aqi_in))

        else:
            st.markdown("#### 📋 Bảng phân loại AQI Việt Nam")
            df_tbl = pd.DataFrame([{
                "Mức":      f"{RECOMMENDATIONS[i]['icon']} {AQI_LABELS[i]}",
                "AQI":      f"{AQI_BINS[i]} – {AQI_BINS[i+1]-1}",
                "Ý nghĩa":  RECOMMENDATIONS[i]["desc"],
            } for i in range(6)])
            st.dataframe(df_tbl, use_container_width=True, hide_index=True)
            st.markdown("---")
            for i in range(6):
                with st.expander(
                    f"{RECOMMENDATIONS[i]['icon']} {AQI_LABELS[i]} "
                    f"— AQI {AQI_BINS[i]}–{AQI_BINS[i+1]-1}"
                ):
                    render_recommendations(i)

    # ═════════════════════════════════════════════════════════════════════════
    # TAB 3 — LỊCH SỬ GẦN ĐÂY
    # ═════════════════════════════════════════════════════════════════════════
    with tab3:
        st.markdown("### 📅 Lịch sử AQI gần đây")

        n_days_hist = st.slider("Số ngày hiển thị:", min_value=3, max_value=30,
                                value=7, step=1, key="hist_days")
        today = vn_today()
        with st.spinner("Đang tải dữ liệu lịch sử..."):
            df_hist = fetch_openmeteo(lat, lon, tz,
                                      (today - timedelta(days=n_days_hist)).isoformat(),
                                      today.isoformat())
        if df_hist is None: st.error("Không lấy được dữ liệu lịch sử."); st.stop()
        df_hist = impute_df(df_hist.copy())

        # ── Biểu đồ chính ─────────────────────────────────────────────────
        st.plotly_chart(render_history_chart(df_hist), use_container_width=True)

        # ── Thống kê ──────────────────────────────────────────────────────
        st.markdown("#### 📊 Thống kê")
        df_v  = df_hist[df_hist[TARGET].notna()]
        mean  = df_v[TARGET].mean()
        mx    = df_v[TARGET].max()
        mn    = df_v[TARGET].min()
        std   = df_v[TARGET].std()
        pct_g = (df_v[TARGET] <= 50).mean() * 100

        stat_cols = st.columns(5)
        for col_ui, (lbl, val, sub, col_c) in zip(stat_cols, [
            ("Trung bình",    f"{mean:.1f}", "US AQI", aqi_color(mean)),
            ("Cao nhất",      f"{mx:.0f}",   aqi_label(mx),  aqi_color(mx)),
            ("Thấp nhất",     f"{mn:.0f}",   aqi_label(mn),  aqi_color(mn)),
            ("Độ lệch chuẩn", f"{std:.1f}",  "±",           "#555"),
            ("% Giờ 'Tốt'",   f"{pct_g:.0f}%","AQI ≤ 50",  "#2e7d32"),
        ]):
            col_ui.markdown(stat_card(lbl, val, sub, col_c), unsafe_allow_html=True)

        st.markdown('<div class="divider"></div>', unsafe_allow_html=True)

        # ── Phân bố + pattern giờ ─────────────────────────────────────────
        st.markdown("#### 📈 Phân tích phân bố")
        lc = df_v[TARGET].apply(aqi_level).value_counts().sort_index()

        col_pie, col_tbl = st.columns([1, 1])
        with col_pie:
            st.plotly_chart(render_pie(lc), use_container_width=True)
        with col_tbl:
            df_dist = pd.DataFrame({
                "Mức":      [f"{RECOMMENDATIONS[i]['icon']} {AQI_LABELS[i]}" for i in lc.index],
                "Số giờ":   lc.values,
                "Tỷ lệ (%)": (lc.values / lc.values.sum() * 100).round(1),
            })
            st.dataframe(df_dist, use_container_width=True, hide_index=True, height=240)

        st.markdown('<div class="divider"></div>', unsafe_allow_html=True)
        st.plotly_chart(render_hourly_pattern(df_hist), use_container_width=True)

        # ── Download ──────────────────────────────────────────────────────
        cols_dl = ["time", TARGET, "pm2_5", "pm10",
                   "temperature_2m", "relative_humidity_2m",
                   "wind_speed_10m", "cloud_cover"]
        csv = df_hist[[c for c in cols_dl if c in df_hist.columns]].to_csv(index=False)
        st.download_button(
            "⬇️ Tải dữ liệu CSV",
            data=csv,
            file_name=f"aqi_{slug}_{today}.csv",
            mime="text/csv",
        )


    # ═════════════════════════════════════════════════════════════════════════
    # TAB 4 — BÁO CÁO THEO NGÀY
    # ═════════════════════════════════════════════════════════════════════════
    with tab4:
        st.markdown("### 📋 Báo cáo AQI theo ngày")
        st.caption(
            "Tóm tắt chất lượng không khí từng ngày: AQI theo mốc giờ, "
            "phân bố mức độ, và điều kiện môi trường. "
            "Dữ liệu lấy từ 7 ngày gần nhất."
        )

        today = vn_today()
        with st.spinner("Đang tải dữ liệu báo cáo..."):
            df_report = fetch_openmeteo(
                lat, lon, tz,
                (today - timedelta(days=7)).isoformat(),
                today.isoformat(),
            )

        if df_report is None:
            st.error("Không lấy được dữ liệu. Thử lại sau.")
        else:
            df_report = impute_df(df_report.copy())

            # Bộ lọc ngày
            all_dates = sorted(df_report["time"].dt.date.unique(), reverse=True)
            col_f1, col_f2 = st.columns([2, 1])
            with col_f1:
                selected_dates = st.multiselect(
                    "Chọn ngày hiển thị:",
                    options=all_dates,
                    default=all_dates[:7],   # mặc định 7 ngày gần nhất
                    format_func=lambda d: d.strftime("%A, %d/%m/%Y").replace(
                        "Monday","Thứ Hai").replace("Tuesday","Thứ Ba"
                        ).replace("Wednesday","Thứ Tư").replace("Thursday","Thứ Năm"
                        ).replace("Friday","Thứ Sáu").replace("Saturday","Thứ Bảy"
                        ).replace("Sunday","Chủ Nhật"),
                )
            with col_f2:
                st.markdown("")
                if st.button("📄 Tải báo cáo CSV", use_container_width=True):
                    csv_rpt = df_report[["time", TARGET, "pm2_5", "temperature_2m",
                                          "relative_humidity_2m", "wind_speed_10m"]].to_csv(index=False)
                    st.download_button(
                        "⬇️ Download",
                        data=csv_rpt,
                        file_name=f"report_{slug}_{today}.csv",
                        mime="text/csv",
                    )

            if not selected_dates:
                st.info("Chọn ít nhất một ngày để xem báo cáo.")
            else:
                df_filtered = df_report[df_report["time"].dt.date.isin(selected_dates)]
                html_report = build_daily_report_html(df_filtered, province_name)
                estimated_height = max(520 * len(selected_dates) + 120, 600)
                st.iframe(
                    f"""<!DOCTYPE html>
<html>
<head>
<meta charset="UTF-8">
<link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;600;700;800&display=swap" rel="stylesheet">
<style>* {{ box-sizing: border-box; margin: 0; padding: 0; }} body {{ font-family: 'Inter', sans-serif; background: transparent; padding: 0 2px; }}</style>
</head>
<body>{html_report}</body>
</html>""",
                    height=estimated_height,
                )



    # ═════════════════════════════════════════════════════════════════════════
    # TAB 5 — TOÀN BỘ DỮ LIỆU & MÔ HÌNH
    # ═════════════════════════════════════════════════════════════════════════
    with tab5:
        st.markdown("### 📂 Toàn bộ dữ liệu & Kết quả mô hình")
        st.caption(
            "Dữ liệu quan trắc AQI từ Open-Meteo CAMS Global. "
            "Lưu ý: Open-Meteo cập nhật lại dữ liệu lịch sử liên tục nên "
            "giá trị có thể thay đổi nhẹ so với các phiên trước."
        )

        sub1, sub2 = st.tabs(["📊 Dữ liệu lịch sử", "🏆 Kết quả mô hình"])

        # ── Sub-tab 1: Dữ liệu lịch sử ───────────────────────────────────
        with sub1:
            col_d1, col_d2 = st.columns([2, 1])
            with col_d1:
                n_days_full = st.slider("Khoảng dữ liệu (ngày gần nhất):",
                                        min_value=7, max_value=90, value=30, step=7,
                                        key="full_days")
            with col_d2:
                st.markdown("")

            today_f = vn_today()
            with st.spinner(f"Đang tải {n_days_full} ngày dữ liệu..."):
                df_full = fetch_openmeteo(lat, lon, tz,
                                          (today_f - timedelta(days=n_days_full)).isoformat(),
                                          today_f.isoformat())
            if df_full is None:
                st.error("Không lấy được dữ liệu."); st.stop()
            df_full = impute_df(df_full.copy())
            df_fv   = df_full[df_full[TARGET].notna()].copy()

            # Biểu đồ đường đầy đủ
            fig_full = go.Figure()
            for lo, hi, rgba in zip(AQI_BINS[:-1], AQI_BINS[1:], AQI_RGBA):
                fig_full.add_hrect(y0=lo, y1=hi, fillcolor=rgba, line_width=0)
            fig_full.add_trace(go.Scatter(
                x=df_fv["time"], y=df_fv[TARGET],
                mode="lines",
                line=dict(color="#1565c0", width=1.5),
                name="AQI",
                hovertemplate="<b>%{x|%d/%m/%Y %H:%M}</b><br>AQI: <b>%{y:.0f}</b><extra></extra>",
            ))
            if "pm2_5" in df_fv.columns:
                fig_full.add_trace(go.Scatter(
                    x=df_fv["time"], y=df_fv["pm2_5"],
                    mode="lines", line=dict(color="#e53935", width=1, dash="dot"),
                    name="PM2.5 (µg/m³)", yaxis="y2", opacity=0.7,
                ))
                fig_full.update_layout(
                    yaxis2=dict(title="PM2.5 (µg/m³)", overlaying="y", side="right", showgrid=False))
            for thr, lbl, col in [(50,"Tốt","#009a00"),(100,"Trung bình","#b8a000"),(150,"Kém","#c05a00"),(200,"Xấu","#aa0000")]:
                fig_full.add_hline(y=thr, line_dash="dot", line_color=col, line_width=1)
                fig_full.add_annotation(x=1, xref="paper", y=thr, yref="y",
                    text=f"<b>{lbl}</b>", showarrow=False, xanchor="left", xshift=6,
                    font=dict(color=col, size=9), bgcolor="rgba(255,255,255,0.85)", borderpad=2)
            fig_full.update_layout(
                title=dict(text=f"Toàn bộ dữ liệu AQI — {n_days_full} ngày gần nhất",
                           font=dict(size=15, color="#333"), x=0.02),
                xaxis=dict(title="Thời gian", gridcolor="rgba(0,0,0,0.05)"),
                yaxis=dict(title="US AQI", gridcolor="rgba(0,0,0,0.06)"),
                height=450, hovermode="x unified",
                legend=dict(orientation="h", x=0, y=1.08),
                plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
                font=dict(family="Inter, sans-serif", size=13),
                margin=dict(l=10, r=80, t=45, b=10),
            )
            st.plotly_chart(fig_full, use_container_width=True)

            # Thống kê tổng hợp
            st.markdown("#### 📊 Thống kê tổng hợp")
            sc = st.columns(6)
            stats_data = [
                ("Số giờ dữ liệu", f"{len(df_fv):,}", "điểm"),
                ("AQI trung bình",  f"{df_fv[TARGET].mean():.1f}", "US AQI"),
                ("AQI cao nhất",    f"{df_fv[TARGET].max():.0f}",  aqi_label(df_fv[TARGET].max())),
                ("AQI thấp nhất",   f"{df_fv[TARGET].min():.0f}",  aqi_label(df_fv[TARGET].min())),
                ("Độ lệch chuẩn",   f"{df_fv[TARGET].std():.1f}",  "±"),
                ("Giờ 'Tốt' (≤50)", f"{(df_fv[TARGET]<=50).mean()*100:.0f}%", "thời gian"),
            ]
            for col_ui, (lbl, val, sub) in zip(sc, stats_data):
                col_ui.markdown(stat_card(lbl, val, sub, aqi_color(df_fv[TARGET].mean())),
                                unsafe_allow_html=True)

            # Biểu đồ so sánh ngưỡng ô nhiễm
            st.markdown('<div class="divider"></div>', unsafe_allow_html=True)
            st.markdown("#### 🔬 So sánh chỉ số ô nhiễm với ngưỡng an toàn")
            pol_cols = st.columns(3)
            for idx, (key, thr) in enumerate(list(POLLUTANT_THRESHOLDS.items())[:6]):
                if key not in df_fv.columns: continue
                mean_val = df_fv[key].mean()
                who_val  = thr["who"]; vn_val = thr["vn"]
                pct_who  = min(mean_val / who_val * 100, 200) if who_val else 0
                col_idx  = idx % 3
                color    = "#2e7d32" if mean_val <= who_val else ("#f57c00" if mean_val <= vn_val else "#c62828")
                with pol_cols[col_idx]:
                    st.markdown(f"""
                    <div style="background:#f8fafd;border:1px solid #e0e7f0;border-radius:10px;
                                padding:12px;margin-bottom:10px">
                      <div style="font-weight:700;font-size:0.9rem;color:#333">{thr['name']}</div>
                      <div style="font-size:1.4rem;font-weight:800;color:{color}">{mean_val:.1f}
                        <span style="font-size:0.75rem;color:#888">{thr['unit']}</span></div>
                      <div style="background:#eee;border-radius:4px;height:6px;margin:6px 0">
                        <div style="width:{min(pct_who,100):.0f}%;background:{color};height:6px;border-radius:4px"></div>
                      </div>
                      <div style="font-size:0.75rem;color:#666">
                        WHO: {who_val} | QCVN: {vn_val} {thr['unit']}
                      </div>
                    </div>""", unsafe_allow_html=True)

            # Bảng đầy đủ
            st.markdown('<div class="divider"></div>', unsafe_allow_html=True)
            st.markdown("#### 📋 Bảng dữ liệu đầy đủ")
            show_cols = ["time", TARGET, "pm2_5", "pm10", "nitrogen_dioxide", "ozone",
                         "temperature_2m", "relative_humidity_2m", "wind_speed_10m"]
            df_show = df_fv[[c for c in show_cols if c in df_fv.columns]].copy()
            df_show = df_show.rename(columns={
                "time": "Thời gian", TARGET: "US AQI", "pm2_5": "PM2.5",
                "pm10": "PM10", "nitrogen_dioxide": "NO₂", "ozone": "O₃",
                "temperature_2m": "Nhiệt độ (°C)", "relative_humidity_2m": "Độ ẩm (%)",
                "wind_speed_10m": "Gió (km/h)",
            })
            st.dataframe(df_show.sort_values("Thời gian", ascending=False),
                         use_container_width=True, hide_index=True, height=350)
            csv_full = df_show.to_csv(index=False)
            st.download_button("⬇️ Tải toàn bộ dữ liệu CSV", data=csv_full,
                               file_name=f"aqi_full_{slug}_{today_f}.csv", mime="text/csv")

        # ── Sub-tab 2: Kết quả mô hình ────────────────────────────────────
        with sub2:
            st.markdown("#### 🏆 Bảng tổng hợp kết quả các mô hình (PCA 95%)")
            st.caption("RMSE = Root Mean Squared Error (thấp hơn = tốt hơn) | "
                       "WLA = Weighted Level Accuracy — tỷ lệ phân loại đúng mức AQI (cao hơn = tốt hơn)")

            prov_data = MODEL_SUMMARY.get(slug, MODEL_SUMMARY["thanh_hoa"])
            rows_m = []
            for model_name, rmse, wla in prov_data["models"]:
                is_best = model_name == prov_data["best"]
                rows_m.append({
                    "Mô hình":    ("⭐ " if is_best else "") + model_name,
                    "RMSE":       f"{rmse:.3f}",
                    "WLA (%)":    f"{wla:.1f}",
                    "_rmse":      rmse,
                    "_best":      is_best,
                })
            df_models = pd.DataFrame(rows_m).drop(columns=["_rmse", "_best"])

            def _style_model(row):
                name = row["Mô hình"]
                if "⭐" in name:
                    return ["background-color:#fff9c4;font-weight:700"]*len(row)
                return [""]*len(row)

            st.dataframe(
                pd.DataFrame(rows_m).drop(columns=["_rmse","_best"]).style.apply(_style_model, axis=1),
                use_container_width=True, hide_index=True,
            )

            # Biểu đồ RMSE so sánh
            df_plot_m = pd.DataFrame(prov_data["models"], columns=["Model","RMSE","WLA"])
            df_plot_m = df_plot_m.sort_values("RMSE")
            colors_m  = ["#ffd700" if m == prov_data["best"] else "#90caf9" for m in df_plot_m["Model"]]
            fig_m = go.Figure(go.Bar(
                x=df_plot_m["RMSE"], y=df_plot_m["Model"],
                orientation="h",
                marker_color=colors_m,
                text=[f"{v:.3f}" for v in df_plot_m["RMSE"]],
                textposition="outside",
            ))
            fig_m.update_layout(
                title=dict(text=f"So sánh RMSE — {prov_data['name']}", font=dict(size=14), x=0.02),
                xaxis_title="RMSE (thấp hơn = tốt hơn)",
                yaxis=dict(tickfont=dict(size=11)),
                height=500, showlegend=False,
                plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
                font=dict(family="Inter, sans-serif", size=12),
                margin=dict(l=10, r=80, t=45, b=10),
            )
            st.plotly_chart(fig_m, use_container_width=True)

            # So sánh 4 tỉnh
            st.markdown("#### 📊 Best model tổng hợp 4 tỉnh")
            summary_rows = []
            for s, data in MODEL_SUMMARY.items():
                best_model = next((m for m in data["models"] if m[0] == data["best"]), None)
                if best_model:
                    summary_rows.append({
                        "Tỉnh": data["name"],
                        "Best model": best_model[0],
                        "n_PC (PCA 95%)": data["n_pc"],
                        "RMSE": f"{best_model[1]:.3f}",
                        "WLA (%)": f"{best_model[2]:.1f}",
                    })
            st.dataframe(pd.DataFrame(summary_rows), use_container_width=True, hide_index=True)



try:
    main()
except Exception as _e:
    import traceback as _tb
    st.error(f"❌ Lỗi khởi động: {_e}")
    st.code(_tb.format_exc())
