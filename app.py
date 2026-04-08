"""
╔══════════════════════════════════════════════════════════════════╗
║      AQI Forecast Web Demo — Miền Trung Việt Nam                ║
║      Auto-sync artifacts từ Google Drive mỗi lần khởi động      ║
╚══════════════════════════════════════════════════════════════════╝

Cấu trúc thư mục:
    DATN/
    ├── app.py
    ├── requirements.txt
    ├── credentials.json        ← Tải từ Google Cloud Console (1 lần)
    ├── .streamlit/
    │   └── secrets.toml        ← (Tùy chọn) lưu DRIVE_FOLDER_ID
    └── best_pca_models/        ← Tự tạo khi sync lần đầu

Chạy:
    streamlit run app.py
"""

# ── Standard lib ─────────────────────────────────────────────────────────────
import io
import json
import os
import warnings
from datetime import date, datetime, timedelta, timezone
from pathlib import Path

# ── Third-party ──────────────────────────────────────────────────────────────
import joblib
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import requests
import streamlit as st

warnings.filterwarnings("ignore")


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 1 ── CẤU HÌNH
# ═══════════════════════════════════════════════════════════════════════════════

BASE_DIR         = Path(__file__).parent
BEST_MODEL_DIR   = BASE_DIR / "best_pca_models"
CREDENTIALS_FILE = BASE_DIR / "credentials.json"
TOKEN_FILE       = BASE_DIR / "token.json"
SYNC_STATE_FILE  = BASE_DIR / ".sync_state.json"   # lưu folder_id + last sync time

DRIVE_FOLDER_NAME = "best_pca_models"
DRIVE_SCOPES      = ["https://www.googleapis.com/auth/drive.readonly"]
SYNC_FILE_EXTS    = {".pkl", ".csv"}

# ── Hằng số pipeline (đồng bộ notebook 06) ───────────────────────────────────
TARGET      = "us_aqi"
HORIZONS    = [1, 3, 6, 12, 24, 48, 72]
TARGET_COLS = [f"target_t{h}h" for h in HORIZONS]

# Biến luôn được giữ bất kể tương quan (đồng bộ notebook 06 — FORCE_KEEP)
FORCE_KEEP = [
    "hour_sin", "hour_cos",
    "dow_sin",  "dow_cos",
    "month_sin","month_cos",
    "is_dry_season",
    "pm25_pm10_ratio",
    "aqi_diff_24h",
]

META_COLS_EXT = ["time", "province", "slug",
                 "source_aq", "source_weather", "season", "year"]

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
    "us_aqi":               (0, 500),
    "pm2_5":                (0, 500),
    "pm10":                 (0, 1000),
    "carbon_monoxide":      (0, 50_000),
    "nitrogen_dioxide":     (0, 1000),
    "sulphur_dioxide":      (0, 2000),
    "ozone":                (0, 600),
    "temperature_2m":       (-10, 50),
    "relative_humidity_2m": (0, 100),
    "pressure_msl":         (900, 1100),
    "wind_speed_10m":       (0, 150),
    "shortwave_radiation":  (0, 1500),
}

# ── Thang AQI ─────────────────────────────────────────────────────────────────
AQI_BINS        = [0, 50, 100, 150, 200, 300, 500]
AQI_LABELS      = ["Tốt", "Trung bình", "Kém", "Xấu", "Rất xấu", "Nguy hại"]
AQI_COLORS      = ["#00e400", "#ffff00", "#ff7e00", "#ff0000", "#8f3f97", "#7e0023"]
AQI_TEXT_COLORS = ["#000",    "#000",    "#000",    "#fff",    "#fff",    "#fff"]

# ── Tỉnh ─────────────────────────────────────────────────────────────────────
PROVINCES = {
    "Thanh Hóa": {"slug": "thanh_hoa", "lat": 19.808, "lon": 105.776},
    "Nghệ An":   {"slug": "nghe_an",   "lat": 19.234, "lon": 104.920},
    "Hà Tĩnh":   {"slug": "ha_tinh",   "lat": 18.343, "lon": 105.906},
    "Huế":       {"slug": "hue",       "lat": 16.462, "lon": 107.595},
}

# ── Khuyến nghị ──────────────────────────────────────────────────────────────
RECOMMENDATIONS = {
    0: {
        "icon": "🟢", "label": "Tốt",
        "desc": "Chất lượng không khí tốt. Không ảnh hưởng tới sức khỏe.",
        "general":   ["Thích hợp mọi hoạt động ngoài trời.", "Lý tưởng để chạy bộ, đạp xe, dã ngoại."],
        "sensitive": ["Nhóm nhạy cảm hoạt động bình thường."],
        "safe_hours":"Tất cả các giờ trong ngày đều an toàn.",
        "activities":["🏃 Chạy bộ / đi bộ ngoài trời", "🚴 Đạp xe", "⚽ Thể thao ngoài trời", "🌳 Picnic / dã ngoại"],
        "avoid":     [],
    },
    1: {
        "icon": "🟡", "label": "Trung bình",
        "desc": "Chất lượng không khí chấp nhận được. Một số chất ô nhiễm ảnh hưởng người rất nhạy cảm.",
        "general":   ["Đa số người hoạt động ngoài trời bình thường.", "Hạn chế hoạt động cường độ cao kéo dài."],
        "sensitive": ["Người hen suyễn, tim mạch nên hạn chế tập nặng.", "Đeo khẩu trang N95 khi ra ngoài lâu."],
        "safe_hours":"Sáng sớm 5–8h và chiều tối 17–20h thường tốt hơn.",
        "activities":["🚶 Đi bộ nhẹ nhàng", "🏋️ Tập trong nhà", "🛒 Sinh hoạt bình thường"],
        "avoid":     ["❌ Tránh cardio ngoài trời > 1 giờ"],
    },
    2: {
        "icon": "🟠", "label": "Kém",
        "desc": "Chất lượng không khí kém. Có thể gây hại cho nhóm nhạy cảm.",
        "general":   ["Giảm thời gian hoạt động ngoài trời.", "Đóng cửa sổ, bật máy lọc không khí nếu có."],
        "sensitive": ["Người già, trẻ em, bà bầu nên ở trong nhà.", "Bắt buộc đeo khẩu trang N95/KN95."],
        "safe_hours":"Tương đối an toàn: 6–8h sáng và 18–21h tối. Tránh 10–16h.",
        "activities":["🏠 Ưu tiên hoạt động trong nhà", "🚗 Di chuyển bằng xe có điều hòa"],
        "avoid":     ["❌ Tránh tập ngoài trời", "❌ Không mở cửa sổ ban ngày"],
    },
    3: {
        "icon": "🔴", "label": "Xấu",
        "desc": "Chất lượng không khí xấu. Ảnh hưởng sức khỏe toàn dân.",
        "general":   ["Hạn chế ra ngoài tối đa.", "Đóng kín cửa, dùng máy lọc không khí."],
        "sensitive": ["Người bệnh hô hấp, tim mạch phải ở trong nhà.", "Liên hệ bác sĩ nếu có triệu chứng."],
        "safe_hours":"Không có khung giờ an toàn. Nếu bắt buộc, chọn trước 7h sáng.",
        "activities":["🏠 Ở trong nhà", "📱 Làm việc / học online"],
        "avoid":     ["❌ Không ra ngoài không cần thiết", "❌ Không mở cửa sổ", "❌ Không tập ngoài trời"],
    },
    4: {
        "icon": "🟣", "label": "Rất xấu",
        "desc": "Rất xấu. Khẩn cấp với nhóm nhạy cảm.",
        "general":   ["Không ra ngoài trừ khẩn cấp.", "Máy lọc không khí HEPA hoạt động liên tục."],
        "sensitive": ["Nguy hiểm — ở trong nhà tuyệt đối.", "Gọi 115 nếu khó thở, đau ngực."],
        "safe_hours":"Không có khung giờ an toàn.",
        "activities":["🏠 Ở trong nhà tuyệt đối"],
        "avoid":     ["❌ Tuyệt đối không ra ngoài", "❌ Không mở cửa sổ"],
    },
    5: {
        "icon": "⛔", "label": "Nguy hại",
        "desc": "Nguy hại — tình trạng khẩn cấp môi trường.",
        "general":   ["Khẩn cấp. Thực hiện theo chỉ dẫn cơ quan chức năng.", "Ở trong nhà kín, máy lọc HEPA."],
        "sensitive": ["Sơ tán nếu được.", "Hotline y tế: 1800 599 920."],
        "safe_hours":"Không có khung giờ an toàn. Tình trạng khẩn cấp.",
        "activities":["🏠 Ở trong nhà kín tuyệt đối"],
        "avoid":     ["❌ Không ra ngoài bất kỳ lý do nào"],
    },
}


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 2 ── GOOGLE DRIVE AUTO-SYNC
# ═══════════════════════════════════════════════════════════════════════════════

def _load_drive_creds():
    """
    Load / refresh Google Drive credentials.
    Ưu tiên: token.json (refresh tự động) → credentials.json (lần đầu).
    Trả về Credentials object hoặc None nếu chưa cấu hình.
    """
    try:
        from google.auth.transport.requests import Request
        from google.oauth2.credentials import Credentials
        from google_auth_oauthlib.flow import InstalledAppFlow
    except ImportError:
        return None, "Thiếu thư viện. Chạy: pip install google-api-python-client google-auth-oauthlib"

    creds = None

    if TOKEN_FILE.exists():
        try:
            creds = Credentials.from_authorized_user_file(str(TOKEN_FILE), DRIVE_SCOPES)
        except Exception:
            TOKEN_FILE.unlink(missing_ok=True)

    if not creds or not creds.valid:
        if creds and creds.expired and creds.refresh_token:
            try:
                creds.refresh(Request())
                TOKEN_FILE.write_text(creds.to_json())
            except Exception as e:
                TOKEN_FILE.unlink(missing_ok=True)
                return None, f"Token hết hạn, không thể refresh: {e}"
        else:
            if not CREDENTIALS_FILE.exists():
                return None, (
                    "Chưa tìm thấy `credentials.json`.\n"
                    "Xem hướng dẫn Setup trong sidebar để tạo file này."
                )
            try:
                flow = InstalledAppFlow.from_client_secrets_file(
                    str(CREDENTIALS_FILE), DRIVE_SCOPES
                )
                creds = flow.run_local_server(port=0, open_browser=True)
                TOKEN_FILE.write_text(creds.to_json())
            except Exception as e:
                return None, f"Lỗi đăng nhập OAuth: {e}"

    return creds, None


def _get_drive_folder_id(service) -> tuple[str | None, str | None]:
    """Lấy folder ID từ cache hoặc tìm trên Drive theo tên."""
    # Thử đọc cache
    if SYNC_STATE_FILE.exists():
        try:
            state = json.loads(SYNC_STATE_FILE.read_text())
            if fid := state.get("folder_id"):
                return fid, None
        except Exception:
            pass

    # Tìm trên Drive
    try:
        results = (
            service.files()
            .list(
                q=(f"name='{DRIVE_FOLDER_NAME}' "
                   f"and mimeType='application/vnd.google-apps.folder' "
                   f"and trashed=false"),
                fields="files(id, name)",
                pageSize=10,
            )
            .execute()
        )
        files = results.get("files", [])
    except Exception as e:
        return None, f"Lỗi tìm folder: {e}"

    if not files:
        return None, (
            f"Không tìm thấy thư mục `{DRIVE_FOLDER_NAME}` trên Google Drive.\n"
            "Đảm bảo thư mục tồn tại và tài khoản có quyền truy cập."
        )

    folder_id = files[0]["id"]

    # Lưu cache
    state = json.loads(SYNC_STATE_FILE.read_text()) if SYNC_STATE_FILE.exists() else {}
    state["folder_id"] = folder_id
    SYNC_STATE_FILE.write_text(json.dumps(state, indent=2))

    return folder_id, None


def _parse_drive_time(dt_str: str) -> datetime:
    return datetime.fromisoformat(dt_str.replace("Z", "+00:00"))


def _file_needs_update(drive_file: dict, local_path: Path) -> bool:
    """True nếu file local không tồn tại hoặc cũ hơn Drive."""
    if not local_path.exists():
        return True
    local_mtime  = datetime.fromtimestamp(local_path.stat().st_mtime, tz=timezone.utc)
    drive_mtime  = _parse_drive_time(drive_file["modifiedTime"])
    return drive_mtime > local_mtime


def run_sync(force: bool = False) -> dict:
    """
    Thực hiện sync artifacts từ Google Drive.

    Args:
        force: Nếu True, bỏ qua kiểm tra thời gian và tải lại tất cả.

    Returns:
        dict với các key:
            ok          (bool)   : sync thành công không
            downloaded  (int)    : số file đã tải
            skipped     (int)    : số file bỏ qua
            total       (int)    : tổng số file trên Drive
            message     (str)    : thông báo kết quả
            error       (str|None): thông báo lỗi nếu có
    """
    result = {"ok": False, "downloaded": 0, "skipped": 0, "total": 0,
              "message": "", "error": None}

    # 1. Xác thực
    try:
        from googleapiclient.discovery import build
        from googleapiclient.http import MediaIoBaseDownload
    except ImportError:
        result["error"] = "Thiếu thư viện Google. Chạy: pip install google-api-python-client google-auth-oauthlib"
        return result

    creds, err = _load_drive_creds()
    if err:
        result["error"] = err
        return result

    try:
        service = build("drive", "v3", credentials=creds, cache_discovery=False)
    except Exception as e:
        result["error"] = f"Không thể kết nối Drive API: {e}"
        return result

    # 2. Lấy folder ID
    folder_id, err = _get_drive_folder_id(service)
    if err:
        result["error"] = err
        return result

    # 3. Liệt kê file trên Drive
    try:
        files = (
            service.files()
            .list(
                q=f"'{folder_id}' in parents and trashed=false",
                fields="files(id, name, modifiedTime, size)",
                pageSize=100,
            )
            .execute()
            .get("files", [])
        )
    except Exception as e:
        result["error"] = f"Lỗi liệt kê files: {e}"
        return result

    # Lọc theo extension
    files = [f for f in files if Path(f["name"]).suffix in SYNC_FILE_EXTS]
    result["total"] = len(files)

    if not files:
        result["ok"]      = True
        result["message"] = "Không tìm thấy file .pkl/.csv trên Drive."
        return result

    # 4. Tải từng file nếu cần
    BEST_MODEL_DIR.mkdir(parents=True, exist_ok=True)
    downloaded = 0

    for f in files:
        local_path = BEST_MODEL_DIR / f["name"]
        if not force and not _file_needs_update(f, local_path):
            result["skipped"] += 1
            continue

        # Download
        try:
            buf      = io.BytesIO()
            request  = service.files().get_media(fileId=f["id"])
            dl       = MediaIoBaseDownload(buf, request)
            done     = False
            while not done:
                _, done = dl.next_chunk()
            local_path.write_bytes(buf.getvalue())

            # Đồng bộ thời gian modify để lần sau không tải lại
            drive_ts = _parse_drive_time(f["modifiedTime"]).timestamp()
            os.utime(local_path, (drive_ts, drive_ts))
            downloaded += 1
        except Exception as e:
            result["error"] = f"Lỗi tải {f['name']}: {e}"
            return result

    # 5. Lưu thời điểm sync thành công
    state = json.loads(SYNC_STATE_FILE.read_text()) if SYNC_STATE_FILE.exists() else {}
    state["last_sync"] = datetime.now(tz=timezone.utc).isoformat()
    state["last_downloaded"] = downloaded
    SYNC_STATE_FILE.write_text(json.dumps(state, indent=2))

    result["ok"]         = True
    result["downloaded"] = downloaded
    result["skipped"]    = result["total"] - downloaded
    result["message"]    = (
        f"Đã tải {downloaded} file mới." if downloaded
        else "Tất cả model đã up-to-date ✓"
    )
    return result


def get_last_sync_info() -> dict:
    """Đọc thông tin sync lần cuối từ cache."""
    if not SYNC_STATE_FILE.exists():
        return {}
    try:
        return json.loads(SYNC_STATE_FILE.read_text())
    except Exception:
        return {}


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 3 ── FEATURE ENGINEERING (đồng bộ notebook 01 & 06)
# ═══════════════════════════════════════════════════════════════════════════════

def impute_df(df: pd.DataFrame) -> pd.DataFrame:
    """Điền missing theo đúng quy trình preprocessing của pipeline."""
    df = df.copy()
    extra = ["aerosol_optical_depth", "dust", "dew_point_2m",
             "apparent_temperature", "precipitation", "rain",
             "cloud_cover", "wind_direction_10m", "wind_gusts_10m", "european_aqi"]
    impute_cols = [c for c in list(PHYSICAL_BOUNDS) + extra if c in df.columns]

    for col, (lo, hi) in PHYSICAL_BOUNDS.items():
        if col in df.columns:
            df.loc[(df[col] < lo) | (df[col] > hi), col] = np.nan

    for col in impute_cols:
        limit = 3 if col == TARGET else 6
        df[col] = df[col].interpolate(
            method="linear", limit=limit, limit_direction="both")
        still_nan = df[col].isna()
        if still_nan.any():
            rolled = df[col].rolling(24, min_periods=3, center=True).mean()
            df.loc[still_nan, col] = rolled[still_nan]

    return df.ffill().bfill()


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    81 features — đồng bộ hoàn toàn với notebook 01.
    Cần ít nhất 25 hàng liên tiếp để tính đủ rolling 24h.
    """
    df = df.sort_values("time").reset_index(drop=True).copy()

    # Lag features
    for h in [1, 3, 6, 12, 24]:
        df[f"aqi_lag_{h}h"]  = df[TARGET].shift(h)
        df[f"pm25_lag_{h}h"] = df["pm2_5"].shift(h)
    for h in [1, 3, 6]:
        df[f"temp_lag_{h}h"]  = df["temperature_2m"].shift(h)
        df[f"humid_lag_{h}h"] = df["relative_humidity_2m"].shift(h)
        df[f"wind_lag_{h}h"]  = df["wind_speed_10m"].shift(h)

    # Rolling statistics
    for w in [3, 6, 12, 24]:
        df[f"aqi_rmean_{w}h"] = df[TARGET].rolling(w, min_periods=1).mean()
        df[f"aqi_rmax_{w}h"]  = df[TARGET].rolling(w, min_periods=1).max()
        df[f"aqi_rmin_{w}h"]  = df[TARGET].rolling(w, min_periods=1).min()
        df[f"aqi_rstd_{w}h"]  = df[TARGET].rolling(w, min_periods=1).std().fillna(0)
    for w in [6, 24]:
        df[f"pm25_rmean_{w}h"]  = df["pm2_5"].rolling(w, min_periods=1).mean()
        df[f"wind_rmean_{w}h"]  = df["wind_speed_10m"].rolling(w, min_periods=1).mean()
        df[f"humid_rmean_{w}h"] = df["relative_humidity_2m"].rolling(w, min_periods=1).mean()

    # Diff features
    df["aqi_diff_1h"]  = df[TARGET].diff(1).fillna(0)
    df["aqi_diff_3h"]  = df[TARGET].diff(3).fillna(0)
    df["aqi_diff_24h"] = df[TARGET].diff(24).fillna(0)

    # Thời gian sin/cos
    h_col = df["time"].dt.hour
    m_col = df["time"].dt.month
    dw    = df["time"].dt.dayofweek

    df["hour_sin"]  = np.sin(2 * np.pi * h_col / 24)
    df["hour_cos"]  = np.cos(2 * np.pi * h_col / 24)
    df["month_sin"] = np.sin(2 * np.pi * m_col / 12)
    df["month_cos"] = np.cos(2 * np.pi * m_col / 12)
    df["dow_sin"]   = np.sin(2 * np.pi * dw / 7)
    df["dow_cos"]   = np.cos(2 * np.pi * dw / 7)

    df["hour"]        = h_col
    df["month"]       = m_col
    df["day_of_week"] = dw
    df["day"]         = df["time"].dt.day
    df["year"]        = df["time"].dt.year

    df["season"] = m_col.map({
        3:"Mùa khô", 4:"Mùa khô", 5:"Mùa khô",
        6:"Mùa khô", 7:"Mùa khô", 8:"Mùa khô",
        9:"Mùa mưa",10:"Mùa mưa",11:"Mùa mưa",
       12:"Mùa mưa", 1:"Mùa mưa", 2:"Mùa mưa",
    })
    df["is_dry_season"] = df["season"].map({"Mùa khô": 1, "Mùa mưa": 0}).astype(int)

    # Interaction features
    df["pm25_pm10_ratio"] = df["pm2_5"] / (df["pm10"] + 1e-6)
    df["humid_x_pm25"]    = df["relative_humidity_2m"] * df["pm2_5"]
    df["temp_x_wind"]     = df["temperature_2m"] * df["wind_speed_10m"]

    # Target columns (dummy khi inference — không dùng)
    for h in HORIZONS:
        df[f"target_t{h}h"] = df[TARGET].shift(-h)

    return df


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 4 ── DATA FETCHING (Open-Meteo API)
# ═══════════════════════════════════════════════════════════════════════════════

@st.cache_data(ttl=3600, show_spinner=False)
def fetch_data(lat: float, lon: float, start_date: str, end_date: str) -> pd.DataFrame | None:
    """
    Gọi Air Quality API + Archive Weather API của Open-Meteo.
    Cache 1 giờ để tránh gọi lại liên tục.
    """
    tz = "Asia/Bangkok"
    try:
        aq_resp = requests.get(
            "https://air-quality-api.open-meteo.com/v1/air-quality",
            params={
                "latitude": lat, "longitude": lon,
                "start_date": start_date, "end_date": end_date,
                "timezone": tz, "domains": "cams_global", "cell_selection": "land",
                "hourly": ",".join(AQ_VARS),
            },
            timeout=30,
        )
        aq_resp.raise_for_status()

        wt_resp = requests.get(
            "https://archive-api.open-meteo.com/v1/archive",
            params={
                "latitude": lat, "longitude": lon,
                "start_date": start_date, "end_date": end_date,
                "timezone": tz,
                "hourly": ",".join(WEATHER_VARS),
            },
            timeout=30,
        )
        wt_resp.raise_for_status()

    except requests.RequestException as e:
        st.error(f"❌ Lỗi Open-Meteo API: {e}")
        return None

    df_aq = pd.DataFrame(aq_resp.json().get("hourly", {}))
    df_wt = pd.DataFrame(wt_resp.json().get("hourly", {}))

    if df_aq.empty or df_wt.empty:
        return None

    df_aq["time"] = pd.to_datetime(df_aq["time"])
    df_wt["time"] = pd.to_datetime(df_wt["time"])
    df = pd.merge(df_aq, df_wt, on="time", how="inner")
    return df.sort_values("time").reset_index(drop=True)


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 5 ── MODEL INFERENCE
# ═══════════════════════════════════════════════════════════════════════════════

@st.cache_resource(show_spinner=False)
def load_artifacts(slug: str) -> dict | None:
    """Load 5 artifact files. Cache vào RAM (st.cache_resource)."""
    files = {
        "model":       f"{slug}_best_model.pkl",
        "scaler_pca":  f"{slug}_scaler_pca.pkl",
        "pca":         f"{slug}_pca.pkl",
        "strong_vars": f"{slug}_strong_vars.pkl",
        "info":        f"{slug}_inference_info.pkl",
    }
    arts = {}
    for key, fname in files.items():
        path = BEST_MODEL_DIR / fname
        if not path.exists():
            return None
        arts[key] = joblib.load(path)
    return arts


def predict_aqi(features_df: pd.DataFrame, arts: dict) -> dict | None:
    """
    Inference pipeline:
        features_df[strong_vars] → StandardScaler → PCA → model.predict()

    Đồng bộ hoàn toàn với hàm predict_aqi() trong notebook 06.
    """
    strong_vars = arts["strong_vars"]

    # Đảm bảo tất cả strong_vars có mặt
    sample = features_df.copy()
    for v in strong_vars:
        if v not in sample.columns:
            sample[v] = 0.0

    X = sample[strong_vars].iloc[[-1]].values
    X = np.nan_to_num(X, nan=0.0)

    X_scaled = arts["scaler_pca"].transform(X)
    X_pca    = arts["pca"].transform(X_scaled)
    pred     = arts["model"].predict(X_pca)[0]
    pred     = np.clip(pred, 0, 500)

    return {h: float(p) for h, p in zip(HORIZONS, pred)}


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 6 ── UI HELPERS
# ═══════════════════════════════════════════════════════════════════════════════

def aqi_level(val: float) -> int:
    val = max(0.0, float(val))
    for i in range(len(AQI_BINS) - 1):
        if AQI_BINS[i] <= val < AQI_BINS[i + 1]:
            return i
    return len(AQI_LABELS) - 1


def aqi_color(val: float) -> str:
    return AQI_COLORS[aqi_level(val)]


def aqi_badge(val: float) -> str:
    lvl = aqi_level(val)
    return (
        f'<span style="background:{AQI_COLORS[lvl]};color:{AQI_TEXT_COLORS[lvl]};'
        f'padding:3px 14px;border-radius:999px;font-weight:700">'
        f'{val:.0f} — {AQI_LABELS[lvl]}</span>'
    )


def render_gauge(value: float, subtitle: str = "") -> go.Figure:
    lvl = aqi_level(value)
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=value,
        number={"font": {"size": 52, "color": AQI_COLORS[lvl]}},
        title={"text": f"US AQI<br><span style='font-size:13px;color:#888'>{subtitle}</span>",
               "font": {"size": 15}},
        gauge={
            "axis": {"range": [0, 300], "tickwidth": 1, "tickcolor": "#aaa"},
            "bar":  {"color": AQI_COLORS[lvl], "thickness": 0.28},
            "steps": [
                {"range": [0,   50],  "color": "#00e400"},
                {"range": [50,  100], "color": "#ffff00"},
                {"range": [100, 150], "color": "#ff7e00"},
                {"range": [150, 200], "color": "#ff0000"},
                {"range": [200, 300], "color": "#8f3f97"},
            ],
            "threshold": {
                "line": {"color": "#222", "width": 3},
                "thickness": 0.75, "value": value,
            },
        },
    ))
    fig.update_layout(height=290, margin=dict(l=20, r=20, t=50, b=10),
                      paper_bgcolor="rgba(0,0,0,0)")
    return fig


def render_forecast_chart(predictions: dict) -> go.Figure:
    hs  = list(predictions.keys())
    vs  = list(predictions.values())
    fig = go.Figure(go.Bar(
        x=[f"t+{h}h" for h in hs],
        y=vs,
        marker_color=[aqi_color(v) for v in vs],
        marker_line_color="rgba(0,0,0,0.15)",
        marker_line_width=1.5,
        text=[f"<b>{v:.0f}</b><br>{AQI_LABELS[aqi_level(v)]}" for v in vs],
        textposition="outside",
        textfont={"size": 11},
    ))
    for thr, lbl, col in [
        (50, "Tốt", "#00e400"), (100, "Trung bình", "#ffff00"),
        (150, "Kém", "#ff7e00"), (200, "Xấu", "#ff0000"),
    ]:
        fig.add_hline(y=thr, line_dash="dot", line_color=col, line_width=1.5,
                      annotation_text=f" {lbl}", annotation_position="left",
                      annotation_font_color=col, annotation_font_size=11)

    fig.update_layout(
        title={"text": "Dự báo AQI — 7 Chân trời", "font": {"size": 16}},
        xaxis_title="Chân trời dự báo",
        yaxis_title="US AQI",
        yaxis_range=[0, max(max(vs) * 1.3, 180)],
        height=400, showlegend=False,
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
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


def render_recommendations(level: int):
    rec    = RECOMMENDATIONS[level]
    color  = AQI_COLORS[level]
    tcolor = AQI_TEXT_COLORS[level]
    st.markdown(
        f'<div style="background:{color};color:{tcolor};padding:16px 22px;'
        f'border-radius:14px;margin-bottom:18px">'
        f'<h2 style="margin:0;font-size:1.5rem">{rec["icon"]} Mức {rec["label"]}</h2>'
        f'<p style="margin:6px 0 0;font-size:1rem;opacity:0.9">{rec["desc"]}</p>'
        f'</div>',
        unsafe_allow_html=True,
    )
    c1, c2 = st.columns(2)
    with c1:
        st.subheader("📋 Khuyến nghị chung")
        for item in rec["general"]:
            st.markdown(f"• {item}")
        st.subheader("⚠️ Nhóm nhạy cảm")
        for item in rec["sensitive"]:
            st.markdown(f"• {item}")
        if rec["avoid"]:
            st.subheader("🚫 Cần tránh")
            for item in rec["avoid"]:
                st.markdown(item)
    with c2:
        st.subheader("⏰ Khung giờ an toàn")
        st.info(rec["safe_hours"])
        st.subheader("✅ Hoạt động phù hợp")
        for act in rec["activities"]:
            st.markdown(f"• {act}")


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 7 ── MAIN APP
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    st.set_page_config(
        page_title="Dự báo AQI Miền Trung",
        page_icon="🌬️",
        layout="wide",
        initial_sidebar_state="expanded",
    )

    st.markdown("""
    <style>
    .block-container { padding-top: 1.5rem; }
    .main-title {
        font-size: 2.1rem; font-weight: 800; letter-spacing: -0.5px;
        background: linear-gradient(120deg, #1565c0, #0097a7);
        -webkit-background-clip: text; -webkit-text-fill-color: transparent;
    }
    .sync-ok   { color: #2e7d32; font-weight: 600; }
    .sync-warn { color: #e65100; font-weight: 600; }
    .info-card {
        background: #e8f4fd; border-left: 4px solid #1565c0;
        padding: 10px 16px; border-radius: 0 10px 10px 0; margin: 8px 0;
        font-size: 0.92rem;
    }
    </style>
    """, unsafe_allow_html=True)

    # ══════════════════════════════════════════════════════════════════════════
    # SIDEBAR
    # ══════════════════════════════════════════════════════════════════════════
    with st.sidebar:
        st.markdown("## 🌬️ AQI Miền Trung VN")
        st.markdown("---")

        # ── Chọn tỉnh ────────────────────────────────────────────────────────
        province_name = st.selectbox("📍 Chọn tỉnh", list(PROVINCES.keys()))
        prov  = PROVINCES[province_name]
        slug  = prov["slug"]
        lat   = prov["lat"]
        lon   = prov["lon"]

        st.markdown(
            f'<div class="info-card"><b>{province_name}</b><br>'
            f'📌 {lat:.3f}°N &nbsp; {lon:.3f}°E<br>'
            f'🕐 UTC+7 (Asia/Bangkok)</div>',
            unsafe_allow_html=True,
        )

        # ── Auto-sync panel ───────────────────────────────────────────────────
        st.markdown("---")
        st.markdown("### ☁️ Google Drive Sync")

        # Hiển thị trạng thái lần sync cuối
        sync_info = get_last_sync_info()
        if ls := sync_info.get("last_sync"):
            last_dt  = datetime.fromisoformat(ls).astimezone()
            last_str = last_dt.strftime("%H:%M  %d/%m/%Y")
            n_dl     = sync_info.get("last_downloaded", 0)
            st.markdown(
                f'<p class="sync-ok">✅ Sync lần cuối: {last_str}<br>'
                f'&nbsp;&nbsp;&nbsp;Đã tải: {n_dl} file mới</p>',
                unsafe_allow_html=True,
            )
        else:
            st.markdown(
                '<p class="sync-warn">⚠️ Chưa có lịch sử sync</p>',
                unsafe_allow_html=True,
            )

        # Nút sync thủ công
        col_sync, col_force = st.columns(2)
        with col_sync:
            btn_sync = st.button("🔄 Sync ngay", use_container_width=True,
                                 help="Tải file mới hơn từ Drive")
        with col_force:
            btn_force = st.button("⬇️ Force sync", use_container_width=True,
                                  help="Tải lại tất cả, bất kể file local")

        if btn_sync or btn_force:
            with st.spinner("Đang sync từ Google Drive..."):
                res = run_sync(force=btn_force)
            if res["ok"]:
                st.success(f"✅ {res['message']}\n"
                           f"(Tải: {res['downloaded']} | Bỏ qua: {res['skipped']} / {res['total']})")
                load_artifacts.clear()
            else:
                st.error(f"❌ {res['error']}")

        # Setup guide
        with st.expander("📖 Hướng dẫn Setup Drive"):
            st.markdown("""
**Lần đầu cấu hình (làm 1 lần duy nhất):**

1. Vào [Google Cloud Console](https://console.cloud.google.com)
2. **Tạo project** mới (nếu chưa có)
3. **Enable Google Drive API:**
   `APIs & Services → Library → Google Drive API → Enable`
4. **Tạo OAuth credentials:**
   `Credentials → Create Credentials → OAuth client ID → Desktop app`
5. **Tải file JSON** → đổi tên thành `credentials.json`
6. **Đặt vào** thư mục chứa `app.py`
7. **Chạy app** → trình duyệt mở → đăng nhập Google

Sau lần đầu, token được lưu vào `token.json`  
và sẽ **tự refresh** — không cần đăng nhập lại.
""")

        # ── Model info ────────────────────────────────────────────────────────
        st.markdown("---")
        arts = load_artifacts(slug)
        if arts:
            info = arts["info"]
            st.success("✅ Model đã sẵn sàng")
            st.markdown(f"""
| | |
|---|---|
| **Algorithm** | {info.get('model_name','N/A')} |
| **n PC** | {info.get('n_comp','N/A')} |
| **RMSE (test)** | {info.get('test_rmse_avg','N/A')} |
| **WLA (test)** | {info.get('test_wla_avg','N/A')}% |
""")
        else:
            st.error("❌ Chưa có model\nSync từ Drive hoặc copy thủ công")

        if st.button("🔄 Làm mới dữ liệu AQI", use_container_width=True):
            st.cache_data.clear()
            st.rerun()

    # ══════════════════════════════════════════════════════════════════════════
    # AUTO-SYNC KHI MỞ APP (chỉ chạy 1 lần mỗi session)
    # ══════════════════════════════════════════════════════════════════════════
    if "boot_sync_done" not in st.session_state:
        st.session_state["boot_sync_done"] = True
        with st.spinner("⏳ Đang kiểm tra cập nhật model từ Google Drive..."):
            res = run_sync(force=False)
        if res["ok"]:
            if res["downloaded"] > 0:
                load_artifacts.clear()
                st.toast(f"☁️ Đã tải {res['downloaded']} file model mới từ Drive!", icon="✅")
            # Không toast nếu không có gì mới → không gây phân tâm
        else:
            if res["error"] and "credentials.json" not in (res["error"] or ""):
                st.toast(f"⚠️ Sync Drive: {res['error']}", icon="⚠️")
            # Lỗi do chưa cấu hình credentials → im lặng, vẫn dùng model local

    # ══════════════════════════════════════════════════════════════════════════
    # HEADER
    # ══════════════════════════════════════════════════════════════════════════
    st.markdown(' ')
    st.markdown('<div class="main-title">🌬️ Dự báo Chất lượng Không khí</div>',
                unsafe_allow_html=True)
    st.markdown(
        f"Miền Trung Việt Nam &nbsp;·&nbsp; **{province_name}** &nbsp;·&nbsp; "
        f"*Cập nhật: {datetime.now().strftime('%H:%M  %d/%m/%Y')}*"
    )
    st.markdown("---")

    # ══════════════════════════════════════════════════════════════════════════
    # TABS
    # ══════════════════════════════════════════════════════════════════════════
    tab1, tab2, tab3 = st.tabs([
        "📡 Dự báo Realtime",
        "📊 Phân loại & Khuyến nghị",
        "📅 Lịch sử 3 ngày",
    ])

    # ── Lấy dữ liệu chung cho tab 1 & 3 ─────────────────────────────────────
    today     = date.today()
    start_str = (today - timedelta(days=5)).isoformat()
    end_str   = today.isoformat()

    # ══════════════════════════════════════════════════════════════════════════
    # TAB 1 — DỰ BÁO REALTIME
    # ══════════════════════════════════════════════════════════════════════════
    with tab1:
        if arts is None:
            st.error(
                "❌ **Không tìm thấy model artifacts.**\n\n"
                "Vui lòng sync từ Google Drive (sidebar) hoặc copy thủ công "
                f"thư mục `best_pca_models/` vào `{BASE_DIR}`."
            )
            st.stop()

        with st.spinner(f"⏳ Đang lấy dữ liệu từ Open-Meteo cho {province_name}..."):
            df_raw = fetch_data(lat, lon, start_str, end_str)

        if df_raw is None or df_raw.empty:
            st.error("Không lấy được dữ liệu từ Open-Meteo. Thử lại sau.")
            st.stop()

        with st.spinner("🔧 Đang tính features & chạy mô hình..."):
            df_feat = build_features(impute_df(df_raw))
            df_feat = df_feat.dropna(subset=["aqi_lag_24h"])

        if df_feat.empty:
            st.error("Không đủ dữ liệu để tính features (cần ít nhất 25 giờ liên tiếp).")
            st.stop()

        preds = predict_aqi(df_feat, arts)
        row   = df_feat.iloc[-1]
        cur_aqi  = float(row[TARGET]) if pd.notna(row[TARGET]) else preds[1]
        cur_time = row["time"]
        lvl      = aqi_level(cur_aqi)

        # ── Hàng 1: Gauge + metrics ──────────────────────────────────────────
        st.markdown("### 📌 Thời điểm hiện tại")
        cg, cm = st.columns([1, 2])

        with cg:
            st.plotly_chart(
                render_gauge(cur_aqi, cur_time.strftime("%H:%M  %d/%m/%Y")),
                use_container_width=True,
            )
            st.markdown(
                f'<div style="text-align:center;margin-top:-10px">'
                f'{RECOMMENDATIONS[lvl]["icon"]} <b>{AQI_LABELS[lvl]}</b><br>'
                f'<small style="color:#666">{RECOMMENDATIONS[lvl]["desc"]}</small></div>',
                unsafe_allow_html=True,
            )

        with cm:
            st.markdown("**🔬 Chỉ số ô nhiễm:**")
            m1, m2, m3, m4 = st.columns(4)
            for col_ui, (lbl, key, unit) in zip(
                [m1, m2, m3, m4],
                [("PM2.5", "pm2_5", "µg/m³"), ("PM10", "pm10", "µg/m³"),
                 ("NO₂",  "nitrogen_dioxide", "µg/m³"), ("O₃", "ozone", "µg/m³")],
            ):
                val = row.get(key, np.nan)
                col_ui.metric(f"{lbl} ({unit})",
                              f"{val:.1f}" if pd.notna(val) else "N/A")

            st.markdown("**🌤️ Thời tiết:**")
            w1, w2, w3, w4 = st.columns(4)
            for col_ui, (lbl, key, unit) in zip(
                [w1, w2, w3, w4],
                [("🌡️ Nhiệt độ", "temperature_2m", "°C"),
                 ("💧 Độ ẩm",   "relative_humidity_2m", "%"),
                 ("💨 Gió",     "wind_speed_10m", "km/h"),
                 ("☁️ Mây",     "cloud_cover", "%")],
            ):
                val = row.get(key, np.nan)
                col_ui.metric(lbl, f"{val:.1f} {unit}" if pd.notna(val) else "N/A")

            # Alert nếu AQI nguy hiểm
            if lvl >= 3:
                st.error(
                    f"🚨 **Cảnh báo:** AQI đang ở mức **{AQI_LABELS[lvl]}**. "
                    f"{RECOMMENDATIONS[lvl]['desc']}"
                )
            elif lvl == 2:
                st.warning(
                    f"⚠️ **Lưu ý:** AQI đang ở mức **{AQI_LABELS[lvl]}**. "
                    f"{RECOMMENDATIONS[lvl]['desc']}"
                )

        st.markdown("---")

        # ── Hàng 2: Biểu đồ dự báo ──────────────────────────────────────────
        st.markdown("### 📈 Dự báo 7 Chân trời")
        st.plotly_chart(render_forecast_chart(preds), use_container_width=True)

        # Bảng chi tiết
        rows_tbl = []
        for h, v in preds.items():
            lv = aqi_level(v)
            rows_tbl.append({
                "Chân trời":   f"t+{h}h",
                "Thời điểm":   (datetime.now() + timedelta(hours=h)).strftime("%H:%M  %d/%m"),
                "AQI dự báo":  f"{v:.0f}",
                "Mức chất lượng": f"{RECOMMENDATIONS[lv]['icon']} {AQI_LABELS[lv]}",
            })

        def _highlight(row):
            lv = aqi_level(float(row["AQI dự báo"]))
            return [f"background:{AQI_COLORS[lv]};color:{AQI_TEXT_COLORS[lv]}"] * len(row)

        st.dataframe(
            pd.DataFrame(rows_tbl).style.apply(_highlight, axis=1),
            use_container_width=True, hide_index=True,
        )

        st.markdown(
            '<div class="info-card">💡 <b>Lưu ý:</b> Dự báo dựa trên dữ liệu '
            'CAMS Global (Open-Meteo) và mô hình ML huấn luyện trên dữ liệu 2022–2024. '
            'Độ chính xác giảm dần theo chân trời xa.</div>',
            unsafe_allow_html=True,
        )

    # ══════════════════════════════════════════════════════════════════════════
    # TAB 2 — PHÂN LOẠI & KHUYẾN NGHỊ
    # ══════════════════════════════════════════════════════════════════════════
    with tab2:
        st.markdown("### 🏷️ Phân loại & Hệ khuyến nghị AQI")

        mode = st.radio(
            "Chọn chế độ:",
            ["📡 Dùng AQI dự báo t+1h", "✏️ Nhập AQI thủ công", "📖 Xem tất cả mức"],
            horizontal=True,
        )

        if mode == "📡 Dùng AQI dự báo t+1h":
            try:
                val_input = preds[1]
                st.info(f"AQI dự báo t+1h của **{province_name}**: {aqi_badge(val_input)}",
                        icon="📡")
                st.markdown("&nbsp;", unsafe_allow_html=True)
                render_recommendations(aqi_level(val_input))
            except NameError:
                st.warning("Chuyển sang Tab 1 trước để tải dữ liệu dự báo.")

        elif mode == "✏️ Nhập AQI thủ công":
            val_input = st.slider("Giá trị AQI:", 0, 500, 75, 1)
            st.markdown(f"**Mức bạn nhập:** {aqi_badge(val_input)}", unsafe_allow_html=True)
            st.markdown("&nbsp;", unsafe_allow_html=True)
            render_recommendations(aqi_level(val_input))

        else:  # Xem tất cả
            st.markdown("#### 📋 Bảng phân loại VN-AQI")
            st.dataframe(
                pd.DataFrame([{
                    "Mức": f"{RECOMMENDATIONS[i]['icon']} {AQI_LABELS[i]}",
                    "Khoảng AQI": f"{AQI_BINS[i]} – {AQI_BINS[i+1]-1}",
                    "Mô tả": RECOMMENDATIONS[i]["desc"],
                } for i in range(6)]),
                use_container_width=True, hide_index=True,
            )
            st.markdown("---")
            for i in range(6):
                with st.expander(
                    f"{RECOMMENDATIONS[i]['icon']} {AQI_LABELS[i]} "
                    f"(AQI {AQI_BINS[i]}–{AQI_BINS[i+1]-1})"
                ):
                    render_recommendations(i)

    # ══════════════════════════════════════════════════════════════════════════
    # TAB 3 — LỊCH SỬ 3 NGÀY
    # ══════════════════════════════════════════════════════════════════════════
    with tab3:
        st.markdown("### 📅 Lịch sử AQI 3 ngày gần nhất")

        with st.spinner(f"Đang tải dữ liệu lịch sử {province_name}..."):
            hist_start = (today - timedelta(days=3)).isoformat()
            df_hist    = fetch_data(lat, lon, hist_start, end_str)

        if df_hist is None or df_hist.empty:
            st.error("Không lấy được dữ liệu lịch sử.")
            st.stop()

        df_hist = impute_df(df_hist)
        st.plotly_chart(render_history_chart(df_hist), use_container_width=True)

        # Thống kê tóm tắt
        dv = df_hist[df_hist[TARGET].notna()]
        st.markdown("#### 📊 Thống kê")
        s1, s2, s3, s4, s5 = st.columns(5)
        s1.metric("Trung bình", f"{dv[TARGET].mean():.1f}")
        s2.metric("Cao nhất",   f"{dv[TARGET].max():.0f}")
        s3.metric("Thấp nhất",  f"{dv[TARGET].min():.0f}")
        s4.metric("Độ lệch chuẩn", f"{dv[TARGET].std():.1f}")
        s5.metric("% Giờ 'Tốt'", f"{(dv[TARGET]<=50).mean()*100:.0f}%")

        # Phân bố theo mức AQI
        st.markdown("#### 📈 Phân bố theo mức AQI")
        lc = dv[TARGET].apply(aqi_level).value_counts().sort_index()
        c1, c2 = st.columns([1, 1])
        with c1:
            st.dataframe(
                pd.DataFrame({
                    "Mức": [f"{RECOMMENDATIONS[i]['icon']} {AQI_LABELS[i]}" for i in lc.index],
                    "Số giờ": lc.values,
                    "Tỷ lệ (%)": (lc.values / lc.values.sum() * 100).round(1),
                }),
                use_container_width=True, hide_index=True,
            )
        with c2:
            fig_pie = go.Figure(go.Pie(
                labels=[f"{RECOMMENDATIONS[i]['icon']} {AQI_LABELS[i]}" for i in lc.index],
                values=lc.values,
                marker_colors=[AQI_COLORS[i] for i in lc.index],
                hole=0.42,
            ))
            fig_pie.update_layout(
                height=260, margin=dict(l=10, r=10, t=10, b=10),
                paper_bgcolor="rgba(0,0,0,0)",
            )
            st.plotly_chart(fig_pie, use_container_width=True)

        # AQI theo giờ trong ngày
        st.markdown("#### ⏰ AQI trung bình theo giờ trong ngày")
        df_hist["_hour"] = df_hist["time"].dt.hour
        hourly = df_hist.groupby("_hour")[TARGET].agg(["mean", "std"]).reset_index()
        fig_h = go.Figure()
        fig_h.add_trace(go.Scatter(
            x=hourly["_hour"], y=hourly["mean"] + hourly["std"].fillna(0),
            mode="lines", line=dict(width=0), showlegend=False,
        ))
        fig_h.add_trace(go.Scatter(
            x=hourly["_hour"], y=hourly["mean"] - hourly["std"].fillna(0),
            fill="tonexty", fillcolor="rgba(21,101,192,0.12)",
            mode="lines", line=dict(width=0), name="±1 Std",
        ))
        fig_h.add_trace(go.Scatter(
            x=hourly["_hour"], y=hourly["mean"],
            mode="lines+markers", line=dict(color="#1565c0", width=2.5),
            marker=dict(size=6), name="AQI TB",
        ))
        fig_h.update_layout(
            xaxis=dict(tickmode="array", tickvals=list(range(0, 24, 3)),
                       ticktext=[f"{h:02d}:00" for h in range(0, 24, 3)]),
            yaxis_title="AQI", xaxis_title="Giờ trong ngày",
            height=310, hovermode="x unified",
            plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
        )
        st.plotly_chart(fig_h, use_container_width=True)

        # Download
        csv_cols = ["time", TARGET, "pm2_5", "pm10",
                    "temperature_2m", "relative_humidity_2m",
                    "wind_speed_10m", "cloud_cover"]
        csv_cols = [c for c in csv_cols if c in df_hist.columns]
        st.download_button(
            "⬇️ Tải CSV dữ liệu lịch sử",
            data=df_hist[csv_cols].to_csv(index=False).encode("utf-8"),
            file_name=f"aqi_{slug}_{today}.csv",
            mime="text/csv",
        )


if __name__ == "__main__":
    main()
