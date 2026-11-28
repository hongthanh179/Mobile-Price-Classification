import streamlit as st
import os
import json
from pathlib import Path

import requests

# Cấu hình trang hiển thị
st.set_page_config(page_title="Phân loại giá điện thoại", layout="wide")
API_URL = os.getenv("MOBILE_PRICE_API_URL", "http://127.0.0.1:8000/predict")
PROJECT_ROOT = Path(__file__).resolve().parent
MODEL_OPTIONS = {
    "Linear SVM — Nguyễn Hồng Thanh": "svm",
    "Random Forest — Nguyễn Hoàng Thái": "random_forest",
}

# Tiêu đề và giới thiệu
st.title("Dự đoán phân khúc giá thiết bị di động")
st.write(
    "Ứng dụng hỗ trợ ước tính phân khúc giá của điện thoại dựa trên cấu hình phần cứng. "
    "Chọn một trong hai mô hình do các thành viên xây dựng để so sánh dự đoán."
)
selected_model_label = st.selectbox("Mô hình dự đoán", list(MODEL_OPTIONS))
st.caption(f"Mô hình đang chọn: {selected_model_label}")

try:
    with open(PROJECT_ROOT / "reports" / "metrics.json", encoding="utf-8") as file:
        svm_metrics = json.load(file)
    with open(PROJECT_ROOT / "reports" / "metrics_rf.json", encoding="utf-8") as file:
        rf_metrics = json.load(file)
except (OSError, json.JSONDecodeError) as error:
    st.error(f"Không đọc được metrics để so sánh mô hình: {error}")
else:
    st.subheader("So sánh kết quả trên tập kiểm thử")
    st.dataframe(
        [
            {
                "Thành viên": "Nguyễn Hồng Thanh",
                "Mô hình": "Linear SVM",
                "Accuracy": f"{svm_metrics['accuracy']:.2%}",
                "Macro F1": f"{svm_metrics['macro avg']['f1-score']:.2%}",
            },
            {
                "Thành viên": "Nguyễn Hoàng Thái",
                "Mô hình": "Random Forest",
                "Accuracy": f"{rf_metrics['test']['accuracy']:.2%}",
                "Macro F1": f"{rf_metrics['test']['f1_macro']:.2%}",
            },
        ],
        use_container_width=True,
    )
st.write("---")

col1, col2, col3 = st.columns(3)

with col1:
    st.subheader("Bộ xử lý và bộ nhớ")
    ram = st.slider("Dung lượng RAM (MB)", 256, 4000, 2048, step=128)
    battery_power = st.slider("Dung lượng pin (mAh)", 500, 2000, 1200, step=50)
    int_memory = st.slider("Bộ nhớ trong (GB)", 2, 64, 32)
    n_cores = st.selectbox("Số nhân CPU", [1, 2, 4, 6, 8], index=3)
    clock_speed = st.slider("Xung nhịp CPU (GHz)", 0.5, 3.0, 1.8, step=0.1)

with col2:
    st.subheader("Màn hình và camera")
    px_width = st.slider("Độ phân giải màn hình ngang (pixel)", 500, 2000, 1200)
    px_height = st.slider("Độ phân giải màn hình dọc (pixel)", 100, 2000, 800)
    sc_h = st.slider("Chiều dài máy (cm)", 5, 20, 14)
    sc_w = st.slider("Chiều rộng màn hình (cm)", 1, 18, 7)
    pc = st.slider("Độ phân giải camera chính (MP)", 0, 20, 10)
    fc = st.slider("Độ phân giải camera phụ (MP)", 0, 20, 5)

with col3:
    st.subheader("Kích thước và kết nối")
    mobile_wt = st.slider("Khối lượng máy (g)", 80, 200, 140)
    m_dep = st.slider("Độ dày thân máy (cm)", 0.1, 1.0, 0.5, step=0.1)
    talk_time = st.slider("Thời gian đàm thoại liên tục (giờ)", 2, 20, 10)

    st.write("Khả năng kết nối:")
    dual_sim = 1 if st.checkbox("Hỗ trợ 2 SIM", value=True) else 0
    four_g = 1 if st.checkbox("Hỗ trợ mạng 4G", value=True) else 0
    three_g = 1 if st.checkbox("Hỗ trợ mạng 3G", value=True) else 0
    wifi = 1 if st.checkbox("Hỗ trợ Wi-Fi", value=True) else 0
    blue = 1 if st.checkbox("Hỗ trợ Bluetooth", value=True) else 0
    touch_screen = 1 if st.checkbox("Màn hình cảm ứng", value=True) else 0

st.write("---")

if st.button("Dự đoán phân khúc", type="primary", use_container_width=True):
    payload = {
        "battery_power": battery_power,
        "blue": blue,
        "clock_speed": clock_speed,
        "dual_sim": dual_sim,
        "fc": fc,
        "four_g": four_g,
        "int_memory": int_memory,
        "m_dep": m_dep,
        "mobile_wt": mobile_wt,
        "n_cores": n_cores,
        "pc": pc,
        "px_height": px_height,
        "px_width": px_width,
        "ram": ram,
        "sc_h": sc_h,
        "sc_w": sc_w,
        "talk_time": talk_time,
        "three_g": three_g,
        "touch_screen": touch_screen,
        "wifi": wifi,
    }
    try:
        response = requests.post(
            API_URL,
            params={"model": MODEL_OPTIONS[selected_model_label]},
            json=payload,
            timeout=10,
        )
        response.raise_for_status()
        result = response.json()
    except requests.RequestException as error:
        st.error(f"Không gọi được API dự đoán ({API_URL}): {error}")
    else:
        st.markdown(f"### {result['model']} — {result['segment']}")
        st.write(result["description"])