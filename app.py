import json
import os
import sys
from pathlib import Path

import pandas as pd
import requests
import streamlit as st

ROOT_DIR = Path(__file__).resolve().parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from src.mobile_price.api import (
    MODEL_LABELS,
    MobileSpecifications,
    PRICE_DESCRIPTIONS,
    build_feature_frame,
    load_artifacts,
)

st.set_page_config(page_title="Phân loại giá điện thoại", layout="wide")
API_URL = os.getenv("MOBILE_PRICE_API_URL", "http://127.0.0.1:8000/predict")
PROJECT_ROOT = ROOT_DIR
MODEL_OPTIONS = {
    "Linear SVM — Nguyễn Hồng Thanh": "svm",
    "Random Forest — Nguyễn Hoàng Thái": "random_forest",
}

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
        width="stretch",
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


def predict_locally(payload: dict, model_name: str) -> dict:
    specifications = MobileSpecifications(**payload)
    classifier, scaler = load_artifacts(model_name)
    feature_frame = build_feature_frame(specifications)
    scaled_features = scaler.transform(feature_frame)

    if hasattr(classifier, "feature_names_in_"):
        scaled_features = pd.DataFrame(
            scaled_features,
            columns=classifier.feature_names_in_,
        )

    prediction = int(classifier.predict(scaled_features)[0])
    segment, description = PRICE_DESCRIPTIONS[prediction]
    return {
        "model": MODEL_LABELS[model_name],
        "segment": segment,
        "description": description,
    }


if st.button("Dự đoán phân khúc", type="primary", width="stretch"):
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

    model_name = MODEL_OPTIONS[selected_model_label]
    try:
        response = requests.post(
            API_URL,
            params={"model": model_name},
            json=payload,
            timeout=10,
        )
        response.raise_for_status()
        result = response.json()
    except requests.RequestException as error:
        st.warning(f"Không gọi được API; đang thử dự đoán cục bộ: {error}")
        try:
            result = predict_locally(payload, model_name)
        except FileNotFoundError as model_error:
            st.error(f"Không tìm thấy model để dự đoán: {model_error}")
        else:
            st.markdown(f"### {result['model']} — {result['segment']}")
            st.write(result["description"])
    else:
        st.markdown(f"### {result['model']} — {result['segment']}")
        st.write(result["description"])