import streamlit as st
import numpy as np
import pandas as pd
import joblib
from pathlib import Path

# Cấu hình trang hiển thị
st.set_page_config(page_title="Phân loại giá điện thoại", layout="wide")

# Xác định đường dẫn thư mục gốc để nạp mô hình ổn định
BASE_DIR = Path(__file__).resolve().parent


@st.cache_resource
def load_artifacts():
    model_path = BASE_DIR / 'models' / 'best_svm_model.joblib'
    scaler_path = BASE_DIR / 'models' / 'scaler.joblib'

    if not model_path.exists() or not scaler_path.exists():
        raise FileNotFoundError("Chưa tìm thấy mô hình hoặc bộ chuẩn hóa trong thư mục models/.")

    model = joblib.load(model_path)
    scaler = joblib.load(scaler_path)
    return model, scaler


try:
    model, scaler = load_artifacts()
except Exception as e:
    st.error(f"Lỗi khởi tạo hệ thống: {e}")
    st.stop()

# Tiêu đề và giới thiệu
st.title("Dự đoán phân khúc giá thiết bị di động")
st.write(
    "Ứng dụng hỗ trợ ước tính phân khúc giá của điện thoại dựa trên cấu hình phần cứng. "
    "Mô hình sử dụng thuật toán Support Vector Machine (Linear SVM) đã được tối ưu qua kiểm định chéo."
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

# Tính toán các đặc trưng phái sinh
px_area = px_height * px_width
sc_area = sc_h * sc_w

# Khớp đúng thứ tự các thuộc tính như trong quá trình huấn luyện
input_data = pd.DataFrame([[
    battery_power, blue, clock_speed, dual_sim, fc, four_g,
    int_memory, m_dep, mobile_wt, n_cores, pc, px_height,
    px_width, ram, sc_h, sc_w, talk_time, three_g,
    touch_screen, wifi, px_area, sc_area
]], columns=[
    'battery_power', 'blue', 'clock_speed', 'dual_sim', 'fc', 'four_g',
    'int_memory', 'm_dep', 'mobile_wt', 'n_cores', 'pc', 'px_height',
    'px_width', 'ram', 'sc_h', 'sc_w', 'talk_time', 'three_g',
    'touch_screen', 'wifi', 'px_area', 'sc_area'
])

st.write("---")

if st.button("Dự đoán phân khúc", type="primary", use_container_width=True):
    scaled_input = scaler.transform(input_data)
    prediction = int(model.predict(scaled_input)[0])

    price_descriptions = {
        0: (
            "Phân khúc phổ thông (Mức 0)",
            "Cấu hình ở mức cơ bản, tối ưu cho nhu cầu liên lạc thông thường và các tác vụ nhẹ."
        ),
        1: (
            "Phân khúc tầm trung (Mức 1)",
            "Phần cứng cân đối giữa hiệu năng xử lý, thời lượng pin và giá thành."
        ),
        2: (
            "Phân khúc cận cao cấp (Mức 2)",
            "Bộ nhớ và khả năng hiển thị ở mức tốt, đáp ứng mượt mà các tác vụ đa nhiệm và giải trí."
        ),
        3: (
            "Phân khúc cao cấp (Mức 3)",
            "Trang bị các thông số kỹ thuật hàng đầu, đặc biệt là dung lượng RAM và màn hình độ phân giải cao."
        )
    }

    title, detail = price_descriptions[prediction]

    st.markdown(f"### Kết quả: {title}")
    st.write(detail)