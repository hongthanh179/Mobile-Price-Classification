from pathlib import Path
import matplotlib.patches as patches
import matplotlib.pyplot as plt

BASE_DIR = Path(__file__).resolve().parent
OUTPUT_PATH = BASE_DIR / "reports" / "figures" / "system_pipeline_flow.png"
OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

fig, ax = plt.subplots(figsize=(10, 8.5), facecolor="white")
ax.set_facecolor("white")

steps = [
    (
        "1. GIAO DIỆN NGƯỜI DÙNG (STREAMLIT UI)",
        "Tiếp nhận 20 thông số phần cứng qua Slider & Checkbox\n(RAM, Pin, CPU, Kích thước, Camera, Kết nối...)",
        "#2563EB",
        "#EFF6FF",
    ),
    (
        "2. KỸ NGHỆ ĐẶC TRƯNG (FEATURE ENGINEERING)",
        "Tính toán 2 đặc trưng phái sinh diện tích:\npx_area = px_height * px_width | sc_area = sc_h * sc_w\n→ Hoàn thiện vector 22 thuộc tính",
        "#0D9488",
        "#F0FDFA",
    ),
    (
        "3. CHUẨN HÓA DỮ LIỆU (PREPROCESSING)",
        "Nạp scaler.joblib từ thư mục models/\nÁp dụng biến đổi Z-score: X_scaled = scaler.transform(X)",
        "#D97706",
        "#FFFBEB",
    ),
    (
        "4. SUY LUẬN MÔ HÌNH (MODEL INFERENCE)",
        "Nạp best_svm_model.joblib (Linear SVM, C=1.0)\nThực hiện dự đoán phân khúc: y_pred = model.predict(X_scaled)",
        "#7C3AED",
        "#F5F3FF",
    ),
    (
        "5. TRẢ KẾT QUẢ ĐẦU RA (OUTPUT)",
        "Phân loại thành 1 trong 4 phân khúc (0, 1, 2, 3)\nHiển thị nhãn trực quan và mô tả đặc điểm cấu hình",
        "#059669",
        "#ECFDF5",
    ),
]

y_positions = [7.2, 5.4, 3.6, 1.8, 0.0]
box_width, box_height = 8.4, 1.25

for i, (title, desc, border_col, fill_col) in enumerate(steps):
    y = y_positions[i]
    rect = patches.FancyBboxPatch(
        (-box_width / 2, y),
        box_width,
        box_height,
        boxstyle="round,pad=0.15",
        linewidth=1.8,
        edgecolor=border_col,
        facecolor=fill_col,
    )
    ax.add_patch(rect)

    ax.text(
        0, y + 0.92, title,
        fontsize=11, fontweight="bold", color="#1E293B",
        ha="center", va="center",
    )
    ax.text(
        0, y + 0.38, desc,
        fontsize=9.5, color="#334155",
        ha="center", va="center", linespacing=1.35,
    )

    if i < len(steps) - 1:
        ax.annotate(
            "",
            xy=(0, y_positions[i + 1] + box_height + 0.16),
            xytext=(0, y - 0.16),
            arrowprops=dict(
                arrowstyle="-|>", color="#475569",
                lw=2.0, mutation_scale=15, shrinkA=0, shrinkB=0,
            ),
        )

ax.set_xlim(-5, 5)
ax.set_ylim(-0.4, 8.8)
ax.axis("off")

plt.savefig(OUTPUT_PATH, dpi=300, bbox_inches="tight", facecolor="white", edgecolor="none")
plt.close()
print(f"✓ Đã xuất ảnh thành công tại: {OUTPUT_PATH}")