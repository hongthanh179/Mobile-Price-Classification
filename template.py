import os
from pathlib import Path
import logging

logging.basicConfig(level=logging.INFO, format='[%(asctime)s]: %(message)s:')

project_name = "mobile_price"

# Danh sách toàn bộ các thư mục và file cần tạo theo chuẩn tài liệu
list_of_files = [
    ".github/workflows/.gitkeep",
    # Dữ liệu
    "data/raw/.gitkeep",
    "data/processed/.gitkeep",
    "data/external/.gitkeep",
    # Notebooks nghiên cứu, thử nghiệm
    "notebooks/01_exploratory_data_analysis.ipynb",
    "notebooks/02_model_experiments.ipynb",
    # Mã nguồn chuẩn hóa (src)
    f"src/{project_name}/__init__.py",
    f"src/{project_name}/data_preprocessing.py",
    f"src/{project_name}/model_training.py",
    f"src/{project_name}/model_evaluation.py",
    f"src/{project_name}/utils.py",
    # Lưu trữ mô hình và kết quả
    "models/.gitkeep",
    "reports/figures/.gitkeep",
    "reports/metrics.json",
    # Cấu hình và môi trường
    "config/params.yaml",
    "requirements.txt",
    "setup.py",
    ".gitignore",
    "README.md"
]

for filepath in list_of_files:
    filepath = Path(filepath)
    filedir, filename = os.path.split(filepath)

    if filedir != "":
        os.makedirs(filedir, exist_ok=True)
        logging.info(f"Đã tạo thư mục: {filedir} cho file: {filename}")

    if (not os.path.exists(filepath)) or (os.path.getsize(filepath) == 0):
        with open(filepath, "w") as f:
            pass
        logging.info(f"Đã tạo file trống: {filepath}")
    else:
        logging.info(f"{filename} đã tồn tại!")