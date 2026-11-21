import os
import joblib
import yaml
from pathlib import Path
from typing import Any

# Tự động lấy thư mục gốc của đồ án (Mobile-Price-Classification)
PROJECT_ROOT = Path(__file__).resolve().parents[2]

def get_project_root() -> Path:
    return PROJECT_ROOT

def save_object(obj: Any, file_path: str | Path) -> None:
    file_path = Path(file_path)
    file_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(obj, file_path)
    print(f"[UTILS] Đã lưu đối tượng tại: {file_path}")

def load_object(file_path: str | Path) -> Any:
    file_path = Path(file_path)
    if not file_path.exists():
        raise FileNotFoundError(f"[UTILS] Không tìm thấy file: {file_path}")
    obj = joblib.load(file_path)
    print(f"[UTILS] Đã tải đối tượng từ: {file_path}")
    return obj

def read_yaml(path_to_yaml: str | Path) -> dict:
    with open(path_to_yaml, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f) or {}