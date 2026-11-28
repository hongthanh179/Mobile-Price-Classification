import json
from typing import Any, cast

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

try:
    from .utils import get_project_root, load_object
except ImportError:
    from utils import get_project_root, load_object


def evaluate_model(
    data_path: str | Path | None = None,
    model_path: str | Path | None = None,
    scaler_path: str | Path | None = None,
    metrics_path: str | Path | None = None,
    figure_path: str | Path | None = None,
) -> dict[str, Any]:
    root = get_project_root()
    data_path = Path(data_path) if data_path else root / "data/processed/train_cleaned.csv"
    model_path = Path(model_path) if model_path else root / "models/best_svm_model.joblib"
    scaler_path = Path(scaler_path) if scaler_path else root / "models/scaler.joblib"
    metrics_path = Path(metrics_path) if metrics_path else root / "reports/metrics.json"
    figure_path = Path(figure_path) if figure_path else root / "reports/figures/confusion_matrix_evaluated.png"

    model = load_object(model_path)
    scaler = load_object(scaler_path)

    df = pd.read_csv(data_path)
    X = df.drop(columns=['price_range'])
    y = df['price_range']

    _, X_test, _, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    X_test_scaled = scaler.transform(X_test)
    y_pred = model.predict(X_test_scaled)

    acc = accuracy_score(y_test, y_pred)
    report = cast(
        dict[str, Any],
        classification_report(
            y_test, y_pred,
            target_names=['0 (Low)', '1 (Medium)', '2 (High)', '3 (Very High)'],
            output_dict=True
        ),
    )

    print(f"\n[EVALUATION] Độ chính xác tổng thể (Accuracy): {acc * 100:.2f}%")

    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    with open(metrics_path, 'w', encoding='utf-8') as f:
        json.dump(report, f, indent=4, ensure_ascii=False)
    print(f"[EVALUATION] Đã lưu metrics vào: {metrics_path}")

    figure_path.parent.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(6, 5))
    cm = confusion_matrix(y_test, y_pred)
    sns.heatmap(
        cm, annot=True, fmt='d', cmap='Blues',
        xticklabels=['0 (Low)', '1 (Mid)', '2 (High)', '3 (Very High)'],
        yticklabels=['0 (Low)', '1 (Mid)', '2 (High)', '3 (Very High)']
    )
    plt.title('Confusion Matrix - Evaluation Stage', fontweight='bold')
    plt.xlabel('Dự đoán (Predicted)')
    plt.ylabel('Thực tế (Actual)')
    plt.tight_layout()
    plt.savefig(figure_path, dpi=300)
    plt.close()
    print(f"[EVALUATION] Đã lưu Confusion Matrix vào: {figure_path}")

    return report


if __name__ == '__main__':
    evaluate_model()