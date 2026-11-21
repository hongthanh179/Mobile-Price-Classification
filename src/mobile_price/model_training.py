import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import classification_report

try:
    from .utils import get_project_root, save_object
except ImportError:
    from utils import get_project_root, save_object


def train_best_model(
        processed_data_path: str | Path = None,
        model_save_path: str | Path = None,
        scaler_save_path: str | Path = None
):
    root = get_project_root()
    processed_data_path = Path(
        processed_data_path) if processed_data_path else root / "data/processed/train_cleaned.csv"
    model_save_path = Path(model_save_path) if model_save_path else root / "models/best_svm_model.joblib"
    scaler_save_path = Path(scaler_save_path) if scaler_save_path else root / "models/scaler.joblib"

    df = pd.read_csv(processed_data_path)
    X = df.drop(columns=['price_range'])
    y = df['price_range']

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    model = SVC(kernel='linear', C=1.0, random_state=42)
    model.fit(X_train_scaled, y_train)

    y_pred = model.predict(X_test_scaled)
    print("\n[TRAINING] Kết quả đánh giá sơ bộ:")
    print(classification_report(y_test, y_pred))

    save_object(model, model_save_path)
    save_object(scaler, scaler_save_path)
    return model, scaler


if __name__ == '__main__':
    train_best_model()