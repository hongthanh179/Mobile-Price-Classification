import pandas as pd
from pathlib import Path

try:
    from .utils import get_project_root
except ImportError:
    from utils import get_project_root


def clean_data(
    raw_data_path: str | Path | None = None,
    output_path: str | Path | None = None,
) -> pd.DataFrame:
    root = get_project_root()
    raw_data_path = Path(raw_data_path) if raw_data_path else root / "data/raw/train.csv"
    output_path = Path(output_path) if output_path else root / "data/processed/train_cleaned.csv"

    df = pd.read_csv(raw_data_path)

    # Xử lý giá trị 0 phi lý bằng median
    sc_w_median = df.loc[df['sc_w'] > 0, 'sc_w'].median()
    px_h_median = df.loc[df['px_height'] > 0, 'px_height'].median()

    df['sc_w'] = df['sc_w'].replace(0, sc_w_median)
    df['px_height'] = df['px_height'].replace(0, px_h_median)

    # Feature Engineering
    df['px_area'] = df['px_height'] * df['px_width']
    df['sc_area'] = df['sc_h'] * df['sc_w']

    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)
    print(f"[PREPROCESSING] Dữ liệu sạch đã lưu tại: {output_path}")
    return df


if __name__ == '__main__':
    clean_data()