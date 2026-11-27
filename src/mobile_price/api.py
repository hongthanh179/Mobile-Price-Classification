from functools import lru_cache
from typing import Any, Literal

import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from .utils import get_project_root


FEATURE_COLUMNS = [
    "battery_power",
    "blue",
    "clock_speed",
    "dual_sim",
    "fc",
    "four_g",
    "int_memory",
    "m_dep",
    "mobile_wt",
    "n_cores",
    "pc",
    "px_height",
    "px_width",
    "ram",
    "sc_h",
    "sc_w",
    "talk_time",
    "three_g",
    "touch_screen",
    "wifi",
    "px_area",
    "sc_area",
]

PRICE_DESCRIPTIONS = {
    0: (
        "Phân khúc phổ thông (Mức 0)",
        "Cấu hình ở mức cơ bản, tối ưu cho nhu cầu liên lạc thông thường và các tác vụ nhẹ.",
    ),
    1: (
        "Phân khúc tầm trung (Mức 1)",
        "Phần cứng cân đối giữa hiệu năng xử lý, thời lượng pin và giá thành.",
    ),
    2: (
        "Phân khúc cận cao cấp (Mức 2)",
        "Bộ nhớ và khả năng hiển thị ở mức tốt, đáp ứng mượt mà các tác vụ đa nhiệm và giải trí.",
    ),
    3: (
        "Phân khúc cao cấp (Mức 3)",
        "Trang bị các thông số kỹ thuật hàng đầu, đặc biệt là dung lượng RAM và màn hình độ phân giải cao.",
    ),
}


class MobileSpecifications(BaseModel):
    battery_power: int = Field(ge=500, le=2000)
    blue: Literal[0, 1]
    clock_speed: float = Field(gt=0, le=3)
    dual_sim: Literal[0, 1]
    fc: int = Field(ge=0, le=20)
    four_g: Literal[0, 1]
    int_memory: int = Field(ge=2, le=64)
    m_dep: float = Field(ge=0.1, le=1)
    mobile_wt: int = Field(ge=80, le=200)
    n_cores: Literal[1, 2, 4, 6, 8]
    pc: int = Field(ge=0, le=20)
    px_height: int = Field(ge=100, le=2000)
    px_width: int = Field(ge=500, le=2000)
    ram: int = Field(ge=256, le=4000)
    sc_h: int = Field(ge=5, le=20)
    sc_w: int = Field(ge=1, le=18)
    talk_time: int = Field(ge=2, le=20)
    three_g: Literal[0, 1]
    touch_screen: Literal[0, 1]
    wifi: Literal[0, 1]


class PredictionResponse(BaseModel):
    prediction: int
    segment: str
    description: str
    model: str


@lru_cache(maxsize=1)
def load_artifacts() -> tuple[Any, Any]:
    root = get_project_root()
    model_path = root / "models" / "best_svm_model.joblib"
    scaler_path = root / "models" / "scaler.joblib"
    missing = [path for path in (model_path, scaler_path) if not path.exists()]
    if missing:
        missing_paths = ", ".join(str(path) for path in missing)
        raise FileNotFoundError(f"Required model artifact(s) not found: {missing_paths}")
    return joblib.load(model_path), joblib.load(scaler_path)


def build_feature_frame(specifications: MobileSpecifications) -> pd.DataFrame:
    values = specifications.model_dump()
    values["px_area"] = values["px_height"] * values["px_width"]
    values["sc_area"] = values["sc_h"] * values["sc_w"]
    return pd.DataFrame(
        [[values[column] for column in FEATURE_COLUMNS]],
        columns=FEATURE_COLUMNS,
    )


app = FastAPI(
    title="Mobile Price Classification API",
    description="Predicts a mobile phone price segment using the project's Linear SVM.",
    version="1.0.0",
)


@app.post("/predict", response_model=PredictionResponse)
def predict_price_segment(
    specifications: MobileSpecifications,
) -> PredictionResponse:
    try:
        model, scaler = load_artifacts()
    except FileNotFoundError as error:
        raise HTTPException(status_code=503, detail=str(error)) from error

    feature_frame = build_feature_frame(specifications)
    scaled_features = scaler.transform(feature_frame)
    prediction = int(model.predict(scaled_features)[0])
    segment, description = PRICE_DESCRIPTIONS[prediction]
    return PredictionResponse(
        prediction=prediction,
        segment=segment,
        description=description,
        model="Linear SVM",
    )
