import unittest
from unittest.mock import patch

from pydantic import ValidationError

from src.mobile_price import api


class StubScaler:
    def transform(self, features):
        self.features = features.copy()
        return features.to_numpy()


class StubModel:
    def predict(self, features):
        return [2]


def valid_specifications():
    return {
        "battery_power": 1200,
        "blue": 1,
        "clock_speed": 1.8,
        "dual_sim": 1,
        "fc": 5,
        "four_g": 1,
        "int_memory": 32,
        "m_dep": 0.5,
        "mobile_wt": 140,
        "n_cores": 6,
        "pc": 10,
        "px_height": 800,
        "px_width": 1200,
        "ram": 2048,
        "sc_h": 14,
        "sc_w": 7,
        "talk_time": 10,
        "three_g": 1,
        "touch_screen": 1,
        "wifi": 1,
    }


class PredictionApiTests(unittest.TestCase):
    def test_feature_frame_adds_derived_fields_in_model_order(self):
        specifications = api.MobileSpecifications(**valid_specifications())

        features = api.build_feature_frame(specifications)

        self.assertEqual(list(features.columns), api.FEATURE_COLUMNS)
        self.assertEqual(features.loc[0, "px_area"], 960000)
        self.assertEqual(features.loc[0, "sc_area"], 98)

    def test_prediction_uses_artifacts_and_returns_segment(self):
        scaler = StubScaler()
        model = StubModel()
        specifications = api.MobileSpecifications(**valid_specifications())

        with patch.object(
            api, "load_artifacts", return_value=(model, scaler)
        ) as load:
            response = api.predict_price_segment(specifications)

        self.assertEqual(response.prediction, 2)
        self.assertEqual(response.model, "Linear SVM")
        load.assert_called_once_with("svm")
        self.assertEqual(list(scaler.features.columns), api.FEATURE_COLUMNS)

    def test_prediction_can_select_random_forest(self):
        scaler = StubScaler()
        model = StubModel()
        specifications = api.MobileSpecifications(**valid_specifications())

        with patch.object(api, "load_artifacts", return_value=(model, scaler)) as load:
            response = api.predict_price_segment(
                specifications,
                model="random_forest",
            )

        load.assert_called_once_with("random_forest")
        self.assertEqual(response.model, "Random Forest")
        self.assertEqual(response.prediction, 2)

    def test_rejects_non_binary_connection_values(self):
        payload = valid_specifications()
        payload["wifi"] = 2

        with self.assertRaises(ValidationError):
            api.MobileSpecifications(**payload)


if __name__ == "__main__":
    unittest.main()
