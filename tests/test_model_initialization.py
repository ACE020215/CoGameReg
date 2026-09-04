"""Lightweight tests that do not require the third-party benchmark data."""

import unittest

from cogamereg import (
    CoGameReg_main,
    CoGameReg_validation_data_benchmark,
    XGBRegressor,
    initialize_model_zoo,
)


class PublicInterfaceTests(unittest.TestCase):
    def test_public_entry_points_are_callable(self):
        self.assertTrue(callable(CoGameReg_main))
        self.assertTrue(callable(CoGameReg_validation_data_benchmark))

    @unittest.skipIf(XGBRegressor is None, "xgboost is not installed")
    def test_model_pool_xgboost_parameters_are_forwarded(self):
        config = {
            "xgb": {
                "n_estimators": 37,
                "learning_rate": 0.07,
                "max_depth": 2,
                "subsample": 0.65,
                "colsample_bytree": 0.75,
                "reg_alpha": 0.2,
                "reg_lambda": 2.5,
            }
        }

        model = initialize_model_zoo(config, random_state=42)["xgb"]
        params = model.get_params()

        for key, expected in config["xgb"].items():
            self.assertEqual(params[key], expected)
        self.assertIsNone(params["random_state"])


if __name__ == "__main__":
    unittest.main()
