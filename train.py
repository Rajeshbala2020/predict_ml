from __future__ import annotations

import logging
from pathlib import Path

import joblib
import yfinance as yf
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

from features import FEATURE_COLUMNS, build_features

logger = logging.getLogger(__name__)

DEFAULT_SYMBOL = "AAPL"
MODEL_PATH = Path("model.pkl")


def fetch_training_data(symbol: str = DEFAULT_SYMBOL, period: str = "2y"):
    logger.info("Downloading historical data for %s (%s)", symbol, period)
    data = yf.download(symbol, period=period, interval="1d", auto_adjust=True, progress=False)
    if data.empty:
        raise RuntimeError(f"No historical data fetched for symbol={symbol}.")
    return data


def train_and_save_model(symbol: str = DEFAULT_SYMBOL, model_path: Path = MODEL_PATH) -> dict:
    data = fetch_training_data(symbol=symbol)
    frame = build_features(data)

    if len(frame) < 100:
        raise RuntimeError("Insufficient rows after feature engineering to train reliably.")

    split = int(len(frame) * 0.8)
    train_set = frame.iloc[:split]
    test_set = frame.iloc[split:]

    x_train = train_set[FEATURE_COLUMNS]
    y_train = train_set["target"]
    x_test = test_set[FEATURE_COLUMNS]
    y_test = test_set["target"]

    model = RandomForestClassifier(
        n_estimators=250,
        max_depth=8,
        random_state=42,
        n_jobs=-1,
    )
    model.fit(x_train, y_train)

    preds = model.predict(x_test)
    accuracy = float(accuracy_score(y_test, preds))

    payload = {
        "model": model,
        "feature_columns": FEATURE_COLUMNS,
        "symbol": symbol,
    }
    joblib.dump(payload, model_path)
    logger.info("Model saved to %s with test accuracy %.4f", model_path, accuracy)
    return {"symbol": symbol, "accuracy": accuracy, "rows": len(frame)}


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
    result = train_and_save_model()
    logger.info("Training complete: %s", result)
