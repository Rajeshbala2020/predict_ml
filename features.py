from __future__ import annotations

import pandas as pd


def compute_rsi(series: pd.Series, period: int = 14) -> pd.Series:
    delta = series.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)

    avg_gain = gain.rolling(window=period, min_periods=period).mean()
    avg_loss = loss.rolling(window=period, min_periods=period).mean()

    rs = avg_gain / avg_loss.replace(0, pd.NA)
    rsi = 100 - (100 / (1 + rs))
    return rsi.fillna(50)


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        raise ValueError("Cannot build features from empty dataframe.")

    frame = df.copy()
    frame["return"] = frame["Close"].pct_change()
    frame["sma_10"] = frame["Close"].rolling(window=10).mean()
    frame["rsi_14"] = compute_rsi(frame["Close"], period=14)
    frame["volume"] = frame["Volume"]
    frame["target"] = (frame["Close"].shift(-1) > frame["Close"]).astype(int)
    frame = frame.dropna().copy()
    return frame


FEATURE_COLUMNS = ["return", "sma_10", "rsi_14", "volume"]
