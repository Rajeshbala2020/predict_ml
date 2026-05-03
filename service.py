from __future__ import annotations

import json
import logging
import re
from contextlib import contextmanager
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path

import joblib
import pandas as pd
import yfinance as yf
from sqlalchemy.orm import Session

from db import get_latest_prediction, get_watchlist_users, insert_notification, insert_prediction
from features import FEATURE_COLUMNS, build_features
from train import MODEL_PATH, train_and_save_model

logger = logging.getLogger(__name__)

CONFIDENCE_THRESHOLD = 70.0

# BSE scrip → Yahoo symbol when `${code}.BO` has no data on Yahoo (same equity on NSE/other).
_BSE_SCRIP_YAHOO_ALIASES: dict[str, tuple[str, ...]] = {
    "532939": ("RPOWER.NS", "RPOWER.BO"),  # Reliance Power Ltd
}


@contextmanager
def _yfinance_log_suppressed():
    """yfinance logs ERROR on every failed download(); suppress during our multi-attempt fetch."""
    lg = logging.getLogger("yfinance")
    prev = lg.level
    lg.setLevel(logging.CRITICAL)
    try:
        yield
    finally:
        lg.setLevel(prev)


def _canonical_symbol_from_input(raw: str) -> str:
    """Bare code for DB/watchlist: strip .NS/.NSE/.BO from input (matches Supabase watchlist keys)."""
    u = raw.strip().upper()
    for suf in (".NS", ".NSE", ".BO"):
        if u.endswith(suf):
            return u[: -len(suf)]
    return u


def _bse_numeric_from_symbol(s: str) -> str | None:
    """Bare 5–7 digit scrip, or 12345.BO / 12345.NS / 12345.NSE."""
    u = s.strip().upper()
    m = re.fullmatch(r"(\d{5,7})\.(?:BO|NS|NSE)", u)
    if m:
        return m.group(1)
    if re.fullmatch(r"\d{5,7}", u):
        return u
    return None


def _yfinance_ticker_candidates(symbol: str) -> list[str]:
    """Yahoo download order. Numeric scrip: known aliases first (e.g. Reliance Power 532939 → RPOWER.NS)."""
    s = symbol.strip().upper()
    out: list[str] = []

    def add(x: str) -> None:
        if x and x not in out:
            out.append(x)

    code = _bse_numeric_from_symbol(s)
    if code:
        for alias in _BSE_SCRIP_YAHOO_ALIASES.get(code, ()):
            add(alias)
        # User / DB may pass 532939.BO explicitly — still try it after aliases.
        if "." in s:
            add(s)
        add(f"{code}.BO")
        add(f"{code}.NS")
        add(code)
        return out

    if "." in s:
        add(s)
        return out
    add(s)
    add(f"{s}.NS")
    add(f"{s}.BO")
    return out


def _df_usable(df: pd.DataFrame | None) -> bool:
    return df is not None and isinstance(df, pd.DataFrame) and not df.empty and len(df) >= 12


def _normalize_download_df(raw: pd.DataFrame) -> pd.DataFrame:
    """Flatten yfinance.download multi-index columns when present."""
    if not isinstance(raw.columns, pd.MultiIndex) or raw.columns.nlevels < 2:
        return raw
    raw = raw.copy()
    try:
        return raw.droplevel(1, axis=1)
    except Exception:
        try:
            return raw.droplevel(0, axis=1)
        except Exception:
            raw.columns = ["-".join(str(p) for p in c) if isinstance(c, tuple) else str(c) for c in raw.columns]
            return raw


def _yahoo_chart_ohlcv(ticker: str) -> pd.DataFrame | None:
    """
    Yahoo Finance chart API (v8) — often returns bars when yfinance.download/history fails
    (e.g. BSE tickers with missing tz metadata or 'delisted' false positives).
    """
    safe = urllib.parse.quote(ticker, safe="-._~")
    for range_param in ("2y", "5y", "10y", "max"):
        for interval in ("1d", "1wk"):
            url = f"https://query1.finance.yahoo.com/v8/finance/chart/{safe}?interval={interval}&range={range_param}"
            req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0 (compatible; stock-ml/1.0)"})
            try:
                with urllib.request.urlopen(req, timeout=20) as resp:
                    payload = json.loads(resp.read().decode("utf-8"))
            except (urllib.error.URLError, TimeoutError, json.JSONDecodeError, ValueError, OSError) as exc:
                logger.debug("chart API HTTP failed %s: %s", ticker, exc)
                continue

            chart = payload.get("chart") or {}
            if chart.get("error"):
                continue
            results = chart.get("result")
            if not results:
                continue
            res = results[0]
            ts = res.get("timestamp")
            quotes = (res.get("indicators") or {}).get("quote") or []
            if not ts or not quotes:
                continue
            q0 = quotes[0]
            closes = q0.get("close") or []
            if not closes or all(x is None for x in closes):
                continue

            n = len(ts)
            o = (q0.get("open") or [None] * n)[:n]
            h = (q0.get("high") or [None] * n)[:n]
            lows = (q0.get("low") or [None] * n)[:n]
            c = (q0.get("close") or [None] * n)[:n]
            v = (q0.get("volume") or [0] * n)[:n]

            idx = pd.to_datetime(ts, unit="s")
            df = pd.DataFrame({"Open": o, "High": h, "Low": lows, "Close": c, "Volume": v}, index=idx)
            df["Volume"] = pd.to_numeric(df["Volume"], errors="coerce").fillna(0.0)
            for col in ("Open", "High", "Low", "Close"):
                df[col] = pd.to_numeric(df[col], errors="coerce")
            df = df.dropna(subset=["Close"])
            if _df_usable(df):
                logger.info("Loaded OHLCV via Yahoo chart API for %s", ticker)
                return df
    return None


def _yf_download_series(ticker: str) -> pd.DataFrame | None:
    """
    Load prices: Yahoo chart API first (most reliable for awkward BSE tickers), then
    Ticker.history(), then yfinance.download().
    """
    chart_df = _yahoo_chart_ohlcv(ticker)
    if chart_df is not None:
        return chart_df

    with _yfinance_log_suppressed():
        tk = yf.Ticker(ticker)
        periods = ("max", "2y", "1y", "6mo")
        intervals = ("1d", "1wk")

        for period in periods:
            for interval in intervals:
                for use_repair in (True, False):
                    kw: dict = {
                        "period": period,
                        "interval": interval,
                        "auto_adjust": True,
                        "actions": False,
                        "prepost": False,
                    }
                    if use_repair:
                        kw["repair"] = True
                    h = None
                    while kw:
                        try:
                            h = tk.history(**kw)
                            break
                        except TypeError:
                            if "repair" in kw:
                                del kw["repair"]
                                continue
                            if "prepost" in kw:
                                del kw["prepost"]
                                continue
                            logger.debug("history unsupported kwargs for %s: %s", ticker, kw)
                            break
                        except Exception as exc:
                            logger.debug("history failed %s %s: %s", ticker, kw, exc)
                            break
                    if h is not None and _df_usable(h):
                        return h

        for period in periods:
            try:
                raw = yf.download(
                    ticker,
                    period=period,
                    interval="1d",
                    auto_adjust=True,
                    progress=False,
                    threads=False,
                )
            except Exception as exc:
                logger.debug("download failed %s %s: %s", ticker, period, exc)
                continue
            if raw is None or raw.empty:
                continue
            raw = _normalize_download_df(raw)
            if _df_usable(raw):
                return raw
    return None


def _yahoo_search_first_indian_ticker(query: str) -> str | None:
    """Match Next.js resolve fallback: Yahoo search for first .NS / .BO hit."""
    q = urllib.parse.quote(query.strip())
    url = f"https://query2.finance.yahoo.com/v1/finance/search?q={q}&quotesCount=12"
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0 (compatible; stock-ml/1.0)"})
    try:
        with urllib.request.urlopen(req, timeout=12) as resp:
            payload = json.loads(resp.read().decode("utf-8"))
    except (urllib.error.URLError, TimeoutError, json.JSONDecodeError, ValueError, OSError) as exc:
        logger.debug("Yahoo search failed for q=%s: %s", query, exc)
        return None
    for row in payload.get("quotes") or []:
        sym = str(row.get("symbol") or "")
        if sym.endswith(".NS") or sym.endswith(".BO"):
            return sym
    return None


class StockPredictor:
    def __init__(self, model_path: Path = MODEL_PATH):
        self.model_path = model_path
        self.model = None
        self.feature_columns = FEATURE_COLUMNS
        self.train_symbol = "AAPL"

    def ensure_model(self) -> None:
        if self.model_path.exists():
            payload = joblib.load(self.model_path)
            self.model = payload["model"]
            self.feature_columns = payload.get("feature_columns", FEATURE_COLUMNS)
            self.train_symbol = payload.get("symbol", "AAPL")
            return

        logger.info("Model file not found. Triggering training.")
        train_and_save_model(symbol=self.train_symbol, model_path=self.model_path)
        payload = joblib.load(self.model_path)
        self.model = payload["model"]
        self.feature_columns = payload.get("feature_columns", FEATURE_COLUMNS)

    def predict(self, symbol: str) -> dict:
        self.ensure_model()
        symbol_out = _canonical_symbol_from_input(symbol)
        candidates = _yfinance_ticker_candidates(symbol)

        search_hit = _yahoo_search_first_indian_ticker(symbol_out)
        if search_hit and search_hit not in candidates:
            candidates = [*candidates, search_hit]

        data: pd.DataFrame | None = None
        for ticker in candidates:
            raw = _yf_download_series(ticker)
            if raw is not None and not raw.empty:
                data = raw
                logger.info("Using Yahoo ticker %s for input=%s (stored as %s)", ticker, symbol, symbol_out)
                break

        if data is None or data.empty:
            tried = ", ".join(candidates[:10]) + ("…" if len(candidates) > 10 else "")
            raise RuntimeError(
                f"No price history on Yahoo for input={symbol!r} (tried: {tried}). "
                "Yahoo reports many BSE numeric codes as delisted or moved—open the symbol on finance.yahoo.com, "
                "then set `indian_stocks.yf_symbol` to the exact ticker shown there, or remove the symbol from the watchlist."
            )

        features_df = build_features(data)
        if features_df.empty:
            raise RuntimeError(f"Insufficient data to build features for symbol={symbol}.")

        latest_features = features_df.iloc[[-1]][self.feature_columns]
        raw_prediction = int(self.model.predict(latest_features)[0])
        confidence = float(self.model.predict_proba(latest_features)[0][raw_prediction] * 100.0)
        label = "UP" if raw_prediction == 1 else "DOWN"
        return {"symbol": symbol_out, "prediction": label, "confidence": round(confidence, 2)}


def should_notify(previous_prediction: str | None, current_prediction: str, confidence: float) -> bool:
    changed = previous_prediction is not None and previous_prediction != current_prediction
    high_confidence = confidence > CONFIDENCE_THRESHOLD
    return changed or high_confidence


def create_notifications_if_needed(session: Session, symbol: str, prediction: str, confidence: float) -> int:
    previous = get_latest_prediction(session, symbol)
    previous_prediction = previous.prediction if previous else None
    if not should_notify(previous_prediction=previous_prediction, current_prediction=prediction, confidence=confidence):
        return 0

    users = get_watchlist_users(session, symbol)
    if not users:
        return 0

    message = (
        f"{symbol} prediction is {prediction} with {confidence:.2f}% confidence."
        f"{' (Direction changed)' if previous_prediction and previous_prediction != prediction else ''}"
    )

    for user_id in users:
        insert_notification(session=session, user_id=user_id, symbol=symbol, message=message)
    return len(users)


def predict_and_store(session: Session, predictor: StockPredictor, symbol: str) -> dict:
    result = predictor.predict(symbol=symbol)
    create_notifications_if_needed(
        session=session,
        symbol=result["symbol"],
        prediction=result["prediction"],
        confidence=result["confidence"],
    )
    insert_prediction(
        session=session,
        symbol=result["symbol"],
        prediction=result["prediction"],
        confidence=result["confidence"],
    )
    return result
