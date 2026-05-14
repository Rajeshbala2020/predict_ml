from __future__ import annotations

import logging
import os
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, Query
from pydantic import BaseModel, Field

from db import (
    add_watchlist_item,
    delete_watchlist_item,
    get_session,
    init_db,
    list_notifications_for_user,
    list_predictions,
    list_watchlist_for_user,
    list_watchlist_symbols,
    mark_notification_read,
)
from scheduler import create_scheduler, start_scheduler, stop_scheduler
from research_chat_service import ResearchChatRequest, run_research_chat
from service import StockPredictor, predict_and_store

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger(__name__)

predictor = StockPredictor()
scheduler = None
DB_ENABLED = os.getenv("ML_ENABLE_DB", "1").lower() not in {"0", "false", "no"}


class PredictRequest(BaseModel):
    # Yahoo tickers e.g. JPPOWER.NS, 500111.BO; bare NSE/BSE codes also accepted.
    symbol: str = Field(..., min_length=1, max_length=32)


class WatchlistRequest(BaseModel):
    user_id: str = Field(..., min_length=1, max_length=64)
    symbol: str = Field(..., min_length=1, max_length=20)


def run_watchlist_predictions() -> None:
    session = get_session()
    try:
        symbols = list_watchlist_symbols(session)
        if not symbols:
            symbols = [predictor.train_symbol]

        for symbol in symbols:
            try:
                result = predict_and_store(session=session, predictor=predictor, symbol=symbol)
                logger.info("Scheduled prediction stored: %s", result)
            except RuntimeError as exc:
                logger.warning("Scheduler skipped symbol=%s (no market data): %s", symbol, exc)
            except Exception as exc:
                logger.exception("Scheduler failed for symbol=%s: %s", symbol, exc)
    finally:
        session.close()


@asynccontextmanager
async def lifespan(_: FastAPI):
    global scheduler
    predictor.ensure_model()
    if DB_ENABLED:
        init_db()
        scheduler = create_scheduler(run_watchlist_predictions)
        start_scheduler(scheduler)
    else:
        logger.warning("ML service started with DB disabled (ML_ENABLE_DB=0).")
    yield
    if DB_ENABLED and scheduler:
        stop_scheduler(scheduler)


app = FastAPI(title="Stock ML Service", version="1.0.0", lifespan=lifespan)


@app.get("/")
def root():
    return {
        "status": "ok",
        "service": "stock-ml",
        "db_enabled": DB_ENABLED,
        "message": "Use /predict, /predict-and-store, /research-chat, /watchlist, /notifications, /predictions",
    }


@app.post("/predict")
def predict(payload: PredictRequest):
    try:
        return predictor.predict(symbol=payload.symbol.upper())
    except RuntimeError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.post("/research-chat")
def research_chat(payload: ResearchChatRequest):
    """
    Local HF instruct model: draft + refined answer with self-analysis.
    Requires: pip install -r requirements-research-chat.txt
    """
    try:
        return run_research_chat(payload)
    except RuntimeError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except Exception as exc:
        logger.exception("research-chat failed")
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@app.post("/predict-and-store")
def predict_store(payload: PredictRequest):
    if not DB_ENABLED:
        raise HTTPException(status_code=503, detail="DB mode disabled. Use /predict.")
    session = get_session()
    try:
        result = predict_and_store(session=session, predictor=predictor, symbol=payload.symbol.upper())
        return result
    except RuntimeError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    finally:
        session.close()


@app.post("/watchlist")
def add_watchlist(payload: WatchlistRequest):
    if not DB_ENABLED:
        raise HTTPException(status_code=503, detail="DB mode disabled.")
    session = get_session()
    try:
        row = add_watchlist_item(session=session, user_id=payload.user_id, symbol=payload.symbol)
        return {"id": row.id, "user_id": row.user_id, "symbol": row.symbol}
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    finally:
        session.close()


@app.get("/watchlist/{user_id}")
def get_watchlist(user_id: str):
    if not DB_ENABLED:
        raise HTTPException(status_code=503, detail="DB mode disabled.")
    session = get_session()
    try:
        rows = list_watchlist_for_user(session=session, user_id=user_id)
        return [{"id": row.id, "user_id": row.user_id, "symbol": row.symbol} for row in rows]
    finally:
        session.close()


@app.delete("/watchlist/{user_id}/{symbol}")
def remove_watchlist(user_id: str, symbol: str):
    if not DB_ENABLED:
        raise HTTPException(status_code=503, detail="DB mode disabled.")
    session = get_session()
    try:
        deleted = delete_watchlist_item(session=session, user_id=user_id, symbol=symbol)
        return {"deleted": deleted}
    finally:
        session.close()


@app.get("/notifications/{user_id}")
def get_notifications(user_id: str, unread_only: bool = Query(default=False)):
    if not DB_ENABLED:
        raise HTTPException(status_code=503, detail="DB mode disabled.")
    session = get_session()
    try:
        rows = list_notifications_for_user(session=session, user_id=user_id, unread_only=unread_only)
        return [
            {
                "id": row.id,
                "user_id": row.user_id,
                "symbol": row.symbol,
                "message": row.message,
                "is_read": row.is_read,
                "created_at": row.created_at,
            }
            for row in rows
        ]
    finally:
        session.close()


@app.patch("/notifications/{notification_id}/read")
def mark_read(notification_id: int):
    if not DB_ENABLED:
        raise HTTPException(status_code=503, detail="DB mode disabled.")
    session = get_session()
    try:
        row = mark_notification_read(session=session, notification_id=notification_id)
        if row is None:
            raise HTTPException(status_code=404, detail="Notification not found")
        return {
            "id": row.id,
            "user_id": row.user_id,
            "symbol": row.symbol,
            "message": row.message,
            "is_read": row.is_read,
            "created_at": row.created_at,
        }
    finally:
        session.close()


@app.get("/predictions")
def get_predictions(symbol: str | None = Query(default=None), limit: int = Query(default=50, ge=1, le=500)):
    if not DB_ENABLED:
        raise HTTPException(status_code=503, detail="DB mode disabled.")
    session = get_session()
    try:
        rows = list_predictions(session=session, symbol=symbol, limit=limit)
        return [
            {
                "id": row.id,
                "symbol": row.symbol,
                "prediction": row.prediction,
                "confidence": row.confidence,
                "created_at": row.created_at,
            }
            for row in rows
        ]
    finally:
        session.close()
