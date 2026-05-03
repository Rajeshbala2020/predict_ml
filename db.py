from __future__ import annotations

import os
from datetime import datetime

from sqlalchemy import Boolean, DateTime, Float, Integer, String, create_engine, desc, select
from sqlalchemy.orm import DeclarativeBase, Mapped, Session, mapped_column, sessionmaker


DATABASE_URL = os.getenv("DATABASE_URL", "postgresql+psycopg2://postgres:postgres@localhost:5432/ml_service")

engine = create_engine(
    DATABASE_URL,
    pool_pre_ping=True,
    pool_size=5,
    max_overflow=10,
    future=True,
)
SessionLocal = sessionmaker(bind=engine, autoflush=False, autocommit=False, class_=Session)


class Base(DeclarativeBase):
    pass


class Prediction(Base):
    __tablename__ = "predictions"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, index=True)
    symbol: Mapped[str] = mapped_column(String(20), index=True)
    prediction: Mapped[str] = mapped_column(String(8))
    confidence: Mapped[float] = mapped_column(Float)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow, index=True)


class Watchlist(Base):
    __tablename__ = "watchlist"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, index=True)
    user_id: Mapped[str] = mapped_column(String(64), index=True)
    symbol: Mapped[str] = mapped_column(String(20), index=True)


class Notification(Base):
    __tablename__ = "notifications"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, index=True)
    user_id: Mapped[str] = mapped_column(String(64), index=True)
    symbol: Mapped[str] = mapped_column(String(20), index=True)
    message: Mapped[str] = mapped_column(String(500))
    is_read: Mapped[bool] = mapped_column(Boolean, default=False)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow, index=True)


def init_db() -> None:
    Base.metadata.create_all(bind=engine)


def get_session() -> Session:
    return SessionLocal()


def insert_prediction(session: Session, symbol: str, prediction: str, confidence: float) -> Prediction:
    row = Prediction(symbol=symbol, prediction=prediction, confidence=confidence)
    session.add(row)
    session.commit()
    session.refresh(row)
    return row


def get_latest_prediction(session: Session, symbol: str) -> Prediction | None:
    stmt = select(Prediction).where(Prediction.symbol == symbol).order_by(desc(Prediction.created_at)).limit(1)
    return session.scalars(stmt).first()


def get_watchlist_users(session: Session, symbol: str) -> list[str]:
    stmt = select(Watchlist.user_id).where(Watchlist.symbol == symbol)
    return list(session.scalars(stmt).all())


def insert_notification(session: Session, user_id: str, symbol: str, message: str) -> Notification:
    row = Notification(user_id=user_id, symbol=symbol, message=message, is_read=False)
    session.add(row)
    session.commit()
    session.refresh(row)
    return row


def list_watchlist_symbols(session: Session) -> list[str]:
    stmt = select(Watchlist.symbol).distinct()
    symbols = session.scalars(stmt).all()
    return sorted({symbol.upper() for symbol in symbols})


def add_watchlist_item(session: Session, user_id: str, symbol: str) -> Watchlist:
    normalized = symbol.upper()
    existing = session.scalars(
        select(Watchlist).where(Watchlist.user_id == user_id, Watchlist.symbol == normalized).limit(1)
    ).first()
    if existing:
        return existing

    row = Watchlist(user_id=user_id, symbol=normalized)
    session.add(row)
    session.commit()
    session.refresh(row)
    return row


def list_watchlist_for_user(session: Session, user_id: str) -> list[Watchlist]:
    stmt = select(Watchlist).where(Watchlist.user_id == user_id).order_by(Watchlist.symbol.asc())
    return list(session.scalars(stmt).all())


def delete_watchlist_item(session: Session, user_id: str, symbol: str) -> int:
    normalized = symbol.upper()
    rows = session.scalars(select(Watchlist).where(Watchlist.user_id == user_id, Watchlist.symbol == normalized)).all()
    deleted_count = len(rows)
    for row in rows:
        session.delete(row)
    if deleted_count:
        session.commit()
    return deleted_count


def list_notifications_for_user(session: Session, user_id: str, unread_only: bool = False) -> list[Notification]:
    stmt = select(Notification).where(Notification.user_id == user_id).order_by(Notification.created_at.desc())
    if unread_only:
        stmt = stmt.where(Notification.is_read.is_(False))
    return list(session.scalars(stmt).all())


def mark_notification_read(session: Session, notification_id: int) -> Notification | None:
    row = session.get(Notification, notification_id)
    if row is None:
        return None
    row.is_read = True
    session.commit()
    session.refresh(row)
    return row


def list_predictions(session: Session, symbol: str | None = None, limit: int = 50) -> list[Prediction]:
    stmt = select(Prediction).order_by(Prediction.created_at.desc()).limit(limit)
    if symbol:
        stmt = stmt.where(Prediction.symbol == symbol.upper())
    return list(session.scalars(stmt).all())
