from __future__ import annotations

import logging

from apscheduler.schedulers.background import BackgroundScheduler

logger = logging.getLogger(__name__)


def create_scheduler(job_func) -> BackgroundScheduler:
    scheduler = BackgroundScheduler(timezone="UTC")
    scheduler.add_job(job_func, trigger="interval", minutes=5, id="predict-watchlist", replace_existing=True)
    return scheduler


def start_scheduler(scheduler: BackgroundScheduler) -> None:
    if not scheduler.running:
        scheduler.start()
        logger.info("Scheduler started with 5-minute interval.")


def stop_scheduler(scheduler: BackgroundScheduler) -> None:
    if scheduler.running:
        scheduler.shutdown(wait=False)
        logger.info("Scheduler stopped.")
