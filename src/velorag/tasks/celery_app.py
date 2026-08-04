from __future__ import annotations

from celery import Celery

from velorag.core.settings import get_settings

settings = get_settings()

celery_app = Celery(
    "velorag",
    broker=settings.celery.broker_url,
    backend=settings.celery.result_backend,
)

celery_app.conf.update(
    task_serializer=settings.celery.task_serializer,
    result_serializer=settings.celery.result_serializer,
    accept_content=settings.celery.accept_content,
    timezone=settings.celery.timezone,
    enable_utc=settings.celery.enable_utc,
    task_track_started=True,
    task_acks_late=True,
    worker_prefetch_multiplier=1,
)
