"""Реестр рекомендаторов.

Подключить новый — значит написать класс с `name`, `description` и
`candidates(request)` и добавить его сюда. Больше нигде править не нужно.
"""

from __future__ import annotations

from ..config import Config
from ..db import Database
from ..schedule import ScheduleSource
from .base import Recommendation, Request, Strategy
from .crew import CrewStrategy
from .model import ModelStrategy

FACTORIES: dict[str, type] = {
    CrewStrategy.name: CrewStrategy,
    ModelStrategy.name: ModelStrategy,
}


def register(name: str, factory: type) -> None:
    FACTORIES[name] = factory


def names() -> list[str]:
    return sorted(FACTORIES)


def get(name: str | None = None) -> Strategy:
    """Рекомендатор по имени; неизвестное имя — понятная ошибка, а не молчание."""
    key = name or "crew"
    factory = FACTORIES.get(key)
    if factory is None:
        raise ValueError(f"неизвестный рекомендатель: {key}. Есть: {', '.join(names())}")
    return factory()


def build(
    config: Config,
    db: Database,
    source: ScheduleSource,
    *,
    name: str | None = None,
    limit: int = 5,
    min_rating: float | None = None,
    focus=None,
    llm=None,
    mood: str | None = None,
) -> list[Recommendation]:
    request = Request(
        config=config,
        db=db,
        source=source,
        limit=limit,
        min_rating=min_rating if min_rating is not None else 6.5,
        focus=focus,
        llm=llm,
        mood=mood,
    )
    return list(get(name).candidates(request))
