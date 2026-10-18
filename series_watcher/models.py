"""Модели предметной области — без имён полей конкретного источника.

Источник расписания отдаёт `ShowRecord` и `EpisodeRecord`, а не свои
словари. Внутри у нас ровно две вещи, за которые отвечает любой источник:
идентификатор и время эфира в UTC. Остальное — украшения, и их отсутствие
ничего не ломает.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime


@dataclass(frozen=True)
class ShowRecord:
    """Сериал в понятных нам терминах."""

    external_id: str
    title: str
    kind: str | None = None  # Scripted, Miniseries, Reality…
    status: str | None = None
    network: str | None = None
    premiered: str | None = None
    ended: str | None = None
    runtime: int | None = None
    language: str | None = None
    genres: tuple[str, ...] = ()
    summary: str | None = None
    url: str | None = None
    image: str | None = None
    rating: float | None = None
    updated: int | None = None
    #: Внешние идентификаторы того же сериала у других баз: tvdb, imdb, tmdb.
    identifiers: dict[str, str] = field(default_factory=dict)


@dataclass(frozen=True)
class EpisodeRecord:
    """Эпизод. Время — только UTC, без исключений: источник обязан его привести."""

    season: int
    number: int
    name: str | None = None
    aired_at: datetime | None = None
    runtime: int | None = None
    summary: str | None = None


@dataclass(frozen=True)
class Credit:
    """Человек из состава сериала: роль и его идентификатор у источника."""

    role: str  # уже семейство: creator | director | writer | producer
    person_id: str
    person_name: str


@dataclass(frozen=True)
class WorkCredit:
    """Другая работа человека: что, в какой роли, у какого сериала."""

    role: str
    person_id: str
    person_name: str
    show_external_id: str
    show_title: str = ""


#: Роли, которые мы вообще слушаем. Остальное (актёры, монтаж, звук) — шум.
INTERESTING_ROLES = frozenset({"creator", "director", "writer", "producer"})


@dataclass(frozen=True)
class Candidate:
    """Сериал-кандидат из рекомендаций: ещё не отслеживается."""

    show: ShowRecord
    score: float = 0.0
    reasons: tuple[str, ...] = ()
    via: str = ""