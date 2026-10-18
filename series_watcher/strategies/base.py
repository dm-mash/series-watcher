"""Договорённости между рекомендатором и тем, кто им пользуется.

Рекомендатор — сменяемая деталь: интерфейс один, реализации разные (состав
сериалов, запрос к модели, внешний поиск). Сервис знает только `Strategy`.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, Sequence

from ..config import Config
from ..db import Database, Show
from ..models import ShowRecord
from ..schedule import ScheduleSource


@dataclass(frozen=True)
class Recommendation:
    """Кандидат с объяснением. Объяснение собирается из фактов, не из вдохновения."""

    show: ShowRecord
    score: float
    reasons: tuple[str, ...] = ()
    via: str = ""

    def line(self) -> str:
        parts = [self.show.title]
        if self.show.genres:
            parts.append(f"({', '.join(self.show.genres[:3])})")
        if self.show.rating is not None:
            parts.append(f"рейтинг {self.show.rating}")
        return " ".join(parts)


@dataclass(frozen=True)
class Request:
    """Всё, что нужно стратегии, чтобы что-то предложить."""

    config: Config
    db: Database
    source: ScheduleSource
    limit: int = 5
    min_rating: float = 6.5
    #: Сколько дней расписания смотреть: пул кандидатов — это то, что идёт в эфире.
    schedule_days: int = 3
    #: Сериал, названный в запросе: «похожие на X» — сигналы только по нему.
    focus: Show | None = None

    @property
    def basis(self) -> list[Show]:
        return [self.focus] if self.focus is not None else self.db.list_shows()


class Strategy(Protocol):
    """Рекомендатор. Имя — для CLI и реестра, `describe` — для `--help`."""

    name: str
    description: str

    def candidates(self, request: Request) -> Sequence[Recommendation]:
        """Подходящие сериалы. Пусто — значит, подходящих нет."""
