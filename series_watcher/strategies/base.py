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


class StrategyError(Exception):
    """Рекомендатель не смог отработать: модель не ответила или не настроена.

    Отдельный тип, а не пустой список: «ничего не нашлось» и «сервис не смог»
    — разные вещи, и путать их хуже, чем сказать прямо.
    """


@dataclass(frozen=True)
class Recommendation:
    """Кандидат с объяснением. Объяснение собирается из фактов, не из вдохновения."""

    show: ShowRecord
    score: float
    reasons: tuple[str, ...] = ()
    via: str = ""

    def line(self) -> str:
        parts = [self.show.title]
        if detail := self.line_detail():
            parts.append(detail)
        return " ".join(parts)

    def line_detail(self) -> str:
        """Факты без названия: жанры и рейтинг.

        Отдельно от `line`, потому что в разметке название идёт жирным, а факты —
        моноширинным. Склеивать их в одну строку нельзя.
        """
        parts = []
        if self.show.genres:
            parts.append(f"{', '.join(self.show.genres[:3])}")
        if self.show.rating is not None:
            parts.append(f"рейтинг {self.show.rating}")
        return " · ".join(parts)


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
    #: Клиент модели. Стратегия, которая в нём нужна, требует его настроенным;
    #: та, что не нужна, работает и без него.
    llm: object | None = None
    #: Пожелание словами: «что-то лёгкое на вечер». Для правил непонятно, для
    #: модели — самый прямой способ объяснить, чего человек хочет.
    mood: str | None = None

    @property
    def basis(self) -> list[Show]:
        return [self.focus] if self.focus is not None else self.db.list_shows()


class Strategy(Protocol):
    """Рекомендатор. Имя — для CLI и реестра, `describe` — для `--help`."""

    name: str
    description: str

    def candidates(self, request: Request) -> Sequence[Recommendation]:
        """Подходящие сериалы. Пусто — значит, подходящих нет."""

    def cold_start(self, request: Request, *, limit: int = 3) -> Sequence[Recommendation]:
        """Что предложить, когда список ещё пуст.

        У отдельных рекомендателей тут бывает своя логика, у общих — общий пул.
        """


def airing_shows(request: Request) -> list[ShowRecord]:
    """Сериалы из расписания ближайших дней.

    Расписание без даты — это «что идёт сейчас», пара десятков сериалов: из
    такого пула после отсева почти ничего не остаётся. Поэтому смотрим
    несколько дней вперёд — это несколько запросов, но все ответы кэшируются.
    """
    from datetime import date, timedelta

    shows: dict[str, ShowRecord] = {}
    for offset in range(1, max(request.schedule_days, 1) + 1):
        day = (date.today() + timedelta(days=offset)).isoformat()
        try:
            entries = request.source.get_schedule(day)
        except Exception:  # noqa: BLE001 - расписание не должно ронять подбор
            continue
        for entry in entries:
            shows.setdefault(entry.show.external_id, entry.show)
    return list(shows.values())


def usable_pool(request: Request) -> list[ShowRecord]:
    """Кандидаты, которых вообще можно предлагать.

    Общий отсев для всех стратегий: иначе каждая проверяет одно и то же
    отдельно, и правила расходятся между собой — вчера одно, сегодня другое.
    """
    excluded = {
        f"{request.source.name}:{show.key.external_id}"
        for show in request.db.list_shows()
        if show.key
    }
    excluded |= request.db.rejected_ids()
    return [
        show
        for show in airing_shows(request)
        if f"{request.source.name}:{show.external_id}" not in excluded
        and show.kind == "Scripted"
        and show.status != "Ended"
        and (show.rating is None or show.rating >= request.min_rating)
    ]
