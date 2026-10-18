"""Сервис: что делает бот и CLI поверх базы и источников.

Пока есть добавление сериала и список отслеживаемого. Опрос расписания и
уведомления добавляются следующим этапом, дедупликация там опирается на
`(show_id, season, number)`.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime

from .config import Config
from .db import Database, Show, TrackedShow, tracked_shows
from .tvmaze import TvMazeClient, TvMazeShow, normalize_name


@dataclass(frozen=True)
class Candidate:
    show: TvMazeShow
    score: float

    @property
    def label(self) -> str:
        meta = ", ".join(part for part in (self.show.kind, self.show.status, self.show.network) if part)
        return f"{self.show.name} ({meta})" if meta else self.show.name


@dataclass(frozen=True)
class AddResult:
    status: str  # added | exists | needs_choice | not_found
    show: Show | None = None
    candidates: tuple[Candidate, ...] = ()
    episodes: int = 0
    message: str = ""


@dataclass
class SeriesWatcher:
    config: Config
    db: Database
    tvmaze: TvMazeClient

    def add_show(self, query: str, *, pick: int | None = None, limit: int = 7) -> AddResult:
        """Найти сериал в TVmaze и завести его в списке.

        Название совпало — добавляем сразу. Иначе возвращаем кандидатов, чтобы
        выбор сделал человек: поиск TVmaze идёт по подстроке, и «office» находит
        и «The Office», и «Offices».
        """
        needle = normalize_name(query)
        if not needle:
            return AddResult(status="not_found", message="Пустой запрос")

        found = self.tvmaze.search_shows(query)
        candidates = tuple(Candidate(show=item.show, score=item.score) for item in found[:limit])
        if not candidates:
            return AddResult(status="not_found", message=f"По запросу «{query}» ничего нет")

        exact = next((item for item in candidates if item.show.normalized_name == needle), None)
        chosen: TvMazeShow | None
        if exact is not None:
            chosen = exact.show
        elif pick is not None:
            if pick < 1 or pick > len(candidates):
                return AddResult(
                    status="needs_choice",
                    candidates=candidates,
                    message=f"Нет варианта №{pick}, их {len(candidates)}",
                )
            chosen = candidates[pick - 1].show
        elif len(candidates) == 1:
            chosen = candidates[0].show
        else:
            return AddResult(
                status="needs_choice",
                candidates=candidates,
                message=f"Под «{query}» нашлось несколько сериалов, уточните выбор",
            )

        return self._store(chosen)

    def add_show_by_id(self, tvmaze_id: int) -> AddResult:
        return self._store(self.tvmaze.get_show(tvmaze_id))

    def refresh_show(self, tvmaze_id: int) -> AddResult:
        """Перечитать сериал и его эпизоды из TVmaze."""
        stored = self.db.get_show_by_tvmaze_id(tvmaze_id)
        if stored is None:
            return AddResult(status="not_found", message=f"Сериал {tvmaze_id} не отслеживается")
        return self._store(self.tvmaze.get_show(tvmaze_id))

    def remove_show(self, tvmaze_id: int) -> bool:
        stored = self.db.get_show_by_tvmaze_id(tvmaze_id)
        if stored is None:
            return False
        self.db.remove_show(stored.id)
        return True

    def list_shows(self, *, now: datetime | None = None) -> list[TrackedShow]:
        return tracked_shows(self.db, now=now)

    def _store(self, show: TvMazeShow) -> AddResult:
        known = self.db.get_show_by_tvmaze_id(show.id) is not None
        stored = self.db.upsert_show(show)
        episodes = self.tvmaze.get_episodes(show.id)
        added, updated = self.db.upsert_episodes(stored.id, episodes)
        message = f"эпизодов в базе: {len(episodes)} (новых {added}, обновлено {updated})"
        status = "exists" if known else "added"
        return AddResult(status=status, show=stored, episodes=len(episodes), message=message)