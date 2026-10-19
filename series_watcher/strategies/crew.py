"""Подбор по авторству: кто работал над вашими сериалами и над чем ещё.

Сигнал один и самый честный — общий создатель, режиссёр или сценарист. Роли
приходят уже семействами, вес задан в `schedule.ROLE_WEIGHTS`. Жанры и рейтинг
идут страховкой, когда авторство ничего не дало.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, replace
from datetime import date, timedelta
from typing import Sequence

from ..db import Show, normalize_name
from ..models import ShowRecord
from ..schedule import ROLE_LABELS, ROLE_WEIGHTS, ScheduleError
from .base import Recommendation, Request, usable_pool

SCRIPTED = "Scripted"
#: Статус, в котором сериал уже не выйдет. Трекер ждёт следующего эпизода, поэтому
#: законченные в рекомендации не попадают: новая серия — их и так можно начать с
#: любой точки. Неизвестный статус (нет данных) заканченным не считаем.
ENDED = "Ended"
#: Сколько работ наших авторов догружать из источника: каждый — запрос.
#: Кэш идёт по id, поэтому повторный запуск бесплатный.
DEFAULT_FETCHES = 8
#: Сколько дней расписания смотреть: три дня — это три запроса к источнику.
DEFAULT_SCHEDULE_DAYS = 3
#: Порог для холодного старта: вкуса ещё нет, поэтому берём заметно ниже.
COLD_START_MIN_RATING = 5.5


@dataclass(frozen=True)
class Person:
    id: str
    name: str
    roles: set[str]


def _by_weight(credits: dict[str, set[str]]) -> list[tuple[str, set[str]]]:
    """Люди по убыванию своей лучшей роли: создатель выше режиссёра."""
    return sorted(
        credits.items(),
        key=lambda item: (
            -max((ROLE_WEIGHTS.get(role, 0.0) for role in item[1]), default=0.0),
            item[0],
        ),
    )


def _tracked(record, show: Show) -> ShowRecord:
    """Карточка отслеживаемого сериала — для случая, когда источник молчит."""
    return ShowRecord(
        external_id=str(show.key.external_id) if show.key else "",
        title=show.name,
        kind=show.kind,
        status=show.status,
        network=show.network,
        premiered=show.premiered,
        ended=show.ended,
        runtime=show.runtime,
        language=show.language,
        genres=tuple(show.genres),
        summary=show.summary,
        url=show.url,
        image=show.image,
        rating=show.rating,
    )


class CrewStrategy:
    """Рекомендации по общему авторству, жанрам и тому, что сейчас в эфире."""

    name = "crew"
    description = "по создателям, режиссёрам и сценаристам, затем по жанрам"

    def candidates(self, request: Request) -> Sequence[Recommendation]:
        tracked = request.db.list_shows()
        if not tracked:
            return []

        basis = request.basis
        # Ключ везде один и тот же: «источник:идентификатор».
        excluded = {
            f"{request.source.name}:{show.key.external_id}"
            for show in tracked
            if show.key
        }
        excluded |= request.db.rejected_ids()
        genres = Counter(genre for show in basis for genre in show.genres)
        authored = self._authored(request, basis)

        scored: dict[str, Recommendation] = {}
        spare: list[ShowRecord] = []
        for record in self._pool(request, basis, authored):
            if f"{request.source.name}:{record.external_id}" in excluded:
                continue
            if record.kind != SCRIPTED:
                continue
            if record.status == ENDED:
                continue
            if record.rating is not None and record.rating < request.min_rating:
                continue

            reasons: list[str] = []
            score = 0.0
            credits = _by_weight(authored.get(record.external_id) or {})
            for name, roles in credits:
                for role in sorted(roles, key=lambda item: -ROLE_WEIGHTS.get(item, 0.0)):
                    score += ROLE_WEIGHTS.get(role, 0.0)
                    reasons.append(f"{ROLE_LABELS.get(role, role)}: {name}")

            shared = sorted(
                (genre for genre in record.genres if genres.get(genre)),
                key=lambda genre: (-genres[genre], genre),
            )
            if shared:
                score += len(shared)
                reasons.append("жанры: " + ", ".join(shared))

            if not reasons:
                # Общего с вашим списком ничего нет, но сериал хороший и идёт
                # в эфире. Пригодится, если related-кандидатов не хватило на просьбу.
                if record.rating is not None:
                    spare.append(record)
                continue
            scored[record.external_id] = Recommendation(
                show=record,
                score=score,
                reasons=tuple(dict.fromkeys(reasons)),
                via="авторство" if credits else "жанры",
            )

        ranked = sorted(
            scored.values(), key=lambda item: (-item.score, -(item.show.rating or 0))
        )
        picked = list(ranked[: request.limit])
        if request.focus is None:
            picked += self._top_up(spare, picked, request.limit)
        return picked

    @staticmethod
    def _top_up(
        spare: Sequence[ShowRecord], picked: Sequence[Recommendation], limit: int
    ) -> list[Recommendation]:
        """Добор по рейтингу, когда похожих не набралось.

        Добавленные сериалы честно помечены: общего с вашим списком у них ничего
        нет, и путать их с «похожими» нельзя. При явном фокусе («похожее на X»)
        добора нет вовсе: человек спросил про X, а ему прислали бы чужое.
        """
        taken = {item.show.external_id for item in picked}
        room = limit - len(picked)
        if room <= 0:
            return []
        by_rating = sorted(spare, key=lambda item: (-(item.rating or 0.0), item.title))
        return [
            Recommendation(
                show=record,
                score=record.rating or 0.0,
                reasons=("добор: идёт в эфире и хорошо оценён",),
                via="добор",
            )
            for record in by_rating[:room]
            if record.external_id not in taken
        ]

    def cold_start(self, request: Request, *, limit: int = 3) -> Sequence[Recommendation]:
        """Первый запуск: вкуса ещё нет, предлагаем то, что сейчас в эфире."""
        items: list[Recommendation] = []
        for record in usable_pool(replace(request, min_rating=COLD_START_MIN_RATING)):
            reasons = []
            if record.genres:
                reasons.append("жанры: " + ", ".join(record.genres[:3]))
            if record.network:
                reasons.append(f"канал {record.network}")
            items.append(
                Recommendation(
                    show=record,
                    score=record.rating or 0.0,
                    reasons=tuple(reasons) or ("идёт в эфире",),
                    via="расписание",
                )
            )
        items.sort(key=lambda item: (-item.score, item.show.title))
        return items[:limit]

    # --- сигналы ---------------------------------------------------------

    def _people(self, request: Request, tracked: Sequence[Show]) -> dict[str, Person]:
        people: dict[str, Person] = {}
        for show in tracked:
            if show.key is None:
                continue
            try:
                credits = request.source.get_credits(show.key.external_id)
            except Exception:  # noqa: BLE001 - источник не должен ронять подбор
                continue
            for entry in credits:
                if not entry.person_name:
                    continue
                current = people.setdefault(
                    entry.person_id, Person(id=entry.person_id, name=entry.person_name, roles=set())
                )
                current.roles.add(entry.role)
        return people

    def _authored(self, request: Request, tracked: Sequence[Show]) -> dict[str, dict[str, set[str]]]:
        """Другие работы этих людей: id сериала -> имя -> роли."""
        result: dict[str, dict[str, set[str]]] = {}
        for person_id, person in self._people(request, tracked).items():
            try:
                works = request.source.get_works(person_id)
            except Exception:  # noqa: BLE001
                continue
            for work in works:
                roles = result.setdefault(work.show_external_id, {}).setdefault(person.name, set())
                roles.add(work.role)
        return result

    def _pool(self, request: Request, basis, authored: dict) -> list[ShowRecord]:
        """Всё, что идёт в эфире, плюс работы наших авторов.

        Расписание не обрезается: иначе в первых восьми по рейтингу окажутся
        кулинарные шоу, а сценаристы и режиссёры ваших сериалов до отсева не
        доживут. Лимит здесь — только на догрузку по сети, а не на размер пула.
        """
        shows: dict[str, ShowRecord] = {}
        for record in self._airing(request):
            shows.setdefault(record.external_id, record)
        wanted = list(authored) + [
            str(show.key.external_id) for show in basis if show.key is not None
        ]
        fetched = 0
        for external_id in wanted:
            if external_id in shows or fetched >= DEFAULT_FETCHES:
                continue
            fetched += 1
            try:
                shows[external_id] = request.source.get_show(external_id)
            except Exception:  # noqa: BLE001
                continue
        for show in basis:
            if show.key is not None:
                shows.setdefault(str(show.key.external_id), _tracked(None, show))
        return sorted(shows.values(), key=lambda item: (-(item.rating or 0.0), item.title))

    def _airing(self, request: Request) -> list[ShowRecord]:
        """Сериалы из расписания ближайших дней.

        Расписание без даты — это «что идёт сейчас», пара десятков сериалов: из
        такого пула после отсева почти ничего не остаётся. Поэтому смотрим
        несколько дней вперёд — это несколько запросов, но все ответы кэшируются.
        """
        shows: dict[str, ShowRecord] = {}
        for day in _upcoming(request.schedule_days):
            try:
                entries = request.source.get_schedule(day)
            except Exception:  # noqa: BLE001 - расписание не должно ронять подбор
                continue
            for entry in entries:
                shows.setdefault(entry.show.external_id, entry.show)
        return list(shows.values())


def _upcoming(days: int) -> list[str]:
    today = date.today()
    return [(today + timedelta(days=offset)).isoformat() for offset in range(1, max(days, 1) + 1)]
