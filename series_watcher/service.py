"""Сервис: что делает бот и CLI поверх базы и источников.

Пока есть добавление сериала и список отслеживаемого. Опрос расписания и
уведомления добавляются следующим этапом, дедупликация там опирается на
`(show_id, season, number)`.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Sequence

from .catalog import explain, one_liner, pick_show, short_summary, title_from_description
from .config import Config
from .db import (
    NOT_INTERESTED,
    TELEGRAM_CHANNEL,
    Database,
    Episode,
    Show,
    TrackedShow,
    tracked_shows,
    utcnow,
)
from .recommend import Recommendation, Recommender
from .nlu import (
    ABOUT,
    ADD_SHOW,
    DIGEST,
    LIST,
    MARK_WATCHED,
    RECOMMEND,
    UNSUBSCRIBE,
    WHAT_NEW,
    WHERE_TO_WATCH,
    Intent,
    buttons_for,
)
from .notify import Notifier, footer
from .sources import SourceAnswer, SourceSet
from .tvmaze import TvMazeClient, TvMazeError, TvMazeShow, normalize_name

#: Канал один: сервис рассчитан на одного пользователя в Telegram.
NOTIFY_CHANNEL = TELEGRAM_CHANNEL


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


@dataclass(frozen=True)
class DueEpisode:
    """Эпизод, который уже вышел и о котором ещё не сообщали."""

    show: Show
    episode: Episode


@dataclass(frozen=True)
class ShowPoll:
    show: Show
    episodes_total: int
    new_episodes: int
    updated_episodes: int
    without_time: int
    error: str | None = None


@dataclass(frozen=True)
class PollReport:
    checked_at: datetime
    shows: tuple[ShowPoll, ...]
    due: tuple[DueEpisode, ...]

    @property
    def failed(self) -> tuple[ShowPoll, ...]:
        return tuple(item for item in self.shows if item.error)

    @property
    def new_episodes(self) -> int:
        return sum(item.new_episodes for item in self.shows)


@dataclass(frozen=True)
class RunReport:
    """Что сделал `run`: отправлено, отложено, не отправлено."""

    polled: PollReport
    sent: int = 0
    deferred: int = 0
    skipped: tuple[str, ...] = ()
    failed: tuple[str, ...] = ()

    @property
    def quiet(self) -> bool:
        return self.deferred > 0


@dataclass(frozen=True)
class Reply:
    """Ответ разговорному слою: текст и кнопки, когда разбор неуверенный."""

    text: str
    buttons: tuple[tuple[str, ...], ...] = ()


@dataclass
class SeriesWatcher:
    config: Config
    db: Database
    tvmaze: TvMazeClient
    notifier: Notifier | None = None
    sources: SourceSet | None = None
    llm: object | None = None  # LLMClient | None; тип не импортируем, чтобы не было цикла

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
            return self._add_by_description(query, needle, limit)

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
            guessed = pick_show(
                query, [item.show for item in candidates], llm=self.llm  # type: ignore[arg-type]
            )
            if guessed is not None:
                return self._store(guessed)
            return AddResult(
                status="needs_choice",
                candidates=candidates,
                message=f"Под «{query}» нашлось несколько сериалов, уточните выбор",
            )

        return self._store(chosen)

    def _add_by_description(self, query: str, needle: str, limit: int) -> AddResult:
        """Поиск провалился: пробуем понять описание и искать уже по названию."""
        if needle not in query.casefold():
            return AddResult(status="not_found", message=f"По запросу «{query}» ничего нет")
        title = title_from_description(query, llm=self.llm)  # type: ignore[arg-type]
        if not title:
            hint = ""
            if not getattr(self.llm, "configured", False):
                hint = " Похоже на описание: без модели (LLM_BASE_URL, LLM_MODEL) назвать сериал не по чему."
            return AddResult(
                status="not_found", message=f"По запросу «{query}» ничего нет.{hint}"
            )
        found = self.tvmaze.search_shows(title)
        if not found:
            return AddResult(
                status="not_found", message=f"По описанию «{query}» нашёл «{title}», но в каталоге его нет"
            )
        candidates = tuple(Candidate(show=item.show, score=item.score) for item in found[:limit])
        chosen = pick_show(
            title, [item.show for item in candidates], llm=self.llm  # type: ignore[arg-type]
        )
        if chosen is None:
            chosen = candidates[0].show
        result = self._store(chosen)
        if result.status == "added":
            result = AddResult(
                status=result.status,
                show=result.show,
                candidates=result.candidates,
                episodes=result.episodes,
                message=f"нашёл по описанию: {chosen.name}. {result.message}",
            )
        return result

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

    def poll(
        self,
        *,
        now: datetime | None = None,
        lookback_hours: float | None = None,
        channel: str = NOTIFY_CHANNEL,
    ) -> PollReport:
        """Опросить расписание и собрать, что положено сообщить.

        Список эпизодов каждый раз берётся без кэша: у `/shows/{id}/episodes`
        нет поля `updated`, и единственный способ узнать о переносе эфира —
        спросить. Лимит запросов держит RateLimiter, а не этот код.
        """
        moment = now if now is not None else utcnow()
        hours = self.config.notify_lookback_hours if lookback_hours is None else lookback_hours
        since = moment - timedelta(hours=hours) if hours > 0 else None

        polls: list[ShowPoll] = []
        for show in self.db.list_shows():
            polls.append(self._poll_show(show))

        known = {show.id: show for show in self.db.list_shows()}
        due = tuple(
            DueEpisode(show=known[row_id], episode=episode)
            for row_id, episode in self.db.due_episodes(
                now=moment, since=since, channel=channel
            )
            if row_id in known
        )
        return PollReport(checked_at=moment, shows=tuple(polls), due=due)

    def run(
        self,
        *,
        now: datetime | None = None,
        lookback_hours: float | None = None,
        dry_run: bool = False,
    ) -> RunReport:
        """Опросить расписание и разослать всё, что положено.

        Отправка идёт по одному сообщению на эпизод: упавшее сообщение не должно
        утаскивать за собой остальные. Отметка ставится только после успеха.
        """
        report = self.poll(now=now, lookback_hours=lookback_hours)
        if not report.due:
            return RunReport(polled=report)

        notifier = self.notifier
        if notifier is None:
            return RunReport(polled=report, skipped=tuple("нет отправителя" for _ in report.due))

        quiet = notifier.is_quiet(report.checked_at)
        if quiet and not dry_run:
            return RunReport(polled=report, deferred=len(report.due))

        sent = 0
        skipped: list[str] = []
        failed: list[str] = []
        for item in report.due:
            message = self.episode_message(item)
            if dry_run:
                skipped.append(message)
                continue
            result = notifier.send_episode(item.episode.id, message)
            if result.sent:
                sent += 1
            elif result.detail == "уже отправлено":
                continue
            elif not notifier.configured:
                skipped.append(result.detail)
            else:
                failed.append(f"{item.show.name} {item.episode.code}: {result.detail}")
        return RunReport(
            polled=report,
            sent=sent,
            deferred=len(report.due) if quiet else 0,
            skipped=tuple(skipped),
            failed=tuple(failed),
        )

    def digest(
        self, *, days: int = 7, now: datetime | None = None, recommend: bool = False
    ) -> str:
        """Сводка по ближайшим дням: что и когда выходит.

        С `recommend=True` добавляется подбор — так дайджест раз в неделю
        отвечает на вопрос «что бы посмотреть», а не только «что когда выходит».
        """
        moment = now if now is not None else utcnow()
        horizon = moment + timedelta(days=days)
        rows = self.db.upcoming_episodes(since=moment, until=horizon)
        lines = [f"{_plural_days(days)}:"]
        by_show: dict[str, list[str]] = {}
        for show, episode in rows:
            by_show.setdefault(show.name, []).append(
                f"{_when(episode.airstamp, self.config)} — {episode.code}"
                + (f" «{episode.name}»" if episode.name else "")
            )
        if len(by_show) == 1 and not any(by_show.values()):
            lines.append("пусто")
        for name, items in by_show.items():
            lines.append(name)
            lines.extend(f"  {item}" for item in items)
        if not by_show:
            lines.append("пусто: расписание пустое или сериалов нет")
        if recommend:
            items = self.recommend(limit=3)
            lines.append("")
            lines.append("Что бы посмотреть:")
            if not items:
                lines.append("  подходящего не нашлось")
            for item in items:
                lines.append(f"  • {item.line()} — {item.show.url or ''}".rstrip(" —"))
                blurb = one_liner(item.show.name, item.summary, llm=self.llm)  # type: ignore[arg-type]
                if blurb:
                    lines.append("    " + blurb)
                lines.append("    " + self._why(item))
        return footer(lines)

    def where_to_watch(self, show: Show, *, limit: int = 3) -> SourceAnswer:
        if self.sources is None:
            return SourceAnswer()
        answer = self.sources.where_to_watch(show, limit=limit)
        if answer.offers:
            return answer
        return SourceAnswer(offers=(), source=None, unavailable=answer.unavailable)

    def where_lines(self, show: Show, *, limit: int = 3) -> list[str]:
        """Ответ на «где смотреть» с готовым советом, если ссылки ещё нет."""
        answer = self.where_to_watch(show, limit=limit)
        if answer.offers:
            return answer.lines(limit)
        return answer.lines(limit, hint=add_link_hint(show.tvmaze_id))

    def episode_message(self, item: DueEpisode) -> str:
        episode = item.episode
        lines = [
            f"{item.show.name} — {episode.code}"
            + (f" «{episode.name}»" if episode.name else ""),
            f"Вышел {_when(episode.airstamp, self.config)} ({self.config.tz_name})",
        ]
        brief = self.episode_brief(item.episode)
        if brief:
            lines.append(brief)
        if item.show.url:
            lines.append(item.show.url)
        lines.extend(self.where_lines(item.show))
        return footer(lines)

    def episode_brief(self, episode: Episode) -> str | None:
        """Короткое описание эпизока на русском. Считается один раз и лежит в базе."""
        # Смотрим в базу, а не в переданный объект: копия эпизода могла устареть
        # ещё до того, как текст сжали.
        stored = self.db.get_summary_ru(episode.id)
        if stored:
            return stored
        text = short_summary(episode.summary, llm=self.llm)  # type: ignore[arg-type]
        if text:
            self.db.set_summary_ru(episode.id, text)
        return text

    def handle(self, intent: Intent, *, now: datetime | None = None) -> Reply:
        """Исполнить разобранное намерение и вернуть ответ."""
        handlers = {
            WHAT_NEW: self._answer_what_new,
            DIGEST: self._answer_digest,
            LIST: self._answer_list,
            ABOUT: self._answer_about,
            WHERE_TO_WATCH: self._answer_where_to_watch,
            ADD_SHOW: self._answer_add_show,
            UNSUBSCRIBE: self._answer_unsubscribe,
            MARK_WATCHED: self._answer_mark_watched,
            RECOMMEND: self._answer_recommend,
        }
        handler = handlers.get(intent.name)
        if handler is None:
            return Reply(
                text=plain(["Не понял, что нужно. Напишите проще, например «что нового»."]),
                buttons=buttons_for(intent),
            )
        return handler(intent, now=now)

    def _answer_what_new(self, intent: Intent, *, now: datetime | None = None) -> Reply:
        report = self.poll(now=now)
        if not report.due:
            return Reply(text=plain(["Пока ничего нового не вышло."]))
        lines = ["Что вышло:"]
        for item in report.due:
            title = f" «{item.episode.name}»" if item.episode.name else ""
            lines.append(
                f"{item.show.name} — {item.episode.code}{title} — "
                f"{_when(item.episode.airstamp, self.config)}"
            )
        if len(report.due) == 1:
            lines.extend(self.where_lines(report.due[0].show))
        return Reply(text=footer(lines))

    def _answer_digest(self, intent: Intent, *, now: datetime | None = None) -> Reply:
        return Reply(text=self.digest(days=7, now=now))

    def _answer_list(self, intent: Intent, *, now: datetime | None = None) -> Reply:
        """Отслеживаемые сериалы и ближайший эфир по каждому."""
        items = self.list_shows(now=now)
        if not items:
            return Reply(text=plain(["Список отслеживаемых пуст."]))
        lines = [f"Отслеживаю {len(items)}:"]
        for item in items:
            if item.next_episode and item.next_episode.airstamp:
                title = f" «{item.next_episode.name}»" if item.next_episode.name else ""
                when = f"{item.next_episode.code}{title} — {_when(item.next_episode.airstamp, self.config)}"
            else:
                when = "следующий эпизод не запланирован"
            lines.append(f"{item.show.name} — {when}")
        lines.append("Сводка по дням: «дайджест».")
        return Reply(text=footer(lines))

    def _answer_where_to_watch(self, intent: Intent, *, now: datetime | None = None) -> Reply:
        if intent.show:
            found = self.db.find_shows(intent.show)
            if not found:
                return Reply(text=footer([f"«{intent.show}» не отслеживается."]))
            shows = found
        else:
            shows = self.db.list_shows()
        if not shows:
            return Reply(text=plain(["Список отслеживаемых пуст."]))

        # Название здесь — эхо запроса самого пользователя, а ссылки — из нашего
        # файла: данных TVmaze в ответе нет, подпись не приписываем.
        lines: list[str] = []
        for show in shows:
            lines.append(show.name)
            for line in self.where_lines(show):
                lines.append("  " + line)
        return Reply(text=plain(lines))

    def _answer_add_show(self, intent: Intent, *, now: datetime | None = None) -> Reply:
        if not intent.show:
            return Reply(text=plain(["Какой сериал добавить?"]))
        result = self.add_show(intent.show)
        if result.status == "not_found":
            return Reply(text=footer([f"По запросу «{intent.show}» ничего нет."]))
        if result.status == "needs_choice":
            lines = [result.message]
            buttons = []
            for number, candidate in enumerate(result.candidates, start=1):
                lines.append(f"{number}. {candidate.label}")
                buttons.append((f"{number}. {candidate.show.name}", f"add {candidate.show.id}"))
            return Reply(text=footer(lines), buttons=tuple(buttons))
        assert result.show is not None
        verb = "Уже отслеживается" if result.status == "exists" else "Добавлен"
        return Reply(text=footer([f"{verb}: {result.show.name}", result.show.url or ""]))

    def _answer_unsubscribe(self, intent: Intent, *, now: datetime | None = None) -> Reply:
        if not intent.show:
            return Reply(text=plain(["Какой сериал перестать отслеживать?"]))
        found = self.db.find_shows(intent.show)
        if not found:
            return Reply(text=footer([f"«{intent.show}» не отслеживается."]))
        names = []
        for show in found:
            self.db.remove_show(show.id)
            names.append(show.name)
        return Reply(text=footer([f"Больше не отслеживаю: {', '.join(names)}"]))

    def _answer_mark_watched(self, intent: Intent, *, now: datetime | None = None) -> Reply:
        shows = self.db.find_shows(intent.show) if intent.show else self.db.list_shows()
        if not shows:
            return Reply(text=plain(["Нечего отмечать: список пуст."]))
        target = shows[0]
        episodes = self.db.list_episodes(target.id)
        candidates = _episodes_for(episodes, intent.season, intent.number)
        if not candidates and intent.number is not None:
            # У сериалов вроде «Beyond the Gates» сезон — это год, и его не назовут.
            candidates = _episodes_for(episodes, None, intent.number)
        if not candidates:
            return Reply(text=footer([f"Не нашёл такой эпизод у «{target.name}»."]))
        for episode in candidates:
            self.db.mark_watched(episode.id)
        names = ", ".join(f"{item.code}" for item in candidates)
        return Reply(text=footer([f"Отметил просмотренным: {target.name} — {names}"]))

    def recommend(
        self,
        *,
        limit: int = 5,
        min_rating: float | None = None,
        focus: Show | None = None,
    ) -> list[Recommendation]:
        """Подбор по всему списку или по названному сериалу."""
        if not self.db.list_shows():
            return []
        recommender = Recommender(config=self.config, db=self.db, tvmaze=self.tvmaze)
        options: dict[str, object] = {"limit": limit, "focus": focus}
        if min_rating is not None:
            options["min_rating"] = min_rating
        return recommender.candidates(**options)  # type: ignore[arg-type]

    def cold_start(self, *, limit: int = 3, schedule_days: int = 3) -> list[Recommendation]:
        recommender = Recommender(config=self.config, db=self.db, tvmaze=self.tvmaze)
        return recommender.cold_start(limit=limit, schedule_days=schedule_days)

    def reject(self, tvmaze_id: int, kind: str) -> bool:
        show = self.db.get_show_by_tvmaze_id(tvmaze_id)
        return self.db.reject(tvmaze_id, kind, title=show.name if show else None)

    def cold_start_reply(self) -> Reply:
        """Первый запуск: пустого списка не должно быть причиной молчать."""
        items = self.cold_start()
        if not items:
            return Reply(text=plain(["В расписании ничего подходящего не нашлось. Добавьте сериал по названию."]))
        lines = ["Список отслеживаемых пока пуст. Вот что идёт в эфире:"]
        buttons = []
        for item in items:
            lines.append(f"{item.line()} — {item.show.url or ''}".rstrip(" —"))
            lines.append("  " + "; ".join(item.reasons))
            buttons.append((f"Добавить {item.show.name}", f"add {item.show.id}"))
        lines.append("Или напишите название сериала — добавлю.")
        return Reply(text=footer(lines), buttons=tuple(buttons))

    def add_by_callback(self, payload: str) -> Reply:
        """Кнопки: `add <id>` добавляет, `reject <id>` — отказ от рекомендации."""
        parts = payload.split()
        if len(parts) != 2 or not parts[1].isdigit() or parts[0] not in {"add", "reject"}:
            return Reply(text=plain(["Кнопка устарела, напишите запрос текстом."]))
        if parts[0] == "reject":
            if not self.reject(int(parts[1]), NOT_INTERESTED):
                return Reply(text=plain(["Этот сериал и так не предлагаю."]))
            return Reply(text=plain(["Запомнил: больше не предлагаю."]))
        try:
            result = self.add_show_by_id(int(parts[1]))
        except TvMazeError as exc:
            return Reply(text=footer([f"Не получилось добавить: {exc}"]))
        if result.show is None:
            return Reply(text=plain(["Не получилось добавить сериал."]))
        verb = "Уже отслеживается" if result.status == "exists" else "Добавлен"
        return Reply(text=footer([f"{verb}: {result.show.name}", result.show.url or ""]))

    def _answer_recommend(self, intent: Intent, *, now: datetime | None = None) -> Reply:
        if not self.db.list_shows():
            return self.cold_start_reply()
        focus = self._focus_show(intent.show)
        items = self.recommend(focus=focus)
        if focus is not None and not items:
            return Reply(text=plain([f"Похожего на «{focus.name}» не нашлось."]))
        if not items:
            return Reply(text=plain(["Подходящих сериалов не нашлось."]))
        title = f"Похожее на «{focus.name}»:" if focus is not None else "Похожее на ваш список:"
        lines: list[str] = []
        buttons = []
        for item in items:
            lines.append(f"• {item.line()} — {item.show.url or ''}".rstrip(" —"))
            blurb = one_liner(item.show.name, item.summary, llm=self.llm)  # type: ignore[arg-type]
            if blurb:
                lines.append("  " + blurb)
            lines.append("  " + self._why(item))
            buttons.append((f"Не интересно: {item.show.name}", f"reject {item.show.id}"))
        return Reply(text=footer([title, ""] + lines), buttons=tuple(buttons))

    def _answer_about(self, intent: Intent, *, now: datetime | None = None) -> Reply:
        """«Что за сериал X»: карточка из каталога, а не из нашего списка."""
        if not intent.show:
            return Reply(text=plain(["О каком сериале рассказать?"]))
        tracked = self.db.find_shows(intent.show)
        tvmaze_id = tracked[0].tvmaze_id if tracked else self._search_id(intent.show)
        if tvmaze_id is None:
            return Reply(text=plain([f"«{intent.show}» не нашёл ни в списке, ни в каталоге."]))
        payload = self.tvmaze.get_show(tvmaze_id)

        lines = [f"{payload.name} ({', '.join(payload.genres) or payload.kind or 'сериал'})"]
        if payload.status:
            lines.append(
                "Статус: " + payload.status + (f", канал {payload.network}" if payload.network else "")
            )
        if payload.rating is not None:
            lines.append(f"Рейтинг TVmaze: {payload.rating}")
        blurb = one_liner(payload.name, payload.summary, llm=self.llm)  # type: ignore[arg-type]
        if blurb:
            lines.append(blurb)

        if tracked:
            item = next((it for it in self.list_shows(now=now) if it.show.id == tracked[0].id), None)
            if item and item.next_episode and item.next_episode.airstamp:
                lines.append(f"Следующий эпизод: {_when(item.next_episode.airstamp, self.config)}")
            lines.extend(self.where_lines(tracked[0]))
        else:
            lines.append("Ещё не отслеживается.")
        if payload.url:
            lines.append(payload.url)

        buttons = () if tracked else ((f"Добавить {payload.name}", f"add {payload.id}"),)
        return Reply(text=footer(lines), buttons=buttons)

    def _search_id(self, query: str) -> int | None:
        """Каталог TVmaze: точное совпадение по названию, иначе лучшее из выдачи."""
        try:
            found = self.tvmaze.search_shows(query)
        except TvMazeError:
            return None
        if not found:
            return None
        needle = normalize_name(query)
        exact = next((item for item in found if item.show.normalized_name == needle), None)
        return (exact or found[0]).show.id

    def _focus_show(self, name: str | None) -> Show | None:
        """Сериал, названный в запросе: «похожие на Slow Horses»."""
        if not name:
            return None
        found = self.db.find_shows(name)
        return found[0] if found else None

    def blurb(self, item: Recommendation) -> str | None:
        """Кратко о самом сериале — показывают и бот, и CLI."""
        return one_liner(item.show.name, item.summary, llm=self.llm)  # type: ignore[arg-type]

    def why(self, item: Recommendation) -> str:
        """Одна строка «почему» — её показывают и бот, и CLI."""
        return self._why(item)

    def _why(self, item: Recommendation) -> str:
        """Одна строка «почему»: словами модели, а без неё — фактами из API."""
        prose = explain(item.show.name, item.reasons, llm=self.llm)  # type: ignore[arg-type]
        return prose or "; ".join(item.reasons)

    def _poll_show(self, show: Show) -> ShowPoll:
        try:
            fresh = self.tvmaze.get_show(show.tvmaze_id)
            episodes = self.tvmaze.get_episodes(show.tvmaze_id, refresh=True)
        except TvMazeError as exc:
            return ShowPoll(
                show=show,
                episodes_total=self.db.count_episodes(show.id),
                new_episodes=0,
                updated_episodes=0,
                without_time=self.db.count_episodes_without_time(show.id),
                error=str(exc),
            )
        stored = self.db.upsert_show(fresh)
        added, updated = self.db.upsert_episodes(stored.id, episodes)
        return ShowPoll(
            show=stored,
            episodes_total=self.db.count_episodes(stored.id),
            new_episodes=added,
            updated_episodes=updated,
            without_time=self.db.count_episodes_without_time(stored.id),
        )

    def _store(self, show: TvMazeShow) -> AddResult:
        known = self.db.get_show_by_tvmaze_id(show.id) is not None
        stored = self.db.upsert_show(show)
        episodes = self.tvmaze.get_episodes(show.id)
        added, updated = self.db.upsert_episodes(stored.id, episodes)
        message = f"эпизодов в базе: {len(episodes)} (новых {added}, обновлено {updated})"
        status = "exists" if known else "added"
        return AddResult(status=status, show=stored, episodes=len(episodes), message=message)


def plain(lines: Sequence[str]) -> str:
    """Служебный ответ: данных TVmaze в нём нет, подпись не приписываем."""
    return "\n".join(line for line in lines if line)


def add_link_hint(tvmaze_id: int) -> str:
    """Готовая команда вместо тупика: добавить ссылку на этот сериал."""
    return f"series-watcher where --add {tvmaze_id} --url <ссылка>"


def _when(value: datetime | None, config: Config) -> str:
    if value is None:
        return "время неизвестно"
    return value.astimezone(config.tz).strftime("%d.%m.%Y %H:%M")


def _episodes_for(episodes: Sequence[Episode], season: int | None, number: int | None) -> list[Episode]:
    return [
        episode
        for episode in episodes
        if (season is None or episode.season == season)
        and (number is None or episode.number == number)
    ]


def _plural_days(days: int) -> str:
    """«Ближайшие 1 день», «2 дня», «5 дней» — по-русски это три формы."""
    if days % 10 == 1 and days % 100 != 11:
        word = "день"
    elif 2 <= days % 10 <= 4 and not 12 <= days % 100 <= 14:
        word = "дня"
    else:
        word = "дней"
    return f"Ближайшие {days} {word}"
