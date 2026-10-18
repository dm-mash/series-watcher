"""CLI: add / list и дальше check, run, digest, recommend."""

from __future__ import annotations

import argparse
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Sequence, TextIO

from .bot import Bot
from .config import Config, ConfigError
from .db import NOT_INTERESTED, ALREADY_WATCHED, Database, Show, TrackedShow
from .llm import LLMClient
from .notify import Notifier, NotifyError, TelegramApi
from .service import AddResult, SeriesWatcher
from .tmdb import TmdbClient
from .sources import SourceSet
from .sources.custom import CustomSource
from .sources.justwatch import JustWatchSource
from .tvmaze import TvMazeClient, TvMazeError

TVMAZE_CREDIT = "Данные: TVmaze (CC BY-SA 4.0), https://www.tvmaze.com"

EXIT_OK = 0
EXIT_ERROR = 1
EXIT_NEEDS_CHOICE = 2


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="series-watcher",
        description="Трекер выхода сериалов. Расписание берётся из TVmaze, время хранится в UTC.",
    )
    parser.add_argument("--db", help="путь к SQLite (перекрывает DB_PATH)")
    parser.add_argument("--tz", help="зона показа времени (перекрывает TZ)")
    sub = parser.add_subparsers(dest="command", required=True)

    add = sub.add_parser("add", help="добавить сериал в отслеживаемые")
    add.add_argument("query", nargs="?", help="название сериала")
    add.add_argument("--pick", type=int, help="номер варианта из выдачи поиска")
    add.add_argument("--id", type=int, dest="tvmaze_id", help="добавить по id TVmaze")

    sub.add_parser("list", help="показать отслеживаемые сериалы")

    check = sub.add_parser("check", help="опросить расписание и показать, что пора сообщить")
    check.add_argument(
        "--lookback",
        type=float,
        help="сколько часов назад смотреть (по умолчанию NOTIFY_LOOKBACK_HOURS)",
    )

    run = sub.add_parser("run", help="опросить расписание и разослать уведомления")
    run.add_argument("--lookback", type=float, help="окно назад в часах")
    run.add_argument("--dry-run", action="store_true", help="показать сообщения, не отправляя")
    run.add_argument("--now", help="считать текущим моментом (для проверок), ISO 8601")

    digest = sub.add_parser("digest", help="сводка по ближайшим дням")
    digest.add_argument("--days", type=int, default=7, help="сколько дней вперёд (по умолчанию 7)")
    digest.add_argument("--send", action="store_true", help="отправить в Telegram, а не напечатать")

    recommend = sub.add_parser("recommend", help="подобрать сериалы по вашему списку")
    recommend.add_argument("--limit", type=int, default=5, help="сколько показать (по умолчанию 5)")
    recommend.add_argument("--min-rating", type=float, help="порог рейтинга TMDB/TVmaze")
    recommend.add_argument(
        "--no-tmdb", action="store_true", help="не ходить в TMDB даже если ключ есть"
    )
    recommend.add_argument(
        "--reject", type=int, metavar="ID", help="отказаться от сериала с таким id TVmaze"
    )
    recommend.add_argument(
        "--reject-kind",
        choices=(NOT_INTERESTED, ALREADY_WATCHED),
        default=NOT_INTERESTED,
        help="как отказаться (по умолчанию «не интересно»)",
    )

    bot = sub.add_parser("bot", help="разговорный слой: опрос Telegram и ответы")
    bot.add_argument("--once", action="store_true", help="один заход и выход")
    bot.add_argument("--timeout", type=int, default=25, help="long polling, секунд (по умолчанию 25)")
    return parser


def main(
    argv: Sequence[str] | None = None,
    *,
    stdout: TextIO | None = None,
    stderr: TextIO | None = None,
    transport=None,
) -> int:
    out = stdout or sys.stdout
    err = stderr or sys.stderr
    args = build_parser().parse_args(argv)

    try:
        config = Config.from_env()
        if args.db:
            config = config.replace(db_path=Path(args.db))
        if args.tz:
            config = config.replace(tz_name=args.tz)
    except ConfigError as exc:
        print(f"Ошибка настроек: {exc}", file=err)
        return EXIT_ERROR

    db = Database(config.db_path)
    tvmaze = TvMazeClient(
        base_url=config.tvmaze_base_url,
        cache=db,
        timeout=config.http_timeout,
        retries=config.http_retries,
        ttl=config.tvmaze_cache_ttl,
        user_agent=config.user_agent,
        rate_limit=config.tvmaze_rate_limit,
    )
    notifier = Notifier(config, db, transport=transport)
    sources = SourceSet(
        [
            CustomSource(config.custom_sources_path),
            JustWatchSource(config.justwatch_api_key, country=config.justwatch_country),
        ]
    )
    tmdb = TmdbClient(config.tmdb_api_key, cache=db, transport=transport)
    watcher = SeriesWatcher(
        config=config, db=db, tvmaze=tvmaze, notifier=notifier, sources=sources, tmdb=tmdb
    )
    telegram = TelegramApi(config, notifier.client, token=config.telegram_bot_token)
    llm = LLMClient(config, transport=transport)
    bot = Bot(config, watcher, telegram, llm=llm)
    try:
        if args.command == "add":
            return _cmd_add(watcher, config, args, out, err)
        if args.command == "list":
            return _cmd_list(watcher, config, out)
        if args.command == "check":
            return _cmd_check(watcher, config, args.lookback, out)
        if args.command == "run":
            return _cmd_run(watcher, config, args, out, err)
        if args.command == "digest":
            return _cmd_digest(watcher, config, args, out, err)
        if args.command == "bot":
            return _cmd_bot(bot, args, out, err)
        if args.command == "recommend":
            return _cmd_recommend(watcher, config, args, out, err)
    except TvMazeError as exc:
        print(f"TVmaze: {exc}", file=err)
        return EXIT_ERROR
    except NotifyError as exc:
        print(f"Telegram: {exc}", file=err)
        return EXIT_ERROR
    except ConfigError as exc:
        print(f"Ошибка: {exc}", file=err)
        return EXIT_ERROR
    finally:
        tmdb.close()
        llm.close()
        telegram.close()
        notifier.close()
        tvmaze.close()
        db.close()
    return EXIT_ERROR


def _cmd_add(watcher: SeriesWatcher, config: Config, args: argparse.Namespace, out: TextIO, err: TextIO) -> int:
    if args.tvmaze_id is None and not args.query:
        print("Нужно название сериала или --id", file=err)
        return EXIT_ERROR
    if args.tvmaze_id is not None:
        result = watcher.add_show_by_id(args.tvmaze_id)
    else:
        result = watcher.add_show(args.query, pick=args.pick)
        result = _resolve_choice(watcher, result, out, err)
        if result is None:
            return EXIT_NEEDS_CHOICE

    if result.status == "not_found":
        print(result.message, file=err)
        return EXIT_ERROR

    show = result.show
    assert show is not None
    verb = "Уже отслеживается" if result.status == "exists" else "Добавлен"
    print(f"{verb}: {show.name}", file=out)
    print(f"  {_show_meta(show)}", file=out)
    if result.message:
        print(f"  {result.message}", file=out)
    if show.url:
        print(f"  {show.url}", file=out)
    _print_next(watcher, config, show, out)
    print(TVMAZE_CREDIT, file=out)
    return EXIT_OK


def _resolve_choice(
    watcher: SeriesWatcher, result: AddResult, out: TextIO, err: TextIO
) -> AddResult | None:
    """Показать варианты и, если есть терминал, дать выбрать."""
    while result.status == "needs_choice" and result.candidates:
        print(result.message, file=err)
        for number, candidate in enumerate(result.candidates, start=1):
            print(f"  {number}. {candidate.label}", file=err)
        if not sys.stdin.isatty():
            print("Выберите номер: добавьте --pick N", file=err)
            return None
        raw = input("Номер варианта (Enter — отмена): ").strip()
        if not raw:
            return None
        if not raw.isdigit():
            print("Нужен номер варианта.", file=err)
            continue
        result = _store_picked(watcher, result, int(raw), err)
    return result


def _store_picked(watcher: SeriesWatcher, result: AddResult, pick: int, err: TextIO) -> AddResult:
    candidate = result.candidates[pick - 1] if 1 <= pick <= len(result.candidates) else None
    if candidate is None:
        print(f"Нет варианта №{pick}, их {len(result.candidates)}", file=err)
        return AddResult(status="needs_choice", candidates=result.candidates, message=result.message)
    return watcher.add_show_by_id(candidate.show.id)


def _cmd_list(watcher: SeriesWatcher, config: Config, out: TextIO) -> int:
    items = watcher.list_shows()
    if not items:
        print("Список пуст. Добавьте сериал: series-watcher add «Название»", file=out)
        _print_degradation(config, out)
        return EXIT_OK

    print(f"Отслеживается сериалов: {len(items)} (время: {config.tz_name})", file=out)
    print(file=out)
    for item in items:
        show = item.show
        print(f"  {show.name}", file=out)
        print(f"    {_show_meta(show)}", file=out)
        _print_next(watcher, config, show, out, item)
        print(f"    всего эпизодов: {item.episodes_total}", file=out)
        print(file=out)

    print(TVMAZE_CREDIT, file=out)
    _print_degradation(config, out)
    return EXIT_OK


def _cmd_check(watcher: SeriesWatcher, config: Config, lookback: float | None, out: TextIO) -> int:
    report = watcher.poll(lookback_hours=lookback)
    print(
        f"Проверено в {_format_when(report.checked_at, config)} ({config.tz_name}), "
        f"сериалов: {len(report.shows)}",
        file=out,
    )

    for item in report.shows:
        if item.error:
            print(file=out)
            print(f"  {item.show.name}: не проверился — {item.error}", file=out)
            continue
        parts = [f"эпизодов: {item.episodes_total}", f"новых: {item.new_episodes}"]
        if item.updated_episodes:
            parts.append(f"обновлено: {item.updated_episodes}")
        if item.without_time:
            parts.append(f"без времени эфира: {item.without_time}")
        print(file=out)
        print(f"  {item.show.name} — {' · '.join(parts)}", file=out)

    print(file=out)
    if report.due:
        print("Пора сообщить:", file=out)
        for item in report.due:
            title = f" «{item.episode.name}»" if item.episode.name else ""
            print(
                f"  {item.show.name} — {item.episode.code}{title} — "
                f"{_format_when(item.episode.airstamp, config)}",
                file=out,
            )
    else:
        print("Пора сообщить: нечего, все вышедшие эпизоды уже отмечены.", file=out)

    print(file=out)
    print(TVMAZE_CREDIT, file=out)
    return EXIT_ERROR if report.failed else EXIT_OK


def _cmd_run(watcher: SeriesWatcher, config: Config, args: argparse.Namespace, out: TextIO, err: TextIO) -> int:
    moment = _parse_moment(args.now)
    report = watcher.run(now=moment, lookback_hours=args.lookback, dry_run=args.dry_run)

    for item in report.polled.shows:
        if item.error:
            print(f"  {item.show.name}: не проверился — {item.error}", file=err)

    if report.deferred:
        note = "отложено" if not args.dry_run else "в реальном запуске отложено бы"
        print(
            f"Тихий режим ({_quiet_label(config)}): {note} {report.deferred}. "
            "Отправится утром.",
            file=out,
        )
    for text in report.skipped:
        print(text, file=out)
        print(file=out)
    for failure in report.failed:
        print(f"Не отправлено — {failure}", file=err)

    if report.sent:
        print(f"Отправлено сообщений: {report.sent}", file=out)
    elif not report.skipped and not report.deferred:
        print("Отправлять нечего: всё, что вышло, уже отмечено.", file=out)
    if report.failed:
        return EXIT_ERROR
    if report.polled.failed:
        return EXIT_ERROR
    return EXIT_OK


def _cmd_digest(watcher: SeriesWatcher, config: Config, args: argparse.Namespace, out: TextIO, err: TextIO) -> int:
    text = watcher.digest(days=args.days)
    if not args.send:
        print(text, file=out)
        return EXIT_OK
    result = watcher.notifier.send(text) if watcher.notifier else None
    if result is None or not result.sent:
        detail = result.detail if result else "нет отправителя"
        print(f"Дайджест не отправлен: {detail}", file=err)
        return EXIT_ERROR
    print("Дайджест отправлен", file=out)
    return EXIT_OK


def _cmd_recommend(watcher: SeriesWatcher, config: Config, args: argparse.Namespace, out: TextIO, err: TextIO) -> int:
    if args.reject is not None:
        if watcher.reject(args.reject, args.reject_kind):
            print(f"Запомнил: этот сериал больше не предлагаю ({args.reject_kind}).", file=out)
            return EXIT_OK
        print(f"Сериал {args.reject} и так в отказах.", file=out)
        return EXIT_OK

    if not watcher.db.list_shows():
        print("Список пуст: добавьте сериал, иначе подбирать не от чего.", file=out)
        return EXIT_OK

    items = watcher.recommend(
        limit=args.limit, use_tmdb=not args.no_tmdb, min_rating=args.min_rating
    )
    if not items:
        print("Подходящих сериалов не нашлось.", file=out)
    else:
        print(f"Похожее на ваш список ({len(items)}):", file=out)
        print(file=out)
        for item in items:
            print(f"  {item.line()}", file=out)
            for reason in item.reasons:
                print(f"    {reason}", file=out)
            print(f"    {item.show.url}", file=out)
            print(f"    отказаться: series-watcher recommend --reject {item.show.id}", file=out)
            print(file=out)

    print(TVMAZE_CREDIT, file=out)
    for note in config.degradation_notes():
        if "TMDB" in note:
            print(note, file=out)
    return EXIT_OK


def _cmd_bot(bot: Bot, args: argparse.Namespace, out: TextIO, err: TextIO) -> int:
    if not bot.usable:
        print(
            "боту нужен TELEGRAM_BOT_TOKEN: без него некуда спросить и некуда ответить",
            file=err,
        )
        return EXIT_ERROR
    try:
        bot.run(once=args.once, poll_timeout=args.timeout)
    except KeyboardInterrupt:  # Ctrl-C — обычное дело для демона
        print("остановлено", file=out)
        return EXIT_OK
    return EXIT_OK


def _quiet_label(config: Config) -> str:
    quiet = config.notify_quiet_hours
    if quiet is None:
        return "выключен"
    return f"{quiet[0]:02d}:00-{quiet[1]:02d}:00 {config.tz_name}"


def _parse_moment(value: str | None) -> datetime | None:
    if not value:
        return None
    try:
        moment = datetime.fromisoformat(value)
    except ValueError as exc:
        raise ConfigError(f"--now ожидается как ISO 8601, получено {value!r}") from exc
    if moment.tzinfo is None:
        moment = moment.replace(tzinfo=timezone.utc)
    return moment


def _show_meta(show: Show) -> str:
    parts = []
    if show.genres:
        parts.append(", ".join(show.genres))
    for value in (show.status, show.network, show.premiered):
        if value:
            parts.append(str(value))
    return " · ".join(parts)


def _print_next(
    watcher: SeriesWatcher,
    config: Config,
    show: Show,
    out: TextIO,
    item: TrackedShow | None = None,
) -> None:
    episode = item.next_episode if item else watcher.db.next_episode(show.id)
    if episode is None:
        print("    следующий эпизод: не запланирован", file=out)
        return
    when = _format_when(episode.airstamp, config)
    title = f" «{episode.name}»" if episode.name else ""
    print(f"    следующий: {episode.code}{title} — {when}", file=out)


def _format_when(value: datetime | None, config: Config) -> str:
    if value is None:
        return "время неизвестно"
    local = value.astimezone(config.tz)
    return local.strftime("%d.%m.%Y %H:%M")


def _print_degradation(config: Config, out: TextIO) -> None:
    notes = config.degradation_notes()
    if not notes:
        return
    print(file=out)
    print("Что отключено:", file=out)
    for note in notes:
        print(f"  - {note}", file=out)


if __name__ == "__main__":
    raise SystemExit(main())