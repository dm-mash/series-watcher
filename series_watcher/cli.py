"""CLI: add / list и дальше check, run, digest, recommend."""

from __future__ import annotations

import argparse
import sys
from datetime import datetime
from pathlib import Path
from typing import Sequence, TextIO

from .config import Config, ConfigError
from .db import Database, Show, TrackedShow
from .service import AddResult, SeriesWatcher
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
    return parser


def main(argv: Sequence[str] | None = None, *, stdout: TextIO | None = None, stderr: TextIO | None = None) -> int:
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
    watcher = SeriesWatcher(config=config, db=db, tvmaze=tvmaze)
    try:
        if args.command == "add":
            return _cmd_add(watcher, config, args, out, err)
        if args.command == "list":
            return _cmd_list(watcher, config, out)
    except TvMazeError as exc:
        print(f"TVmaze: {exc}", file=err)
        return EXIT_ERROR
    finally:
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