"""Рассылка: отправка, дедупликация, тихий режим, деградация без токена."""

from __future__ import annotations

from datetime import datetime, timezone

import pytest

from series_watcher.config import Config
from series_watcher.db import TELEGRAM_CHANNEL, Database
from series_watcher.notify import Notifier
from series_watcher.service import SeriesWatcher
from series_watcher.sources import SourceSet
from series_watcher.sources.custom import CustomSource
from series_watcher.schedule import TvMazeSchedule
from series_watcher.tvmaze import TvMazeClient

from conftest import FakeTelegram, load_fixture

from test_check import NOW

AIRING = datetime(2026, 10, 21, 12, 0, tzinfo=timezone.utc)  # 15:00 в Москве, не тихо
NIGHT = datetime(2026, 10, 19, 21, 30, tzinfo=timezone.utc)  # 00:30 в Москве, тихо


@pytest.fixture
def parts(tmp_path, fake_tvmaze, sleeps):
    db = Database(tmp_path / "test.db")
    client = TvMazeClient(
        transport=fake_tvmaze.transport(), cache=db, sleep=sleeps.append, rate_limit=(0, 10.0)
    )
    fake_tvmaze.add("/shows/75194/episodes", load_fixture("episodes_running"))
    watcher = SeriesWatcher(config=Config(tz_name="Europe/Moscow"), db=db, source=TvMazeSchedule(client))
    watcher.add_show_by_id(75194)
    yield watcher, fake_tvmaze, db
    client.close()
    db.close()


def attach(parts, telegram: FakeTelegram, *, quiet=(23, 8), sources: SourceSet | None = None):
    watcher, _, db = parts
    watcher.config = watcher.config.replace(notify_quiet_hours=quiet)
    watcher.notifier = Notifier(
        watcher.config,
        db,
        transport=telegram.transport(),
        token="token",
        chat_id="42",
    )
    watcher.sources = sources or SourceSet([CustomSource("/нет/такого.json")])
    return watcher


def test_run_sends_one_message_per_episode(parts):
    telegram = FakeTelegram()
    watcher = attach(parts, telegram)

    report = watcher.run(now=NOW, lookback_hours=72)

    assert report.sent == 3
    assert len(telegram.sent) == 3
    assert "S2026E193" in telegram.sent[0]["text"]
    assert "где смотреть: не настроено" in telegram.sent[0]["text"]
    assert telegram.sent[0]["text"].count("tvmaze.com") == 2


def test_run_marks_episodes_and_does_not_repeat(parts):
    telegram = FakeTelegram()
    watcher = attach(parts, telegram)

    watcher.run(now=NOW, lookback_hours=72)
    second = watcher.run(now=NOW, lookback_hours=72)

    assert second.sent == 0
    assert len(telegram.sent) == 3
    episodes = watcher.db.list_episodes(watcher.db.list_shows()[0].id)
    sent_codes = {item.code for item in episodes if watcher.db.is_sent(item.id, TELEGRAM_CHANNEL)}
    assert sent_codes == {"S2026E193", "S2026E194", "S2026E195"}


def test_run_message_carries_where_to_watch(parts, tmp_path):
    import json

    path = tmp_path / "where.json"
    show = watcher_show = parts[0].db.list_shows()[0]
    path.write_text(
        json.dumps({str(show.key.external_id): [{"provider": "Кинопоиск", "url": "https://example.tv/x"}]}),
        encoding="utf-8",
    )
    telegram = FakeTelegram()
    watcher = attach(parts, telegram, sources=SourceSet([CustomSource(path)]))

    watcher.run(now=NOW, lookback_hours=72)

    assert watcher_show.name
    assert "Кинопоиск — https://example.tv/x" in telegram.sent[0]["text"]


def test_dry_run_sends_nothing(parts):
    telegram = FakeTelegram()
    watcher = attach(parts, telegram)

    report = watcher.run(now=NOW, lookback_hours=72, dry_run=True)

    assert report.sent == 0
    assert telegram.sent == []
    assert len(report.skipped) == 3
    assert "S2026E193" in report.skipped[0]


def test_night_run_is_deferred(parts):
    telegram = FakeTelegram()
    watcher = attach(parts, telegram)

    report = watcher.run(now=NIGHT, lookback_hours=72)

    assert report.deferred == 1
    assert telegram.sent == []
    assert report.polled.due  # эпизоды ждут утра, а не потеряны


def test_quiet_hours_can_be_switched_off(parts):
    telegram = FakeTelegram()
    watcher = attach(parts, telegram, quiet=None)

    assert watcher.run(now=NIGHT, lookback_hours=72).sent == 1


def test_without_token_nothing_is_marked(parts):
    watcher, _, db = parts
    watcher.notifier = Notifier(watcher.config, db)
    watcher.sources = SourceSet([CustomSource("/нет/такого.json")])

    report = watcher.run(now=NOW, lookback_hours=72)

    assert report.sent == 0
    assert report.skipped
    assert "TELEGRAM_BOT_TOKEN" in report.skipped[0]
    assert db.is_sent(1, TELEGRAM_CHANNEL) is False


def test_failed_send_is_reported_and_not_marked(parts):
    telegram = FakeTelegram(status=500)
    watcher = attach(parts, telegram)

    report = watcher.run(now=NOW, lookback_hours=72)

    assert report.sent == 0
    assert len(report.failed) == 3
    assert "S2026E193" in report.failed[0]
    assert watcher.db.is_sent(1, TELEGRAM_CHANNEL) is False


def test_digest_lists_upcoming_episodes(parts):
    watcher, _, _ = parts

    text = watcher.digest(days=3, now=NOW)

    assert "Ближайшие 3 дня" in text
    assert "Beyond the Gates" in text
    assert "S2026E196" in text
    assert "S2026E193" not in text  # уже вышедшие в дайджест не попадают
    # Подпись с атрибуцией — курсивом и в конце, но она на месте.
    assert text.rstrip().endswith("<i>Данные: TVmaze (CC BY-SA 4.0), https://www.tvmaze.com</i>")


def test_digest_says_when_nothing_is_planned(parts):
    watcher, _, _ = parts

    text = watcher.digest(days=0, now=NOW)

    assert "S2026E196" not in text
    assert "пусто" in text

def test_missing_link_tells_what_to_do(parts):
    watcher, _, _ = parts
    watcher.sources = SourceSet([CustomSource("/нет/такого.json")])

    lines = watcher.where_lines(watcher.db.list_shows()[0])

    assert lines == [
        "где смотреть: не настроено — series-watcher where --add 75194 --url <ссылка>"
    ]


def test_existing_link_is_used(parts, tmp_path):
    import json

    watcher, _, db = parts
    show = db.list_shows()[0]
    path = tmp_path / "where.json"
    path.write_text(json.dumps({str(show.key.external_id): [{"url": "https://example.tv/x"}]}), encoding="utf-8")
    watcher.sources = SourceSet([CustomSource(path)])

    lines = watcher.where_lines(show)

    assert lines == ["где смотреть: ссылка — https://example.tv/x"]


def test_digest_groups_by_day_not_by_show(parts):
    """Вопрос «когда что выходит» — про календарь, а не про сериалы."""
    watcher, _, _ = parts

    text = watcher.digest(days=3, now=NOW)

    # Заголовки — дни, а не сериалы: в трёхдневном окне тут только «Завтра»
    # и пара дней недели.
    assert "Завтра" in text
    assert any(day in text for day in ("Понедельник", "Вторник", "Среда", "Четверг", "Пятница", "Суббота", "Воскресенье"))
    # Один сериал не превращает сводку в простыню: он повторяется по дням.
    assert text.count("Beyond the Gates") <= 3
