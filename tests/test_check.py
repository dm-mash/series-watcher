"""Детектор новых эпизодов: опрос, дедупликация, окно и часовые пояса."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest

from series_watcher.config import Config
from series_watcher.db import Database
from series_watcher.service import NOTIFY_CHANNEL, SeriesWatcher
from series_watcher.schedule import TvMazeSchedule
from series_watcher.tvmaze import TvMazeClient

from conftest import load_fixture, store_show

# Момент, относительно которого считаем «уже вышел». В фикстуре ежедневный сериал:
# 193 вышел 19 октября, 195 — 21 октября в 18:00 UTC.
NOW = datetime(2026, 10, 21, 19, 0, tzinfo=timezone.utc)
DUE_CODES = ["S2026E193", "S2026E194", "S2026E195"]


def count(payload: list[dict]) -> int:
    return len(payload)


@pytest.fixture
def watcher(tmp_path, fake_tvmaze, sleeps) -> SeriesWatcher:
    db = Database(tmp_path / "test.db")
    client = TvMazeClient(
        transport=fake_tvmaze.transport(),
        cache=db,
        sleep=sleeps.append,
        rate_limit=(0, 10.0),
    )
    yield SeriesWatcher(
        config=Config(notify_lookback_hours=72.0), db=db, source=TvMazeSchedule(client)
    )
    client.close()
    db.close()


def episodes_from(code: int) -> list[dict]:
    """Тот же настоящий ответ TVmaze, обрезанный до указанного эпизода.

    TVmaze отдаёт эпизоды по порядку, поэтому такой срез — правдоподобное
    «состояние API», каким оно было до анонса.
    """
    payload = load_fixture("episodes_running")
    index = next(
        i for i, item in enumerate(payload) if item["season"] == 2026 and item["number"] == code
    )
    return payload[: index + 1]


def track(fake_tvmaze, watcher: SeriesWatcher, last_code: int) -> None:
    fake_tvmaze.add("/shows/75194/episodes", episodes_from(last_code))
    watcher.add_show_by_id(75194)


def poll_with(fake_tvmaze, watcher: SeriesWatcher, episodes: list[dict], **kwargs):
    fake_tvmaze.add("/shows/75194/episodes", episodes)
    kwargs.setdefault("now", NOW)
    return watcher.poll(**kwargs)


def test_first_poll_reports_announced_episodes(watcher, fake_tvmaze):
    track(fake_tvmaze, watcher, 192)
    report = poll_with(fake_tvmaze, watcher, episodes_from(193))

    assert report.new_episodes == 1
    assert report.shows[0].episodes_total == count(episodes_from(193))
    assert report.shows[0].without_time == 0


def test_second_poll_without_changes_reports_nothing_new(watcher, fake_tvmaze):
    track(fake_tvmaze, watcher, 192)
    poll_with(fake_tvmaze, watcher, episodes_from(193))
    report = poll_with(fake_tvmaze, watcher, episodes_from(193))

    assert report.new_episodes == 0
    assert report.shows[0].updated_episodes == count(episodes_from(193))
    assert [item.episode.code for item in report.due] == ["S2026E193"]


def test_announced_episode_waits_for_its_air_time(watcher, fake_tvmaze):
    """Анонс есть, эфира ещё нет — сообщать о нём рано."""
    before_airing = datetime(2026, 10, 19, 12, 0, tzinfo=timezone.utc)
    track(fake_tvmaze, watcher, 192)
    first = poll_with(fake_tvmaze, watcher, episodes_from(193), now=before_airing)

    assert first.new_episodes == 1
    assert [item.episode.code for item in first.due] == ["S2026E192"]

    again = poll_with(fake_tvmaze, watcher, episodes_from(193), now=before_airing)
    assert again.new_episodes == 0
    assert [item.episode.code for item in again.due] == ["S2026E192"]


def test_announced_episode_becomes_due_after_airing(watcher, fake_tvmaze):
    track(fake_tvmaze, watcher, 192)
    poll_with(fake_tvmaze, watcher, episodes_from(193))

    report = poll_with(fake_tvmaze, watcher, episodes_from(193))
    assert [item.episode.code for item in report.due] == ["S2026E193"]
    assert report.due[0].episode.aired_at == datetime(2026, 10, 19, 18, 0, tzinfo=timezone.utc)


def test_rescheduled_episode_updates_time_and_keeps_identity(watcher, fake_tvmaze):
    track(fake_tvmaze, watcher, 195)
    before = {
        item.episode.code: item.episode for item in poll_with(fake_tvmaze, watcher, episodes_from(195)).due
    }

    moved = [dict(item) for item in episodes_from(195)]
    for item in moved:
        if item["airdate"] == "2026-10-20":
            # Тот же эпизод, но эфир перенесли на местную полночь 20-го.
            item["airstamp"] = "2026-10-21T04:00:00+00:00"
            item["airtime"] = "00:00"
    report = poll_with(fake_tvmaze, watcher, moved)

    assert report.new_episodes == 0
    after = {item.episode.code: item.episode for item in report.due}
    assert after["S2026E194"].id == before["S2026E194"].id
    assert after["S2026E194"].aired_at == datetime(2026, 10, 21, 4, 0, tzinfo=timezone.utc)
    assert after["S2026E194"].aired_at.hour == 4


def test_due_episodes_are_reported_once(watcher, fake_tvmaze):
    track(fake_tvmaze, watcher, 195)
    report = poll_with(fake_tvmaze, watcher, episodes_from(195))
    assert [item.episode.code for item in report.due] == DUE_CODES

    for item in report.due:
        assert watcher.db.mark_sent(item.episode.id, NOTIFY_CHANNEL) is True

    assert poll_with(fake_tvmaze, watcher, episodes_from(195)).due == ()


def test_watched_episode_is_not_reported(watcher, fake_tvmaze):
    track(fake_tvmaze, watcher, 195)
    report = poll_with(fake_tvmaze, watcher, episodes_from(195))
    watcher.db.mark_watched(report.due[0].episode.id)

    remaining = poll_with(fake_tvmaze, watcher, episodes_from(195)).due
    assert [item.episode.code for item in remaining] == DUE_CODES[1:]


def test_lookback_window_keeps_old_history_out(watcher, fake_tvmaze):
    track(fake_tvmaze, watcher, 195)
    narrow = poll_with(fake_tvmaze, watcher, episodes_from(195), lookback_hours=72)
    wide = poll_with(fake_tvmaze, watcher, episodes_from(195), lookback_hours=24 * 90)

    assert [item.episode.code for item in narrow.due] == DUE_CODES
    assert all(item.episode.aired_at >= NOW - timedelta(hours=72) for item in narrow.due)
    assert len(wide.due) > len(narrow.due)


def test_episode_without_air_time_is_counted_but_never_due(watcher, fake_tvmaze):
    payload = [dict(item) for item in episodes_from(195)]
    for item in payload:
        if item["season"] == 2026 and item["number"] in (193, 194):
            item["airstamp"] = None
    track(fake_tvmaze, watcher, 195)
    report = poll_with(fake_tvmaze, watcher, payload)

    assert report.shows[0].without_time == 2
    assert [item.episode.code for item in report.due] == ["S2026E195"]


def test_one_broken_show_does_not_stop_the_others(watcher, fake_tvmaze):
    track(fake_tvmaze, watcher, 195)
    watcher.add_show_by_id(44933)
    fake_tvmaze.add("/shows/44933/episodes", {"error": "gone"}, status=404)

    report = poll_with(fake_tvmaze, watcher, episodes_from(195))

    assert [item.show.name for item in report.failed] == ["Severance"]
    assert report.shows[0].show.name == "Beyond the Gates"
    assert [item.episode.code for item in report.due] == DUE_CODES


def test_due_episodes_are_in_chronological_order(watcher, fake_tvmaze):
    track(fake_tvmaze, watcher, 195)
    report = poll_with(fake_tvmaze, watcher, episodes_from(195))
    stamps = [item.episode.aired_at for item in report.due]

    assert stamps == sorted(stamps)
    assert stamps == [
        datetime(2026, 10, day, 18, 0, tzinfo=timezone.utc) for day in (19, 20, 21)
    ]