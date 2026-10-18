"""Telegram: деградация без токена, отметка только после успеха, тихий режим."""

from __future__ import annotations

from datetime import datetime, timezone

import pytest

from series_watcher.config import Config
from series_watcher.db import TELEGRAM_CHANNEL, Database
from series_watcher.notify import Notifier, footer

from conftest import FakeTelegram, load_fixture
from series_watcher.tvmaze import parse_episode

TEXT = "Severance — S01E01\n" + footer()


@pytest.fixture
def db(tmp_path) -> Database:
    database = Database(tmp_path / "test.db")
    from series_watcher.tvmaze import parse_show

    payload = dict(load_fixture("show"))
    show = database.upsert_show(parse_show(payload))
    database.upsert_episodes(
        show.id,
        [parse_episode(item, 44933) for item in load_fixture("episodes")[:1]],
    )
    yield database
    database.close()


def notifier(db: Database, telegram: FakeTelegram, **kwargs) -> Notifier:
    config = Config(**kwargs)
    return Notifier(
        config,
        db,
        transport=telegram.transport(),
        token=kwargs.get("telegram_bot_token"),
        chat_id=kwargs.get("telegram_chat_id"),
    )


def test_without_token_nothing_is_sent(db):
    telegram = FakeTelegram()
    sender = notifier(db, telegram)

    assert sender.configured is False
    result = sender.send(TEXT)

    assert result.sent is False
    assert "TELEGRAM_BOT_TOKEN" in result.detail
    assert telegram.sent == []


def test_message_carries_tvmave_attribution(db):
    telegram = FakeTelegram()
    notifier(db, telegram, telegram_bot_token="t", telegram_chat_id="42").send(TEXT)

    assert telegram.sent[0]["text"].endswith("Данные: TVmaze (CC BY-SA 4.0), https://www.tvmaze.com")
    assert telegram.sent[0]["chat_id"] == "42"


def test_episode_is_marked_only_after_success(db):
    episode_id = db.list_episodes(db.list_shows()[0].id)[0].id

    telegram = FakeTelegram(status=500)
    sender = notifier(db, telegram, telegram_bot_token="t", telegram_chat_id="42")

    result = sender.send_episode(episode_id, TEXT)

    assert result.sent is False
    assert db.is_sent(episode_id, TELEGRAM_CHANNEL) is False

    telegram.status = 200
    assert sender.send_episode(episode_id, TEXT).sent is True
    assert db.is_sent(episode_id, TELEGRAM_CHANNEL) is True


def test_telegram_refusal_is_reported(db):
    telegram = FakeTelegram(ok=False, description="chat not found")
    sender = notifier(db, telegram, telegram_bot_token="t", telegram_chat_id="42")
    episode_id = db.list_episodes(db.list_shows()[0].id)[0].id

    result = sender.send_episode(episode_id, TEXT)

    assert result.sent is False
    assert "chat not found" in result.detail
    assert db.is_sent(episode_id, TELEGRAM_CHANNEL) is False


def test_second_call_does_not_send_again(db):
    telegram = FakeTelegram()
    sender = notifier(db, telegram, telegram_bot_token="t", telegram_chat_id="42")

    episode_id = db.list_episodes(db.list_shows()[0].id)[0].id

    assert sender.send_episode(episode_id, TEXT).sent is True
    again = sender.send_episode(episode_id, TEXT)

    assert again.sent is False
    assert again.detail == "уже отправлено"
    assert len(telegram.sent) == 1


def test_quiet_hours_cover_the_night():
    from series_watcher.config import in_quiet_hours

    night = (23, 8)
    assert in_quiet_hours(23, night) is True
    assert in_quiet_hours(3, night) is True
    assert in_quiet_hours(8, night) is False
    assert in_quiet_hours(12, night) is False
    assert in_quiet_hours(12, None) is False


def test_quiet_hours_can_be_switched_off():
    config = Config(notify_quiet_hours=None)
    sender = Notifier(config, Database(":memory:"))
    assert sender.is_quiet(datetime(2026, 10, 20, 3, 0, tzinfo=timezone.utc)) is False


def test_is_quiet_uses_user_timezone():
    moment = datetime(2026, 10, 19, 21, 30, tzinfo=timezone.utc)  # 00:30 в Москве
    sender = Notifier(Config(tz_name="Europe/Moscow"), Database(":memory:"))

    assert sender.is_quiet(moment) is True
    assert Notifier(Config(tz_name="UTC"), Database(":memory:")).is_quiet(moment) is False


def test_quiet_hours_parsing():
    from series_watcher.config import Config, ConfigError

    assert Config.from_env(env_file=None, environ={"QUIET_HOURS": "9-17"}).notify_quiet_hours == (9, 17)
    assert Config.from_env(env_file=None, environ={"QUIET_HOURS": ""}).notify_quiet_hours is None
    assert Config.from_env(env_file=None, environ={}).notify_quiet_hours == (23, 8)
    with pytest.raises(ConfigError, match="QUIET_HOURS"):
        Config.from_env(env_file=None, environ={"QUIET_HOURS": "ночь"})
    with pytest.raises(ConfigError, match="QUIET_HOURS"):
        Config.from_env(env_file=None, environ={"QUIET_HOURS": "9-99"})