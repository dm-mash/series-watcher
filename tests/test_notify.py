"""Telegram: деградация без токена, отметка только после успеха, тихий режим."""

from __future__ import annotations

from datetime import datetime, timezone

import httpx
import pytest

from series_watcher.config import Config
from series_watcher.db import TELEGRAM_CHANNEL, Database
from series_watcher.notify import (
    TVMAZE_CREDIT,
    Notifier,
    _keyboard,
    as_text,
    footer,
    footer_credit,
)

from conftest import FakeTelegram, load_fixture, store_show
from series_watcher.models import EpisodeRecord

NOW = datetime(2026, 10, 18, 12, 0, tzinfo=timezone.utc)
TEXT = "Severance — S01E01\n" + footer()


@pytest.fixture
def db(tmp_path) -> Database:
    database = Database(tmp_path / "test.db")
    show = store_show(database, load_fixture("show"))
    database.upsert_episodes(
        show.id,
        [EpisodeRecord(season=1, number=1, name="Good News About Hell", aired_at=NOW)],
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

def test_telegram_api_sends_inline_keyboard(db):
    from series_watcher.notify import TelegramApi

    telegram = FakeTelegram()
    api = TelegramApi(Config(telegram_bot_token="t"), httpx.Client(transport=telegram.transport()), token="t")
    api.send_message(42, "выбери", buttons=[[("Да", "yes"), ("Нет", "no")], ["Всё"]])

    payload = telegram.sent[0]
    assert payload["chat_id"] == 42
    assert payload["reply_markup"]["inline_keyboard"] == [
        [{"text": "Да", "callback_data": "yes"}, {"text": "Нет", "callback_data": "no"}],
        [{"text": "Всё", "callback_data": "Всё"}],
    ]


def test_telegram_api_reads_updates(db):
    from series_watcher.notify import TelegramApi

    class Updates(FakeTelegram):
        def transport(self):
            import httpx

            def handler(request: httpx.Request) -> httpx.Response:
                self.sent.append(request.url.path)
                return httpx.Response(
                    200, json={"ok": True, "result": [{"update_id": 7, "message": {"text": "дайджест"}}]}
                )

            return httpx.MockTransport(handler)

    fake = Updates()
    api = TelegramApi(Config(telegram_bot_token="t"), httpx.Client(transport=fake.transport()), token="t")

    updates = api.get_updates(offset=7, timeout=5)

    assert updates[0]["update_id"] == 7
    assert fake.sent[0].endswith("/getUpdates")


def test_telegram_api_without_token_is_a_clear_error(db):
    from series_watcher.notify import NotifyError, TelegramApi

    api = TelegramApi(Config(), httpx.Client(transport=FakeTelegram().transport()))
    with pytest.raises(NotifyError, match="TELEGRAM_BOT_TOKEN"):
        api.send_message(1, "привет")


def test_chat_falls_back_to_the_remembered_one(db):
    """Явный TELEGRAM_CHAT_ID важнее; иначе бот пишет туда, откуда заговорили."""
    from series_watcher.notify import Notifier

    db.set_setting("telegram_chat_id", "555")
    remembered = Notifier(Config(), db, token="t")
    assert remembered._chat == "555"

    explicit = Notifier(Config(telegram_chat_id="42"), db, token="t")
    assert explicit._chat == "42"


def test_without_any_chat_notifier_is_not_configured(db):
    from series_watcher.notify import Notifier

    assert Notifier(Config(), db, token="t").configured is False


def test_single_button_row_keeps_its_callback():
    """Раньше ряд из одной кнопки превращался в две: с текстом и с callback.

    Первая кнопка получала callback, равный своей подписи, и нажатие на неё
    ничего не делало. Проверяем payload, а не только то, что кнопки есть.
    """
    assert _keyboard([(("Не интересно: The Americans", "reject 157"),)]) == [
        [{"text": "Не интересно: The Americans", "callback_data": "reject 157"}]
    ]


def test_plain_labels_stay_shortcuts():
    assert _keyboard([["Всё", "Или нет", "Не знаю"]]) == [
        [
            {"text": "Всё", "callback_data": "Всё"},
            {"text": "Или нет", "callback_data": "Или нет"},
            {"text": "Не знаю", "callback_data": "Не знаю"},
        ]
    ]


def test_ambiguous_pair_of_labels_is_rejected():
    """Две строки в ряду — это либо две подписки, либо одна кнопка: пусть
    разработчик решит это явно, а не получит молча две кнопки в Telegram."""
    with pytest.raises(TypeError, match="одна кнопка или две подписки"):
        _keyboard([("Не интересно", "reject 157")])


def test_quiet_credit_is_a_footnote_not_a_list_item():
    """Подпись обязана быть, но читаться должна как сноска, а не как пункт списка."""
    text = footer_credit(["Отслеживаю 4:", "", "• Slow Horses — S06E06 — 22.10.2026 20:00"])
    body, _, credit_line = text.rpartition("\n")
    assert body.rstrip().endswith("20:00")
    assert credit_line == f"<i>{TVMAZE_CREDIT}</i>"


def test_quiet_credit_survives_empty_body():
    assert footer_credit([]) == TVMAZE_CREDIT
    assert footer_credit(["", ""]) == TVMAZE_CREDIT


def test_plain_text_keeps_the_link_address():
    """Ссылка в терминале не щёлкнется, поэтому адрес должен остаться рядом."""
    text = as_text('• <b>NCIS</b> <a href="https://tvmaze.com/shows/60/ncis">страница</a>')

    assert text == "• NCIS страница (https://tvmaze.com/shows/60/ncis)"
    assert "<" not in text and ">" not in text


def test_send_passes_the_markup_flag_to_telegram():
    """Разметка без флага `parse_mode` уезжает в Telegram тегами, а не вёрсткой."""
    sent = {}

    class Api:
        def send_message(self, chat_id, text, *, buttons=(), html=False):
            sent.update(text=text, html=html)
            return {"message_id": 1}

    api = Api()
    api.send_message(42, "<b>жирно</b>", html=True)

    assert sent == {"text": "<b>жирно</b>", "html": True}
