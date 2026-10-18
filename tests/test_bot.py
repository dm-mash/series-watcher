"""Бот и разговорный слой: ответы на намерения и разбор обновлений Telegram."""

from __future__ import annotations

from datetime import datetime, timezone

import pytest

from series_watcher.bot import Bot, text_update
from series_watcher.config import Config
from series_watcher.db import Database
from series_watcher.llm import LLMClient
from series_watcher.notify import TelegramApi
from series_watcher.nlu import Intent
from series_watcher.service import SeriesWatcher
from series_watcher.sources import SourceSet
from series_watcher.tvmaze import TvMazeClient

from conftest import FakeTelegram, load_fixture

from test_check import NOW
from test_nlu import intent_json
from test_llm import FakeLLM


class FakeTelegramApi(TelegramApi):
    """Bot API с заготовленными обновлениями и собранными ответами."""

    def __init__(self, updates, config: Config, token: str | None = "test-token") -> None:
        self._token = token
        self.updates = list(updates)
        self.sent: list[dict] = []
        self.offsets: list[int | None] = []
        self._telegram = FakeTelegram()

    @property
    def token(self):
        return self._token

    def send_message(self, chat_id, text, *, buttons=()):
        self.sent.append({"chat_id": chat_id, "text": text, "buttons": buttons})
        return {"message_id": len(self.sent)}

    def get_updates(self, *, offset=None, timeout=25):
        self.offsets.append(offset)
        if not self.updates:
            return []
        batch, self.updates = self.updates, []
        return batch


@pytest.fixture
def frozen_now(monkeypatch) -> None:
    """Ответы бота не должны зависеть от того, какой сегодня день."""
    monkeypatch.setattr("series_watcher.service.utcnow", lambda: NOW)


@pytest.fixture
def parts(tmp_path, fake_tvmaze, sleeps):
    db = Database(tmp_path / "test.db")
    client = TvMazeClient(
        transport=fake_tvmaze.transport(), cache=db, sleep=sleeps.append, rate_limit=(0, 10.0)
    )
    config = Config(tz_name="Europe/Moscow")
    watcher = SeriesWatcher(config=config, db=db, tvmaze=client, sources=SourceSet([]))
    watcher.add_show_by_id(75194)
    yield watcher, fake_tvmaze, db
    client.close()
    db.close()


def make_bot(watcher, updates, *, llm_fake: FakeLLM | None = None, sleeps=None, token="test-token"):
    api = FakeTelegramApi(updates, watcher.config, token=token)
    llm = None
    if llm_fake is not None:
        config = Config.from_env(
            env_file=None,
            environ={"LLM_BASE_URL": "https://llm.example/v1", "LLM_MODEL": "model-1"},
        )
        llm = LLMClient(config, transport=llm_fake.transport(), sleep=(sleeps or []).append)
    return Bot(watcher.config, watcher, api, llm=llm), api


def test_bot_needs_a_token(parts):
    watcher, _, _ = parts
    bot, _ = make_bot(watcher, [], token=None)
    assert bot.usable is False


def test_what_new_answers_with_fresh_episodes(parts, frozen_now):
    watcher, fake_tvmaze, _ = parts
    bot, api = make_bot(watcher, [text_update("что нового", chat_id=7)])

    handled = bot.poll_once()

    assert len(handled) == 1
    assert handled[0].chat_id == 7
    assert "S2026E193" in api.sent[0]["text"]
    assert "20.10.2026 21:00" in api.sent[0]["text"]
    assert api.sent[0]["text"].endswith("https://www.tvmaze.com")


def test_digest_intent_returns_schedule(parts, frozen_now):
    watcher, _, _ = parts
    bot, api = make_bot(watcher, [text_update("дайджест")])

    bot.poll_once()

    assert "Ближайшие 7 дней" in api.sent[0]["text"]
    assert "Beyond the Gates" in api.sent[0]["text"]


def test_unsubscribe_removes_the_show(parts):
    watcher, _, _ = parts
    bot, api = make_bot(watcher, [text_update("перестань следить за Beyond the Gates")])

    bot.poll_once()

    assert "Больше не отслеживаю: Beyond the Gates" in api.sent[0]["text"]
    assert watcher.db.list_shows() == []


def test_mark_watched_hides_the_episode(parts):
    watcher, _, db = parts
    bot, api = make_bot(watcher, [text_update("отметь 2 сезон 193 серию")])

    bot.poll_once()

    assert "Отметил просмотренным" in api.sent[0]["text"]
    episode = next(
        item
        for item in db.list_episodes(db.list_shows()[0].id)
        if item.season == 2026 and item.number == 193
    )
    assert episode.watched_at is not None


def test_add_show_offers_candidates_as_buttons(parts, fake_tvmaze):
    watcher, _, _ = parts
    fake_tvmaze.add(
        "/search/shows?q=dark knight",
        [
            {"score": 0.8, "show": dict(load_fixture("show"), id=11, name="Dark")},
            {"score": 0.5, "show": dict(load_fixture("show"), id=12, name="Dark Matter")},
        ],
    )
    fake_tvmaze.add("/shows/11", dict(load_fixture("show"), id=11, name="Dark"))
    fake_tvmaze.add("/shows/11/episodes", load_fixture("episodes"))

    bot, api = make_bot(watcher, [text_update("добавь dark knight")])
    bot.poll_once()

    text = api.sent[0]["text"]
    assert "1. Dark" in text
    assert api.sent[0]["buttons"] == (("1. Dark", "add 11"), ("2. Dark Matter", "add 12"))


def test_where_to_watch_without_sources_is_honest(parts):
    watcher, _, _ = parts
    bot, api = make_bot(watcher, [text_update("где смотреть Beyond the Gates")])

    bot.poll_once()

    assert "источники молчат" in api.sent[0]["text"]


def test_unknown_reply_offers_buttons(parts):
    watcher, _, _ = parts
    bot, api = make_bot(watcher, [text_update("привет")])

    bot.poll_once()

    assert "Не понял" in api.sent[0]["text"]
    assert api.sent[0]["buttons"]


def test_model_intent_reaches_the_service(parts):
    watcher, _, _ = parts
    fake = FakeLLM(intent_json(intent="digest", confidence=0.95))
    bot, api = make_bot(watcher, [text_update("ну как там с расписанием")], llm_fake=fake)

    bot.poll_once()

    assert "Ближайшие 7 дней" in api.sent[0]["text"]


def test_offset_moves_forward(parts):
    watcher, _, _ = parts
    bot, api = make_bot(watcher, [text_update("дайджест", update_id=41)])

    bot.poll_once()

    assert api.offsets == [None]


def test_non_text_updates_are_skipped(parts):
    watcher, _, _ = parts
    updates = [{"update_id": 1, "message": {"chat": {"id": 1}, "text": ""}}]
    bot, api = make_bot(watcher, updates)

    assert bot.poll_once() == []
    assert api.sent == []


def test_run_once_returns_after_a_single_pass(parts):
    watcher, _, _ = parts
    bot, api = make_bot(watcher, [text_update("дайджест"), text_update("что нового")])

    bot.run(once=True)

    assert len(api.sent) == 2