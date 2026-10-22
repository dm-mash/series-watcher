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
from series_watcher.schedule import TvMazeSchedule
from series_watcher.tvmaze import TvMazeClient

from conftest import FakeTelegram, load_fixture, pin_schedule

from test_check import NOW
from test_recommend import person_credits
from test_nlu import intent_json
from test_llm import FakeLLM, reply


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

    def send_message(self, chat_id, text, *, buttons=(), html=False):
        self.sent.append({"chat_id": chat_id, "text": text, "buttons": buttons, "html": html})
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
    watcher = SeriesWatcher(
        config=config, db=db, source=TvMazeSchedule(client), sources=SourceSet([])
    )
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
        # Как в бою: клиент достаётся и разговорному слою, и сервису — стратегии
        # нужна модель из watcher, а не из бота.
        watcher.llm = llm
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
    assert api.sent[0]["buttons"] == ((("1. Dark", "add 11"),), (("2. Dark Matter", "add 12"),))


def test_where_to_watch_without_links_says_what_to_do(parts):
    watcher, _, _ = parts
    bot, api = make_bot(watcher, [text_update("где смотреть Beyond the Gates")])

    bot.poll_once()

    assert "где смотреть: не настроено" in api.sent[0]["text"]
    assert "where --add 75194" in api.sent[0]["text"]


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

def callback_update(data: str, *, chat_id: int = 1, update_id: int = 5) -> dict:
    return {
        "update_id": update_id,
        "callback_query": {
            "id": "1",
            "data": data,
            "message": {"chat": {"id": chat_id}, "text": "Что добавить?"},
        },
    }


def test_button_adds_the_show(parts, fake_tvmaze):
    watcher, fake, _ = parts
    fake.add("/shows/44933", load_fixture("show"))
    fake.add("/shows/44933/episodes", load_fixture("episodes"))
    bot, api = make_bot(watcher, [callback_update("add 44933")])

    handled = bot.poll_once()

    assert "Добавлен: Severance" in api.sent[0]["text"]
    assert handled[0].text == "@add 44933"


def test_button_for_unknown_show_is_reported(parts):
    watcher, _, _ = parts
    bot, api = make_bot(watcher, [callback_update("add 999999")])

    bot.poll_once()

    assert "Не получилось добавить" in api.sent[0]["text"]


def test_broken_button_data_asks_for_text(parts):
    watcher, _, _ = parts
    bot, api = make_bot(watcher, [callback_update("мусор")])

    bot.poll_once()

    assert "напишите" in api.sent[0]["text"].lower()


def test_empty_list_offers_cold_start_instead_of_silence(tmp_path, fake_tvmaze, sleeps):
    db = Database(tmp_path / "test.db")
    client = TvMazeClient(
        transport=fake_tvmaze.transport(), cache=db, sleep=sleeps.append, rate_limit=(0, 10.0)
    )
    watcher = SeriesWatcher(config=Config(tz_name="Europe/Moscow"), db=db, source=TvMazeSchedule(client))
    pin_schedule(fake_tvmaze)

    bot, api = make_bot(watcher, [text_update("привет")])
    bot.poll_once()

    assert "Список отслеживаемых пока пуст" in api.sent[0]["text"]
    assert api.sent[0]["buttons"]
    assert api.sent[0]["buttons"][0][0][1].startswith("add ")
    client.close()
    db.close()


def test_chat_is_remembered_for_later_notifications(parts):
    watcher, _, db = parts
    bot, api = make_bot(watcher, [text_update("дайджест", chat_id=4242)])

    bot.poll_once()

    assert db.get_setting("telegram_chat_id") == "4242"


def test_callback_chat_is_remembered_too(parts, fake_tvmaze):
    watcher, fake, db = parts
    fake.add("/shows/44933", load_fixture("show"))
    fake.add("/shows/44933/episodes", load_fixture("episodes"))
    bot, _ = make_bot(watcher, [callback_update("add 44933", chat_id=77)])

    bot.poll_once()

    assert db.get_setting("telegram_chat_id") == "77"


def button(text: str, *, chat_id: int = 1, update_id: int = 9) -> dict:
    """Нажатие кнопки, у которой callback_data равен её подписи."""
    return {
        "update_id": update_id,
        "callback_query": {
            "id": "1",
            "data": text,
            "message": {"chat": {"id": chat_id}, "text": "кнопки"},
        },
    }


def test_suggestion_button_acts_like_a_phrase(parts, frozen_now):
    """Кнопка «Что нового» — это та же реплика, просто набранная за пользователя."""
    watcher, _, _ = parts
    bot, api = make_bot(watcher, [button("Что нового")])

    bot.poll_once()

    assert "S2026E193" in api.sent[0]["text"]


def test_digest_button_shows_the_schedule(parts):
    watcher, _, _ = parts
    bot, api = make_bot(watcher, [button("Сводка")])

    bot.poll_once()

    assert "Ближайшие 7 дней" in api.sent[0]["text"]


def test_where_button_lists_offers(parts):
    watcher, _, _ = parts
    bot, api = make_bot(watcher, [button("Где смотреть")])

    bot.poll_once()

    assert "Beyond the Gates" in api.sent[0]["text"]


def test_recommend_button_is_answered(parts, fake_tvmaze):
    watcher, fake, _ = parts
    fake.add("/shows/44933/crew", load_fixture("crew_show"))
    fake.add("/people/169453/crewcredits", load_fixture("crew_creator"))
    pin_schedule(fake)
    bot, api = make_bot(watcher, [button("Порекомендуй")])

    bot.poll_once()

    assert "Похожее на ваш список" in api.sent[0]["text"]


def test_add_button_still_adds_by_id(parts, fake_tvmaze):
    watcher, fake, db = parts
    fake.add("/shows/44933", load_fixture("show"))
    fake.add("/shows/44933/episodes", load_fixture("episodes"))
    bot, api = make_bot(watcher, [button("add 44933")])

    bot.poll_once()

    assert "Добавлен: Severance" in api.sent[0]["text"]


class FailingApi:
    """Bot API, который падает на отправке — бот обязан выжить."""

    token = "test-token"

    def __init__(self, messages: int = 1) -> None:
        self.sent = 0
        self.offset = None
        self.messages = messages

    def get_updates(self, *, offset=None, timeout=25):
        self.offset = offset
        if self.sent >= self.messages:
            return []
        return [text_update("дайджест", update_id=self.sent + 1)]

    def send_message(self, chat_id, text, *, buttons=(), html=False):
        from series_watcher.notify import NotifyError

        self.sent += 1
        raise NotifyError("Telegram не ответил: соединение оборвалось")


def test_failed_message_does_not_kill_the_bot(parts):
    watcher, _, _ = parts
    api = FailingApi()
    bot = Bot(watcher.config, watcher, api)
    reported: list[str] = []

    bot.run(once=True, on_error=reported.append)

    assert reported == ["Telegram не ответил: соединение оборвалось"]
    assert bot.errors == []  # ошибка забрана и показана, а не копится


def test_errors_are_collected_per_message(parts):
    watcher, _, _ = parts
    api = FailingApi(messages=2)
    bot = Bot(watcher.config, watcher, api)

    bot.poll_once()
    bot.poll_once()

    assert len(bot.errors) == 2

    taken = bot.take_errors()

    assert len(taken) == 2
    assert bot.errors == []  # забрали — не копится


def test_list_button_and_phrase_show_tracked_shows(parts, frozen_now):
    """«Список» и «что нового» — разные ответы, а не один и тот же дайджест."""
    watcher, _, _ = parts
    bot, api = make_bot(watcher, [text_update("а просто список отслеживаемых покажи")])

    bot.poll_once()

    text = api.sent[0]["text"]
    assert "Отслеживаю 1" in text
    assert "Ближайшие 7 дней" not in text
    assert "S2026E196" in text


def test_digest_still_gives_the_week(parts, frozen_now):
    watcher, _, _ = parts
    bot, api = make_bot(watcher, [text_update("сводка")])

    bot.poll_once()

    assert "Ближайшие 7 дней" in api.sent[0]["text"]


def test_named_show_makes_it_about_that_show(parts, fake_tvmaze, frozen_now):
    watcher, fake, _ = parts
    fake.add("/shows/45039", dict(load_fixture("show"), id=45039, name="Slow Horses"))
    fake.add("/shows/45039/episodes", load_fixture("episodes"))
    watcher.add_show_by_id(45039)
    fake.add("/shows/45039/crew", [{"type": "Director", "person": {"id": 7, "name": "Someone"}}])
    fake.add("/people/7/crewcredits", [{"type": "Director", "_links": {"show": {"href": "https://api.tvmaze.com/shows/32", "name": "Fargo"}}}])
    pin_schedule(fake, [])
    fake.add("/shows/32", dict(load_fixture("show"), id=32, name="Fargo", rating={"average": 8.6}))
    watcher.llm = None

    bot, api = make_bot(watcher, [text_update("порекомендуй похожее на slow horses")])
    bot.poll_once()

    assert "Похожее на «Slow Horses»" in api.sent[0]["text"]
    assert "Fargo" in api.sent[0]["text"]


def test_recommendation_has_track_and_reject_buttons(parts, fake_tvmaze, frozen_now):
    watcher, fake, _ = parts
    fake.add("/shows/44933/crew", load_fixture("crew_show"))
    fake.add("/people/169453/crewcredits", load_fixture("crew_creator"))
    pin_schedule(fake)

    bot, api = make_bot(watcher, [text_update("посоветуй что-нибудь")])
    bot.poll_once()

    buttons = api.sent[0]["buttons"]
    callbacks = [data for row in buttons for _, data in row]
    # Ряд из одной кнопки — это `((подпись, callback),)`: иначе Telegram
    # получает лишнюю кнопку с подписью вместо callback.
    assert all(len(row) == 1 for row in buttons)
    assert [data.split()[0] for data in callbacks[:-1]] == ["add", "reject"] * (
        len(buttons[:-1]) // 2
    )
    # Последняя кнопка переключает подбор и несёт вопрос целиком.
    assert callbacks[-1].startswith("pick model|")
    labels = [label for row in buttons for label, _ in row]
    assert any(label.startswith("Не интересно") for label in labels)
    assert any(label.startswith("Отслеживать") for label in labels)


def test_reject_button_removes_the_candidate(parts):
    watcher, _, db = parts
    bot, api = make_bot(watcher, [button("reject 44933")])

    bot.poll_once()

    assert "больше не предлагаю" in api.sent[0]["text"]
    assert "tvmaze:44933" in db.rejected_ids()


def test_about_answers_with_a_card(parts, fake_tvmaze):
    watcher, fake, _ = parts
    watcher.add_show_by_id(44933)
    bot, api = make_bot(watcher, [text_update("что за сериал Severance")])

    bot.poll_once()

    text = api.sent[0]["text"]
    assert "Severance" in text
    assert "Рейтинг TVmaze" in text
    assert "tvmaze.com/shows/44933" in text
    assert "Данные: TVmaze" in text


def test_about_unknown_show(parts):
    watcher, fake, _ = parts
    fake.add("/search/shows?q=zzzqqq", [])
    bot, api = make_bot(watcher, [text_update("что за сериал zzzqqq")])

    bot.poll_once()

    assert "не нашёл ни в списке, ни в каталоге" in api.sent[0]["text"]
    assert "Данные: TVmaze" not in api.sent[0]["text"]


def test_about_offers_to_add_untracked_show(parts, fake_tvmaze):
    watcher, fake, _ = parts
    fake.add("/search/shows?q=Silo", [{"score": 0.9, "show": dict(load_fixture("show"), id=38052, name="Silo")}])
    fake.add("/shows/38052", dict(load_fixture("show"), id=38052, name="Silo"))
    bot, api = make_bot(watcher, [text_update("расскажи про Silo")])

    bot.poll_once()

    assert "Ещё не отслеживается" in api.sent[0]["text"]
    assert api.sent[0]["buttons"] == ((("Отслеживать: Silo", "add 38052"),),)


def test_about_without_name_asks(parts):
    """Намерение поняли, а сериал не назвали — спрашиваем, а не молчим."""
    watcher, _, _ = parts
    bot, api = make_bot(watcher, [text_update("расскажи о")])

    bot.poll_once()

    assert "О каком сериале рассказать" in api.sent[0]["text"]


def test_asked_count_is_honoured(parts, fake_tvmaze):
    from conftest import pin_schedule

    watcher, fake, _ = parts
    watcher.add_show_by_id(75194)
    pin_schedule(fake)
    bot, api = make_bot(watcher, [text_update("посоветуй топ-3")])

    bot.poll_once()

    lines = [line for line in api.sent[0]["text"].splitlines() if line.startswith("•")]
    assert len(lines) <= 3


def test_short_answer_says_how_many_found(parts, fake_tvmaze):
    """Просили пять, нашлось два — говорим об этом прямо, а не молча сокращаем."""
    from conftest import pin_schedule

    watcher, fake, _ = parts
    watcher.add_show_by_id(75194)
    pin_schedule(fake)
    bot, api = make_bot(watcher, [text_update("посоветуй штук 5")])

    bot.poll_once()

    assert "просили 5" in api.sent[0]["text"]


def test_tracking_button_subscribes_to_a_recommendation(parts, fake_tvmaze):
    """Подписаться можно прямо из подборки: кнопка «Отслеживать» шлёт `add <id>`."""
    watcher, fake, _ = parts
    fake.add("/shows/44933/crew", load_fixture("crew_show"))
    fake.add("/people/169453/crewcredits", load_fixture("crew_creator"))
    pin_schedule(fake)

    bot, api = make_bot(watcher, [text_update("посоветуй что-нибудь")])
    bot.poll_once()

    track = [row[0] for row in api.sent[0]["buttons"] if row[0][0].startswith("Отслеживать")]
    assert track, "у рекомендации должна быть кнопка «Отслеживать»"
    label, callback = track[0]
    wanted = callback.split()[1]

    fake.add("/shows/" + wanted, dict(load_fixture("show"), id=int(wanted), name=label.split(": ", 1)[1]))
    fake.add(f"/shows/{wanted}/episodes", load_fixture("episodes"))
    before = [show.name for show in watcher.db.list_shows()]

    bot, api = make_bot(watcher, [callback_update(callback)])
    bot.poll_once()

    assert [show.name for show in watcher.db.list_shows()] == [*before, label.split(": ", 1)[1]]


def test_markup_reaches_telegram_unescaped_by_the_bot(parts, fake_tvmaze):
    """Бот отправляет разметку как разметку: теги не должны попасть в текст ответа."""
    from series_watcher.notify import esc

    watcher, fake, _ = parts
    bot, api = make_bot(watcher, [text_update("что нового")])
    bot.poll_once()
    # «что нового» — простой текст, разметки нет.
    assert api.sent[0]["html"] is False

    watcher.add_show_by_id(75194)
    pin_schedule(fake)
    bot, api = make_bot(watcher, [text_update("список")])
    bot.poll_once()
    assert api.sent[0]["html"] is True
    assert "<b>" in api.sent[0]["text"]
    assert esc("") == ""


def test_html_is_escaped_in_buttons_too():
    """Подпись кнопки с кавычками не должна ломать разметку в тексте."""
    from series_watcher.notify import esc

    assert esc('say "hi" & <bye>') == 'say &quot;hi&quot; &amp; &lt;bye&gt;'


def test_strategy_can_be_named_in_the_phrase(parts, fake_tvmaze):
    """«посоветуй моделью» и «посоветуй по авторству» выбирают способ подбора."""
    from series_watcher.nlu import rules

    assert rules("посоветуй моделью").strategy == "model"
    assert rules("посоветуй через нейросеть").strategy == "model"
    assert rules("посоветуй по авторству").strategy == "crew"
    assert rules("посоветуй моделью похожее на slow horses").mood is None


def test_a_count_is_not_a_mood(parts):
    """«штук 5» — это про количество: уводить подбор в модель нельзя."""
    from series_watcher.nlu import rules

    intent = rules("посоветуй штук 5")
    assert (intent.limit, intent.mood, intent.strategy) == (5, None, None)


def test_switch_button_carries_the_question(parts, fake_tvmaze, frozen_now):
    """Кнопка «по-другому» несёт весь вопрос, а не только способ.

    Фразой «посоветуй по авторству» вопрос терялся: человек нажимал «по-другому»
    и получал общий подбор вместо «того же, но иначе».
    """
    watcher, fake, _ = parts
    fake.add("/shows/44933/crew", load_fixture("crew_show"))
    fake.add("/people/169453/crewcredits", load_fixture("crew_creator"))
    pin_schedule(fake)

    bot, api = make_bot(watcher, [text_update("посоветуй штук 3")])
    bot.poll_once()
    payload = api.sent[0]["buttons"][-1][0][1]

    assert payload.startswith("pick model|")
    strategy, mood, show_name, limit = payload[len("pick ") :].split("|")
    assert strategy == "model"
    assert limit == "3"
    assert mood == "" and show_name == ""


def test_repick_button_answers_the_same_question(parts, fake_tvmaze, frozen_now):
    """Нажатие переключает способ и не теряет пожелание и количество."""
    watcher, fake, _ = parts
    fake.add("/shows/44933/crew", load_fixture("crew_show"))
    fake.add("/people/169453/crewcredits", load_fixture("crew_creator"))
    pin_schedule(fake)

    bot, api = make_bot(watcher, [text_update("посоветуй штук 3")])
    bot.poll_once()
    payload = api.sent[0]["buttons"][-1][0][1]

    fake_llm = FakeLLM(
        reply('{"picks": [{"id": 1, "why": "продолжает то, что вы смотрите"}]}')
    )
    bot, api = make_bot(watcher, [callback_update(payload)], llm_fake=fake_llm)
    bot.poll_once()

    # Ответ модели виден по промпту: вопрос дошёл до модели, а не ушёл в правила.

    prompt = " ".join(
        message["content"] for message in fake_llm.requests[0]["messages"]
    )
    assert "Выбери 3" in prompt


def test_crew_switch_says_it_ignores_the_mood(parts, fake_tvmaze, frozen_now):
    """Пожелание — язык модели. Переключившись на правила, надо об этом сказать."""
    watcher, fake, _ = parts
    fake.add("/shows/44933/crew", load_fixture("crew_show"))
    fake.add("/people/169453/crewcredits", load_fixture("crew_creator"))
    pin_schedule(fake)

    reply = watcher.repick_by_callback("pick crew|комедийное||")

    assert "правила не учитывают" in reply.text


def test_digest_reply_carries_markup(parts, fake_tvmaze):
    """«дайджест» в чате должен уходить разметкой, а не тегами в тексте.

    Без флага разметки Telegram показывает `<b>` как обычные символы — сводка
    приходила простынёй с тегами вместо выделений.
    """
    watcher, fake, _ = parts
    pin_schedule(fake)

    bot, api = make_bot(watcher, [text_update("дайджест")])

    bot.poll_once()

    assert api.sent[0]["html"] is True
    assert "<b>" in api.sent[0]["text"]
