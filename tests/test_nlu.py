"""Разбор фразы: модель, правила и запасной путь."""

from __future__ import annotations

import pytest

from series_watcher.config import Config
from series_watcher.llm import LLMClient
from series_watcher.nlu import (
    ADD_SHOW,
    CONFIDENCE_FLOOR,
    DIGEST,
    LIST,
    MARK_WATCHED,
    RECOMMEND,
    UNKNOWN,
    UNSUBSCRIBE,
    WHAT_NEW,
    WHERE_TO_WATCH,
    buttons_for,
    parse,
    rules,
)

from test_llm import FakeLLM, reply


def client(fake: FakeLLM, sleeps: list[float]) -> LLMClient:
    config = Config.from_env(
        env_file=None,
        environ={"LLM_BASE_URL": "https://llm.example/v1", "LLM_MODEL": "model-1"},
    )
    return LLMClient(config, transport=fake.transport(), sleep=sleeps.append)


def intent_json(**fields) -> str:
    payload = {
        "intent": UNKNOWN,
        "confidence": 0.9,
        "show": "",
        "season": None,
        "number": None,
        **fields,
    }
    return reply(__import__("json").dumps(payload, ensure_ascii=False))


@pytest.mark.parametrize(
    ("phrase", "expected"),
    [
        ("что нового?", WHAT_NEW),
        ("что вышло за неделю", WHAT_NEW),
        ("дайджест на неделю", DIGEST),
        ("где смотреть Severance", WHERE_TO_WATCH),
        ("где его стримят", WHERE_TO_WATCH),
        ("перестань следить за Severance", UNSUBSCRIBE),
        ("удали сериал Dark Angel", UNSUBSCRIBE),
        ("отметь 2 сезон 5 серию", MARK_WATCHED),
        ("посоветуй похожие на Severance", RECOMMEND),
        ("добавь The Office", ADD_SHOW),
        ("привет, как дела", UNKNOWN),
        ("", UNKNOWN),
    ],
)
def test_rules_cover_the_intents(phrase, expected):
    assert rules(phrase).name == expected


def test_rules_pick_up_show_and_episode_numbers():
    intent = rules("добавь Severance")
    assert (intent.show, intent.season, intent.number) == ("Severance", None, None)

    watched = rules("отметь 2 сезон 5 серию")
    assert (watched.season, watched.number) == (2, 5)


def test_rules_do_not_invent_a_show_name():
    assert rules("что нового").show is None
    assert rules("дайджест").show is None


def test_model_intent_wins_when_confident(sleeps):
    fake = FakeLLM(
        intent_json(intent=WHERE_TO_WATCH, confidence=0.95, show="Безупречный Вес")
    )
    intent = parse("ну там же на кинопоиске", client=client(fake, sleeps))

    assert (intent.name, intent.show, intent.confidence) == (
        WHERE_TO_WATCH,
        "Безупречный Вес",
        0.95,
    )


def test_uncertain_model_falls_back_to_rules(sleeps):
    fake = FakeLLM(intent_json(intent=DIGEST, confidence=0.2))
    intent = parse("ну как там с сериалами", client=client(fake, sleeps))

    assert intent.name == UNKNOWN
    assert intent.reason == "нет маркеров"


def test_broken_model_answer_falls_back_to_rules(sleeps):
    fake = FakeLLM(reply("не знаю"))
    assert parse("что нового", client=client(fake, sleeps)).name == WHAT_NEW


def test_unknown_intent_from_model_is_not_trusted(sleeps):
    fake = FakeLLM(intent_json(intent="погода", confidence=1.0))
    intent = parse("добавь Severance", client=client(fake, sleeps))

    assert intent.name == ADD_SHOW
    assert intent.reason == "rules"


def test_without_model_rules_work(sleeps):
    intent = parse("добавь Severance", client=client(FakeLLM(), sleeps))

    assert intent.name == ADD_SHOW
    assert intent.reason == "rules"


def test_schema_is_sent_with_every_request(sleeps):
    from series_watcher.nlu import SCHEMA

    fake = FakeLLM(intent_json(intent=DIGEST))
    parse("ну как там", client=client(fake, sleeps))

    sent = fake.requests[0]["response_format"]["json_schema"]
    assert sent["name"] == "intent"
    assert sent["schema"] == SCHEMA
    assert "what_new" in sent["schema"]["properties"]["intent"]["enum"]


def test_buttons_are_offered_when_unsure():
    assert buttons_for(rules("привет"))
    assert buttons_for(rules("посоветуй")) == (("Что нового", "Сводка", "Порекомендуй"),)


def test_confidence_floor_is_respected():
    assert rules("что нового").confident is True
    assert rules("привет").confident is False
    assert CONFIDENCE_FLOOR == 0.6

def test_start_command_is_understood():
    assert rules("/start").name == WHAT_NEW
    assert parse("старт").name == WHAT_NEW


@pytest.mark.parametrize(
    "phrase",
    ["список", "а просто список отслеживаемых покажи", "что я отслеживаю", "какие сериалы"],
)
def test_list_request_is_not_digest(phrase):
    """«Покажи список» — это list, а не ближайший по смыслу digest."""
    assert rules(phrase).name == LIST


def test_digest_is_still_digest():
    assert rules("сводка").name == DIGEST
    assert rules("дайджест на неделю").name == DIGEST


def test_prompt_tells_the_model_when_to_say_unknown():
    from series_watcher.nlu import SYSTEM_PROMPT

    assert "list" in SYSTEM_PROMPT
    assert "unknown" in SYSTEM_PROMPT
    assert "подставлять соседнее намерение хуже" in SYSTEM_PROMPT
