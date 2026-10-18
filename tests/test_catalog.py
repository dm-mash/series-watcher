"""Модель там, где поиск не справляется: каталог и формулировки."""

from __future__ import annotations

from series_watcher.catalog import (
    explain,
    one_liner,
    pick_show,
    short_summary,
    title_from_description,
)
from series_watcher.config import Config
from series_watcher.schedule import TvMazeSchedule
from series_watcher.tvmaze import parse_show

from conftest import load_fixture

from test_llm import FakeLLM, reply


def shows(*ids_and_names):
    """Карточки источника — так их и видит подбор."""
    adapter = TvMazeSchedule(None)
    return [adapter._show(parse_show(dict(load_fixture("show"), id=i, name=n))) for i, n in ids_and_names]


def client(fake: FakeLLM, sleeps: list[float]):
    from series_watcher.config import Config
    from series_watcher.llm import LLMClient

    config = Config.from_env(
        env_file=None,
        environ={"LLM_BASE_URL": "https://api.groq.com/openai/v1", "LLM_MODEL": "openai/gpt-oss-120b"},
    )
    return LLMClient(config, transport=fake.transport(), sleep=sleeps.append)


def test_description_becomes_a_title(sleeps):
    fake = FakeLLM(reply('{"title": "Severance"}'))
    title = title_from_description("сериал, где людям режут память", llm=client(fake, sleeps))
    assert title == "Severance"
    assert "сериал, где людям режут память" in fake.content


def test_model_may_admit_it_does_not_know(sleeps):
    fake = FakeLLM(reply('{"title": ""}'))
    assert title_from_description("что-то про кота", llm=client(fake, sleeps)) is None


def test_without_model_nothing_is_guessed():
    assert title_from_description("описание", llm=None) is None
    assert pick_show("описание", shows((1, "Dark")), llm=None) is None
    assert short_summary("текст", llm=None) is None
    assert explain("Fargo", ["продюсер"], llm=None) is None


def test_pick_selects_from_candidates(sleeps):
    candidates = shows((11, "Dark"), (12, "Dark Matter"))
    fake = FakeLLM(reply('{"id": 12, "confidence": 0.9}'))

    chosen = pick_show("dark matter", candidates, llm=client(fake, sleeps))

    assert chosen is not None and chosen.title == "Dark Matter"


def test_pick_declines_low_confidence(sleeps):
    fake = FakeLLM(reply('{"id": 11, "confidence": 0.2}'))
    assert pick_show("что-то", shows((11, "Dark")), llm=client(fake, sleeps)) is None


def test_pick_accepts_no_match(sleeps):
    fake = FakeLLM(reply('{"id": null, "confidence": 0.9}'))
    assert pick_show("что-то", shows((11, "Dark")), llm=client(fake, sleeps)) is None


def test_summary_is_one_short_line(sleeps):
    fake = FakeLLM(reply("Марк ведёт расследование.\nВторая строка лишняя."))
    assert short_summary("Mark investigates.", llm=client(fake, sleeps)) == "Марк ведёт расследование."


def test_explain_uses_the_shared_token_budget(sleeps):
    """Маленький лимит обрывает reasoning-модель раньше, чем появится текст."""
    fake = FakeLLM(reply("Коротко: потому что тот же продюсер."))
    explain("Fargo", ["продюсер: John Cameron"], llm=client(fake, sleeps))
    assert fake.requests[0]["max_tokens"] == Config().llm_max_tokens  # не режем до 160


def test_explain_uses_given_reasons(sleeps):
    fake = FakeLLM(reply("Потому что над «Severance» работал тот же продюсер."))
    text = explain("Fargo", ["продюсер: John Cameron", "жанры: Drama"], llm=client(fake, sleeps))
    assert text == "Потому что над «Severance» работал тот же продюсер"  # точка срезана
    assert "John Cameron" in fake.content


def test_broken_json_degrades_to_none(sleeps):
    """Модель ответила не по схеме — сервис молча продолжает без неё."""
    fake = FakeLLM(reply("не знаю"))
    assert title_from_description("описание", llm=client(fake, sleeps)) is None


def test_explain_is_trimmed_to_one_sentence(sleeps):
    """Краткость тут не украшение: лишние предложения и вода отбрасываются."""
    fake = FakeLLM(reply("Тот же продюсер, Graham Yost. Кроме того, это очень хороший сериал."))
    text = explain("Silo", ["продюсер: Graham Yost"], llm=client(fake, sleeps))
    assert text == "Тот же продюсер, Graham Yost"


def test_explain_cut_to_a_readable_length(sleeps):
    fake = FakeLLM(reply("Очень длинное объяснение " * 30))
    text = explain("Silo", ["жанры: Drama"], llm=client(fake, sleeps))
    assert len(text) <= 140
    assert text.endswith("…") is False  # режем по слову, а не по символу


def test_explain_prompt_forbids_padding(sleeps):
    from series_watcher.catalog import EXPLAIN_PROMPT

    assert "РОВНО ОДНО предложение" in EXPLAIN_PROMPT
    assert "известный" in EXPLAIN_PROMPT  # прямо запрещаем


def test_one_liner_describes_the_show_itself(sleeps):
    fake = FakeLLM(reply("Советские шпионы живут в пригороде Вашингтона."))
    text = one_liner(
        "The Americans",
        "The Americans is the story of a married Soviet couple who pose as Americans.",
        llm=client(fake, sleeps),
    )
    assert text == "Советские шпионы живут в пригороде Вашингтона"
    assert "The Americans is the story" in fake.content


def test_one_liner_without_summary_or_model(sleeps):
    assert one_liner("Silo", None, llm=None) is None
    assert one_liner("Silo", "   ", llm=client(FakeLLM(), sleeps)) is None


def test_one_liner_drops_trailing_sentences(sleeps):
    fake = FakeLLM(reply("Драма про побег из бункера. В финале герой умирает."))
    assert one_liner("Silo", "текст", llm=client(fake, sleeps)) == "Драма про побег из бункера"
