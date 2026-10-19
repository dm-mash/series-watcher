"""Подбор моделью: только настоящие кандидаты, пожелание словами."""

from __future__ import annotations

from dataclasses import dataclass, field

import pytest

from series_watcher.catalog import rank_candidates
from series_watcher.models import ShowRecord
from series_watcher.strategies import StrategyError, get as get_strategy, names
from series_watcher.strategies.model import ModelStrategy

from test_llm import FakeLLM, reply


def show(tvmaze_id: str, title: str, **fields) -> ShowRecord:
    base = dict(
        external_id=tvmaze_id,
        title=title,
        kind="Scripted",
        status="Running",
        genres=("Drama",),
        rating=7.5,
        summary="Описание сериала.",
    )
    base.update(fields)
    return ShowRecord(**base)


class FakeLLMClient:
    """Клиент модели для стратегии: отдаёт заготовленный текст ответа."""

    def __init__(self, text: str = "", *, configured: bool = True) -> None:
        self._text = text
        self.configured = configured
        self.calls: list = []

    def complete(self, messages, *, temperature=None, max_tokens=None):
        from series_watcher.llm import Reply

        self.calls.append(messages)
        return Reply(text=self._text, payload={})


def model(**fields) -> ModelStrategy:
    return get_strategy("model")


def test_registry_knows_both_strategies():
    assert set(names()) == {"crew", "model"}
    assert model().name == "model"
    assert ModelStrategy.description


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("1 | продолжает то, что вы уже смотрите\n2 | юмор и быт", ["a", "b"]),
        ("3 | только причина", ["c"]),
        ("99 | сериала нет в списке\n1 | настоящий", ["a"]),
        ("1 | причина\n1 | повтор", ["a"]),
        ("без номера | причина", []),
        ("", []),
    ],
)
def test_only_pool_candidates_survive(text, expected):
    """Главная защита: модель не может назвать то, чего нет в эфире."""
    pool = [show("a", "Alpha"), show("b", "Beta"), show("c", "Gamma")]
    picked = rank_candidates(
        pool, basis=[show("z", "Tracked")], limit=3, llm=FakeLLMClient(text)
    )
    assert [show.external_id for show, _ in picked] == expected


def test_mood_reaches_the_prompt():
    llm = FakeLLMClient("1 | потому что")
    rank_candidates(
        [show("a", "Alpha")], basis=[], limit=1, llm=llm, mood="что-то лёгкое на вечер"
    )
    assert "лёгкое на вечер" in llm.calls[0][1]["content"]


def test_candidates_are_numbered_not_by_real_ids():
    """Номера — по порядку: настоящие id модель подставляет по памяти."""
    from series_watcher.catalog import _facts

    listing = _facts(1, show("84740", "NCIS"))
    assert listing.startswith("1. NCIS")
    assert "84740" not in listing


@pytest.fixture
def db(tmp_path):
    from series_watcher.db import Database

    database = Database(tmp_path / "model.db")
    yield database
    database.close()


def test_without_model_the_strategy_says_so(db):
    """Молча вернуть пустоту — значит соврать: стратегия говорит прямо."""
    request = _request(db, llm=FakeLLMClient(configured=False))
    with pytest.raises(StrategyError, match="не настроен"):
        model().candidates(request)


def test_empty_pool_is_silence_not_an_error(db):
    """Модель настроена, но предлагать нечего — это тишина, а не поломка."""
    request = _request(db, llm=FakeLLMClient("1 | потому что"))
    assert model().candidates(request) == []


def _request(db, *, llm, limit: int = 3, mood: str | None = None, min_rating: float | None = None):
    from series_watcher.config import Config
    from series_watcher.schedule import TvMazeSchedule
    from series_watcher.strategies import Request

    class Source:
        name = "tvmaze"

        def get_schedule(self, day=None):
            return []

    return Request(
        config=Config(),
        db=db,
        source=Source(),
        limit=limit,
        llm=llm,
        mood=mood,
        **({"min_rating": min_rating} if min_rating is not None else {}),
    )
