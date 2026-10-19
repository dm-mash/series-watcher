"""Подбор моделью: она выбирает из настоящих кандидатов и объясняет выбор.

Отличие от `crew` — не в том, что «модель умнее», а в том, что она видит
описания. Правила умеют считать общее авторство и жанры, а настроение
(«что-то лёгкое на вечер») правила не понимают вовсе.

Главное ограничение: список кандидатов — это то, что идёт в эфире сейчас, а не
то, что модель помнит. Всё, что она назвала и чего в списке нет, отбрасывается:
без такой проверки она перечислит классику, которой в эфире нет и которая уже
давно закончилась.
"""

from __future__ import annotations

from dataclasses import replace
from typing import Sequence

from ..catalog import rank_candidates
from ..db import Show
from ..llm import LLMError
from ..models import ShowRecord
from .base import Recommendation, Request, StrategyError, usable_pool

#: Сколько кандидатов показываем модели. Больше — она начинает теряться, а
#: токены растут; меньше — выборка перестаёт быть честной.
MAX_CANDIDATES = 40


class ModelStrategy:
    """Рекомендации от модели по описаниям и жанрам."""

    name = "model"
    description = "модель выбирает из того, что идёт в эфире, и объясняет почему"

    def candidates(self, request: Request) -> Sequence[Recommendation]:
        # Сначала готовность модели, потом пул: «модель не настроена» actionable,
        # а «нечего предложить» при пустом расписании — просто тишина.
        self._require_model(request)
        pool = self._pool(request)
        if not pool:
            return []

        basis = self._basis(request)
        try:
            picked = rank_candidates(
                pool, basis=basis, limit=request.limit, llm=request.llm, mood=request.mood
            )
        except LLMError as exc:
            raise StrategyError(f"модель не ответила: {exc}") from exc
        return [
            Recommendation(
                show=show,
                score=float(request.limit - position),
                reasons=(why,),
                via="модель",
            )
            for position, (show, why) in enumerate(picked)
        ]

    def cold_start(self, request: Request, *, limit: int = 3) -> Sequence[Recommendation]:
        """Пустой список: модель решает, с чего начать, по тем же карточкам."""
        self._require_model(request)
        pool = self._pool(request, min_rating=5.5)[:limit]
        if not pool:
            return []
        try:
            picked = rank_candidates(pool, basis=(), limit=limit, llm=request.llm)
        except LLMError as exc:
            raise StrategyError(f"модель не ответила: {exc}") from exc
        return [
            Recommendation(show=show, score=float(limit - position), reasons=(why,), via="модель")
            for position, (show, why) in enumerate(picked)
        ]

    @staticmethod
    def _require_model(request: Request) -> None:
        """Без модели эта стратегия не работает вовсе — говорим об этом прямо."""
        # Признак, а не isinstance: конкретный класс клиента стратегию не волнует.
        if not getattr(request.llm, "configured", False):
            raise StrategyError("подбор моделью не настроен: задайте LLM_BASE_URL и LLM_MODEL")

    def _pool(self, request: Request, *, min_rating: float | None = None) -> list[ShowRecord]:
        if min_rating is None:
            return usable_pool(request)
        relaxed = replace(request, min_rating=min_rating)
        return usable_pool(relaxed)

    @staticmethod
    def _basis(request: Request) -> list[ShowRecord]:
        """Отслеживаемое в виде карточек — их модель видит как свой вкус."""
        records: list[ShowRecord] = []
        for show in request.basis:
            if show.genres or show.summary:
                records.append(
                    ShowRecord(
                        external_id=str(show.key.external_id) if show.key else "",
                        title=show.name,
                        kind=show.kind,
                        genres=tuple(show.genres),
                        summary=show.summary,
                        rating=show.rating,
                    )
                )
        return records

