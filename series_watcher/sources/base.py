"""Адаптеры источников «где смотреть».

Источник отвечает на один вопрос: где сериал сейчас смотрят. Источников может
быть несколько, и любой из них может быть недоступен — тогда об этом говорим
прямо, а не молча возвращаем пустой список.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable, Protocol, Sequence

from ..db import Show


class SourceError(RuntimeError):
    """Источник не ответил или ответил непонятно."""


@dataclass(frozen=True)
class Offer:
    """Один способ посмотреть сериал."""

    provider: str
    url: str
    kind: str | None = None  # flatrate, free, ads, rent, buy
    price: str | None = None

    def line(self) -> str:
        parts = [f"{self.provider}"]
        if self.kind:
            parts.append(self.kind)
        if self.price:
            parts.append(self.price)
        return " — ".join(parts)


class Source(Protocol):
    """Источник «где смотреть»."""

    name: str

    def available(self) -> bool:
        """Готов ли источник работать (есть ли ключ, файл)."""

    def where_to_watch(self, show: Show) -> Sequence[Offer]:
        """Предложения по сериалу. Пустой список — значит, смотреть негде."""


@dataclass(frozen=True)
class SourceAnswer:
    """Ответ на «где смотреть»: что нашли и каких источников не спросили."""

    offers: tuple[Offer, ...] = ()
    source: str | None = None
    unavailable: tuple[str, ...] = ()

    def lines(self, limit: int = 3) -> list[str]:
        if not self.offers:
            return ["где смотреть: источники молчат"]
        return [f"где смотреть: {offer.line()} — {offer.url}" for offer in self.offers[:limit]]


@dataclass
class SourceSet:
    """Набор источников: первый, кто что-то нашёл, и отвечает."""

    sources: list[Source] = field(default_factory=list)

    def __iter__(self) -> Iterable[Source]:
        return iter(self.sources)

    def available(self) -> list[Source]:
        return [source for source in self.sources if _safe_available(source)]

    def where_to_watch(self, show: Show, *, limit: int = 3) -> SourceAnswer:
        unavailable: list[str] = []
        for source in self.sources:
            if not _safe_available(source):
                unavailable.append(source.name)
                continue
            try:
                offers = [offer for offer in source.where_to_watch(show) if offer.url]
            except Exception:  # noqa: BLE001 - источник внешний, опрос он не должен ронять
                unavailable.append(source.name)
                continue
            if offers:
                return SourceAnswer(offers=tuple(offers[:limit]), source=source.name)
            unavailable.append(source.name)
        return SourceAnswer(unavailable=tuple(unavailable))


def _safe_available(source: Source) -> bool:
    try:
        return bool(source.available())
    except Exception:  # noqa: BLE001 - источник не должен ронять опрос
        return False