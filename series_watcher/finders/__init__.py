"""Поиск раздач: интерфейс один, реализации разные.

Сервис знает только `EpisodeFinder` — как и про источник расписания. Пакет
`torrent-search-mcp` ставится отдельно; без него `build()` возвращает поиск,
который честно отвечает «искать нечем», а не роняет сервис.
"""

from __future__ import annotations

from typing import Callable

from .base import EpisodeFinder, FinderError, FinderUnavailable, TorrentHit, prune_magnet
from .torrent_search import DEFAULT_EXCLUDE, TorrentSearchFinder, build

__all__ = [
    "DEFAULT_EXCLUDE",
    "EpisodeFinder",
    "FinderError",
    "FinderUnavailable",
    "TorrentHit",
    "TorrentSearchFinder",
    "build",
    "get",
    "names",
    "prune_magnet",
    "register",
]

_REGISTRY: dict[str, Callable[..., EpisodeFinder]] = {}


def register(name: str, factory: Callable[..., EpisodeFinder]) -> None:
    _REGISTRY[name] = factory


def get(name: str = "torrent") -> EpisodeFinder:
    factory = _REGISTRY.get(name)
    if factory is None:
        raise FinderError(f"поиск «{name}» неизвестен: есть {', '.join(sorted(_REGISTRY)) or '—'}")
    return factory()


def names() -> tuple[str, ...]:
    return tuple(sorted(_REGISTRY))


register("torrent", TorrentSearchFinder)