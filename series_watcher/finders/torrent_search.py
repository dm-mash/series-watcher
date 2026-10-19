"""Поиск раздач через torrent-search-mcp: apibay, 1337x, EZTV, SubsPlease и другие.

Пакет ставится отдельно (`pip install torrent-search-mcp`), и импортируется
лениво: без него сервис работает как обычно, просто не ищет раздачи. Ключей не
нужно — трекеры публичные.
"""

from __future__ import annotations

import asyncio
import re
from typing import Any, Sequence

from .base import FinderError, FinderUnavailable, TorrentHit, prune_magnet

#: Из-за чего отсекаем находки: они не про серию, а мусор в выдаче трекера.
_NOISE = re.compile(r"\b(software|game|dlc|repack|fitgirl|bundle|cracked|keygen)\b", re.I)
_EPISODE_CODE = re.compile(r"(?:^|[^0-9])s?(\d{1,2})[ex](\d{1,3})(?:[^0-9]|$)", re.I)
#: Какие трекеры оставляем: YTS и FitGirl — это фильмы и игры, нам не нужно.
DEFAULT_EXCLUDE = "yts.mx,fitgirl-repacks.site"


class TorrentSearchFinder:
    """Поиск раздач через библиотеку `torrent_search`."""

    name = "torrent"
    description = "раздачи с apibay, 1337x, EZTV, SubsPlease, Nyaa"

    def __init__(self, *, limit: int = 20, exclude_sources: str = DEFAULT_EXCLUDE) -> None:
        self._limit = limit
        self._exclude_sources = exclude_sources

    def available(self) -> bool:
        return _import_api() is not None

    def search(self, show: str, season: int, number: int, *, limit: int = 5) -> Sequence[TorrentHit]:
        api = _import_api()
        if api is None:
            raise FinderUnavailable("поиск раздач не установлен: pip install torrent-search-mcp")
        query = f"{show} s{season:02d}e{number:02d}".lower()
        try:
            found = asyncio.run(api.search_torrents(query, max_items=self._limit))
        except Exception as exc:  # noqa: BLE001 - трекер мог упасть как угодно
            raise FinderError(f"поиск не удался: {exc}") from exc

        hits: list[TorrentHit] = []
        seen: set[str] = set()
        for item in found:
            hit = _to_hit(item)
            if hit is None or not _fits(hit, season, number):
                continue
            key = hit.filename.lower()
            if key in seen:
                continue
            seen.add(key)
            hits.append(hit)
        # Сиды решают всё: без них раздача мёртвая, с ними — живая.
        hits.sort(key=lambda hit: -hit.seeders)
        return hits[:limit]

    def magnet(self, torrent_id: str) -> str | None:
        api = _import_api()
        if api is None:
            raise FinderUnavailable("поиск раздач не установлен: pip install torrent-search-mcp")
        try:
            magnet = asyncio.run(api.get_torrent(torrent_id))
        except Exception as exc:  # noqa: BLE001
            raise FinderError(f"магнит не отдался: {exc}") from exc
        return prune_magnet(magnet)


def _import_api() -> Any | None:
    """Ленивый импорт: без пакета сервис просто не ищет, а не падает."""
    try:
        from torrent_search import torrent_search_api
    except Exception:  # noqa: BLE001 - ImportError и любой сбой зависимостей
        return None
    return torrent_search_api


def _to_hit(item: Any) -> TorrentHit | None:
    name = (getattr(item, "filename", "") or "").strip()
    if not name or _NOISE.search(name):
        return None
    return TorrentHit(
        torrent_id=str(getattr(item, "id", "") or ""),
        filename=name,
        source=str(getattr(item, "source", "") or "?"),
        seeders=int(getattr(item, "seeders", 0) or 0),
        leechers=int(getattr(item, "leechers", 0) or 0),
        size=getattr(item, "size", None),
        page_url=getattr(item, "page_url", None),
    )


def _fits(hit: TorrentHit, season: int, number: int) -> bool:
    """Название обязано содержать именно нашу серию.

    Код сравниваем как числа: у трекеров «S2E5» и «S02E05» — одно и то же. А вот
    пакеты («S04E01-06») отсекаем: серия там входит, но человек просил одиночную.
    """
    for match in _EPISODE_CODE.finditer(hit.filename):
        try:
            if int(match.group(1)) == season and int(match.group(2)) == number:
                return not _is_pack(hit.filename)
        except ValueError:  # pragma: no cover - регзлярное выражение даёт цифры
            continue
    return False


def _is_pack(filename: str) -> bool:
    """Пакет вида `S04E01-06`: серия входит, но это не она одна."""
    return bool(re.search(r"[ex]\d{1,3}\s*[-–]\s*\d{1,3}", filename, re.I))


def build(**_kwargs: Any) -> TorrentSearchFinder:
    """Сборка поиска из настроек — как `schedule.build_source`."""
    from ..config import Config

    config = Config.from_env()
    return TorrentSearchFinder(
        limit=config.torrent_search_limit,
        exclude_sources=config.torrent_exclude_sources or DEFAULT_EXCLUDE,
    )