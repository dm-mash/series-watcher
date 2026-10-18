"""Источники «где смотреть»: свой файл, его разбор и деградация."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from series_watcher.db import Database
from series_watcher.sources import SourceSet
from series_watcher.sources.custom import CustomSource
from series_watcher.tvmaze import parse_show

from conftest import load_fixture


@pytest.fixture
def show(tmp_path):
    """Сериал ровно в том виде, в каком его видит сервис."""
    from series_watcher.schedule import TvMazeSchedule
    from series_watcher.tvmaze import parse_show

    with Database(tmp_path / "test.db") as db:
        return db.upsert_show("tvmaze", TvMazeSchedule(None)._show(parse_show(load_fixture("show"))))


def test_custom_source_without_file_is_unavailable(tmp_path, show):
    from series_watcher.sources.base import SourceError

    source = CustomSource(tmp_path / "нет-файла.json")

    assert source.available() is False
    with pytest.raises(SourceError, match="не читается"):
        source.where_to_watch(show)


def test_custom_source_reads_offers(tmp_path, show):
    path = tmp_path / "where.json"
    path.write_text(
        json.dumps(
            {
                "44933": [
                    {"provider": "Кинопоиск", "url": "https://example.tv/s/severance", "kind": "подписка"},
                    "https://example.org/severance",
                ],
                "1": [{"provider": "Чужой", "url": "https://example.com/x"}],
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    source = CustomSource(path)

    assert source.available() is True
    offers = source.where_to_watch(show)
    assert [offer.provider for offer in offers] == ["Кинопоиск", "ссылка"]
    assert offers[0].line() == "Кинопоиск — подписка"


def test_custom_source_skips_broken_json(tmp_path, show):
    from series_watcher.sources.base import SourceError

    path = tmp_path / "where.json"
    path.write_text("{не json", encoding="utf-8")

    with pytest.raises(SourceError, match="не читается"):
        CustomSource(path).where_to_watch(show)








def test_source_set_falls_back_to_the_next_source(tmp_path, show):
    class Empty:
        name = "пустой"

        def available(self) -> bool:
            return True

        def where_to_watch(self, other):
            return []

    answer = SourceSet([Empty(), CustomSource(_write(tmp_path, show))]).where_to_watch(show)

    assert answer.source == "свой"
    assert answer.offers[0].provider == "Кинопоиск"


def test_source_set_reports_silence(tmp_path, show):
    answer = SourceSet([CustomSource(tmp_path / "нет.json")]).where_to_watch(show)

    assert answer.offers == ()
    assert answer.unavailable == ("свой",)
    assert answer.lines() == ["где смотреть: не настроено"]


def test_source_set_survives_a_broken_source(tmp_path, show):
    class Broken:
        name = "сломанный"

        def available(self) -> bool:
            return True

        def where_to_watch(self, other):
            raise RuntimeError("упал")

    answer = SourceSet([Broken(), CustomSource(_write(tmp_path, show))]).where_to_watch(show)

    assert answer.source == "свой"
    assert [offer.provider for offer in answer.offers] == ["Кинопоиск"]


def _write(tmp_path: Path, show) -> Path:
    path = tmp_path / "where.json"
    path.write_text(
        json.dumps(
            {str(show.key.external_id): [{"provider": "Кинопоиск", "url": "https://example.tv/x"}]}
        ),
        encoding="utf-8",
    )
    return path

def test_custom_source_adds_links_without_losing_old(tmp_path, show):
    path = tmp_path / "where.json"
    path.write_text(json.dumps({"1": [{"provider": "Старый", "url": "https://example.com/1"}]}), encoding="utf-8")
    source = CustomSource(path)

    source.add(44933, "https://example.tv/s", provider="Кинопоиск", kind="подписка")

    saved = json.loads(path.read_text(encoding="utf-8"))
    assert saved["1"][0]["url"] == "https://example.com/1"
    assert saved["44933"] == [
        {"provider": "Кинопоиск", "url": "https://example.tv/s", "kind": "подписка"}
    ]
    assert [offer.url for offer in source.where_to_watch(show)] == ["https://example.tv/s"]


def test_custom_source_creates_the_file(tmp_path, show):
    path = tmp_path / "нет" / "where.json"

    CustomSource(path).add(44933, "https://example.tv/s")

    assert path.exists()
    assert CustomSource(path).available() is True
