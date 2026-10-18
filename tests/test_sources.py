"""Источники «где смотреть»: свой файл, JustWatch без ключа, деградация."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from series_watcher.db import Database
from series_watcher.sources import SourceSet
from series_watcher.sources.custom import CustomSource
from series_watcher.sources.justwatch import JustWatchSource
from series_watcher.tvmaze import parse_show

from conftest import FakeGraphQL, load_fixture

PROBES = load_fixture("justwatch_probe")


@pytest.fixture
def show(tmp_path):
    """Сериал ровно в том виде, в каком его видит сервис."""
    from series_watcher.tvmaze import parse_show

    with Database(tmp_path / "test.db") as db:
        return db.upsert_show(parse_show(load_fixture("show")))


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


def test_justwatch_without_key_makes_no_request(show):
    fake = FakeGraphQL({"data": {}})
    source = JustWatchSource(None, transport=fake.transport())

    assert source.available() is False
    assert fake.queries == []

    from series_watcher.sources.base import SourceError

    with pytest.raises(SourceError, match="JUSTWATCH_API_KEY"):
        source.where_to_watch(show)


def test_justwatch_reports_api_error(show):
    """Ответ 422 на неверный запрос — реальный, сохранён при разведке схемы."""
    probe = PROBES["title_list_required_args"]
    fake = FakeGraphQL(probe["body"], status=probe["status"])
    source = JustWatchSource("key", transport=fake.transport())

    from series_watcher.sources.base import SourceError

    with pytest.raises(SourceError, match="does not exist in"):
        source.where_to_watch(show)


def test_justwatch_empty_search_gives_no_offers(show):
    """Пустые `edges` — тоже настоящий ответ: сериала в каталоге не нашлось."""
    probe = PROBES["search_empty_edges"]
    fake = FakeGraphQL(probe["body"], status=probe["status"])
    source = JustWatchSource("key", transport=fake.transport())

    assert list(source.where_to_watch(show)) == []
    assert fake.queries[0]["variables"] == {"source": "Severance", "country": "RU"}


def test_justwatch_parses_offers(show):
    """Форма предложений живым запросом не подтверждена: ключа нет."""
    fake = FakeGraphQL(
        {
            "data": {
                "searchTitles": {"edges": [{"node": {"id": "show_1"}}]},
                "nodes": [
                    {
                        "__typename": "Show",
                        "offers": [
                            {
                                "monetizationType": "FLATRATE",
                                "url": {"standardWebUrl": "https://www.justwatch.com/ru/сериал"},
                            },
                            {"monetizationType": "FREE", "url": {}},
                        ],
                    }
                ],
            }
        }
    )
    source = JustWatchSource("key", transport=fake.transport())

    offers = list(source.where_to_watch(show))

    assert len(offers) == 1
    assert offers[0].url.endswith("/сериал")
    assert offers[0].kind == "подписка"


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
    assert answer.lines() == ["где смотреть: источники молчат"]


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
            {str(show.tvmaze_id): [{"provider": "Кинопоиск", "url": "https://example.tv/x"}]}
        ),
        encoding="utf-8",
    )
    return path