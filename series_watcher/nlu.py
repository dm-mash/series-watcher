"""Разбор фразы в намерение и аргументы.

Модель отвечает строго по JSON-схеме. Если модель не настроена или ответ не
разобрался, срабатывают правила: бот продолжает отвечать, просто скромнее. Кнопки
Telegram — запасной путь, когда разбор неуверенный.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Mapping

from .llm import LLMClient, LLMError

WHAT_NEW = "what_new"
ADD_SHOW = "add_show"
WHERE_TO_WATCH = "where_to_watch"
MARK_WATCHED = "mark_watched"
UNSUBSCRIBE = "unsubscribe"
DIGEST = "digest"
LIST = "list"
ABOUT = "about"
RECOMMEND = "recommend"
UNKNOWN = "unknown"

INTENTS = (
    WHAT_NEW,
    ADD_SHOW,
    WHERE_TO_WATCH,
    MARK_WATCHED,
    UNSUBSCRIBE,
    DIGEST,
    LIST,
    ABOUT,
    RECOMMEND,
    UNKNOWN,
)

#: Порог, ниже которого разбор считается неуверенным и предлагаются кнопки.
CONFIDENCE_FLOOR = 0.6

SCHEMA: Mapping[str, Any] = {
    "type": "object",
    "properties": {
        "intent": {"type": "string", "enum": list(INTENTS)},
        "confidence": {"type": "number", "minimum": 0, "maximum": 1},
        "show": {"type": "string", "description": "название сериала, если он назван"},
        "season": {"type": ["integer", "null"], "description": "номер сезона, если назван"},
        "number": {"type": ["integer", "null"], "description": "номер эпизода, если назван"},
    },
    "required": ["intent", "confidence", "show", "season", "number"],
    "additionalProperties": False,
}

SYSTEM_PROMPT = (
    "Ты разбираешь сообщение пользователя сервиса отслеживания сериалов на русском.\n"
    "Намерения: what_new (что нового вышло), add_show (добавить сериал), "
    "where_to_watch (где смотреть), mark_watched (отметить просмотренным), "
    "unsubscribe (перестать отслеживать), digest (сводка по расписанию на "
    "несколько дней), list (показать список отслеживаемых сериалов и когда "
    "выходит следующий эпизод), recommend (порекомендовать сериал), "
    "unknown (не понял).\n"
    "Возвращай только JSON по схеме. Не выдумывай сериалов: если название не названо, "
    "верни пустую строку. confidence — насколько уверен в разборе.\n"
    "digest — это именно расписание на несколько дней вперёд, а «покажи список» и "
    "«что я отслеживаю» — это list. Если запрос ни под одно намерение не подходит, "
    "верни unknown, а не то, что похоже: подставлять соседнее намерение хуже, чем "
    "сказать, что не понял."
)

_RULES: tuple[tuple[str, re.Pattern[str]], ...] = (
    (DIGEST, re.compile(r"(дайджест|сводк|расписан\w* на|что выходит|что дальше|анонс)", re.I)),
    (
        LIST,
        re.compile(
            r"(список|мои сериалы|что я отслежива|что отслежива|какие сериалы|"
            r"что слежу|чего смотрю|покажи что)",
            re.I,
        ),
    ),
    (WHAT_NEW, re.compile(r"(что нового|новые серии|что вышл|вышел ли|новости)", re.I)),
    (RECOMMEND, re.compile(r"(посоветуй|порекоменд|что посмотреть|похожие на)", re.I)),
    (WHERE_TO_WATCH, re.compile(r"(где смотреть|где\s+его\s+стрим|где стримят|где смотр|где найти|можно посмотреть)", re.I)),
    (UNSUBSCRIBE, re.compile(r"(перестань|удали|убери|отпишись|отслеживать не надо|больше не следи)", re.I)),
    (MARK_WATCHED, re.compile(r"(посмотрел|просмотрел|отметь|уже видел|посмотрела)", re.I)),
    (
        ABOUT,
        re.compile(
            r"(что за сериал|что за шоу|что такое|расскажи (?:о|про|об)|"
            r"о чём|о чем\b|за что сериал|про что сериал)",
            re.I,
        ),
    ),
    (ADD_SHOW, re.compile(r"(добавь|отслеживай|подпишись на|хочу смотреть|следи за)", re.I)),
)

_SHOW_AFTER = re.compile(
    r"(?:добавь|отслеживай|следи за|подпишись на|хочу смотреть|похожие на|похожее на"
    r"|где смотреть|перестань следить за|удали сериал|убери сериал"
    r"|что за сериал|что за шоу|что такое|расскажи (?:о|про|об)|о чём|о чем)\s+"
    r"(?:сериал\s+)?(.{2,60})",
    re.I,
)
# «2 сезон 5 серию» и «сезон 2, серия 5» — номера стоят с обеих сторон от слова.
# Номера сезона бывают и двухзначные, и «номер года» — у ежедневных сериалов.
_SIMILAR_RE = re.compile(r"похож", re.I)
_SEASON = re.compile(r"(?:сезон|season)\s*(\d{1,4})|(\d{1,4})\s*(?:сезон|season)", re.I)
_NUMBER = re.compile(
    r"(?:серия|серию|сериал|эпизод|episode)\s*(\d{1,3})|(\d{1,3})\s*(?:серия|серию|эпизод)",
    re.I,
)


@dataclass(frozen=True)
class Intent:
    name: str = UNKNOWN
    confidence: float = 0.0
    show: str | None = None
    season: int | None = None
    number: int | None = None
    reason: str = ""

    @property
    def confident(self) -> bool:
        return self.name != UNKNOWN and self.confidence >= CONFIDENCE_FLOOR


def parse(text: str, *, client: LLMClient | None = None) -> Intent:
    """Сначала модель, при неудаче — правила. Порядок именно такой."""
    if client is not None and client.configured:
        try:
            intent = _from_model(text, client)
            if intent.name != UNKNOWN and intent.confidence >= CONFIDENCE_FLOOR:
                return intent
        except LLMError:
            pass
    return rules(text)


def _from_model(text: str, client: LLMClient) -> Intent:
    data = client.complete_json(
        [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": text},
        ],
        SCHEMA,
        name="intent",
    )
    return _intent_from_payload(data, reason="llm")


def _intent_from_payload(data: Any, *, reason: str) -> Intent:
    if not isinstance(data, Mapping):
        return Intent(reason=f"{reason}: не объект")
    name = data.get("intent")
    if name not in INTENTS:
        return Intent(reason=f"{reason}: неизвестное намерение {name!r}")
    try:
        confidence = float(data.get("confidence", 0.0))
    except (TypeError, ValueError):
        confidence = 0.0
    show = data.get("show")
    return Intent(
        name=name,
        confidence=max(0.0, min(1.0, confidence)),
        show=show.strip() if isinstance(show, str) and show.strip() else None,
        season=_as_int(data.get("season")),
        number=_as_int(data.get("number")),
        reason=reason,
    )


def rules(text: str) -> Intent:
    """Разбор без модели: по словам-маркерам и названию после глагола."""
    phrase = (text or "").strip()
    if not phrase:
        return Intent(reason="пустая реплика")
    if phrase in {"/start", "start", "старт", "начать"}:
        # Первое касание бота: показать, что уже вышло, и куда смотреть дальше.
        return Intent(name=WHAT_NEW, confidence=0.9, reason="rules")

    for name, pattern in _RULES:
        if pattern.search(phrase):
            return Intent(
                name=name,
                confidence=0.8,
                show=_show_after(phrase, name),
                season=_season(phrase),
                number=_number(phrase),
                reason="rules",
            )
    return Intent(name=UNKNOWN, confidence=0.0, reason="нет маркеров")


def _show_after(phrase: str, name: str) -> str | None:
    """Название сериала, если оно есть в реплике.

    У намерений без названия («дайджест», «что нового») пусто, а у рекомендации
    берём только после «похожие на»: «посоветуй что-нибудь» — это про весь список.
    """
    if name == RECOMMEND and not _SIMILAR_RE.search(phrase):
        return None
    if name in (DIGEST, WHAT_NEW):
        return None
    match = _SHOW_AFTER.search(phrase)
    if not match:
        return None
    value = match.group(1).strip(" ?!.,«»\"'")
    return value or None


def _season(phrase: str) -> int | None:
    return _first_group(_SEASON.search(phrase))


def _number(phrase: str) -> int | None:
    return _first_group(_NUMBER.search(phrase))


def _first_group(match: re.Match[str] | None) -> int | None:
    if match is None:
        return None
    for group in match.groups():
        if group is not None:
            return int(group)
    return None


def _as_int(value: Any) -> int | None:
    if isinstance(value, bool) or value is None:
        return None
    if isinstance(value, int):
        return value
    if isinstance(value, str) and value.strip().isdigit():
        return int(value.strip())
    return None


def buttons_for(intent: Intent) -> tuple[tuple[tuple[str, str], ...], ...]:
    """Подсказки, когда разбор неуверенный: кнопки вместо догадок.

    Подпись равна команде: нажатие отправляет её как обычную реплику.
    """
    if intent.name == RECOMMEND:
        return (_shortcuts("Что нового", "Сводка", "Порекомендуй"),)
    return (_shortcuts("Что нового", "Сводка", "Где смотреть"),)


def _shortcuts(*commands: str) -> tuple[tuple[str, str], ...]:
    return tuple((command, command) for command in commands)