"""Разбор фразы в намерение и аргументы.

Модель отвечает строго по JSON-схеме. Если модель не настроена или ответ не
разобрался, срабатывают правила: бот продолжает отвечать, просто скромнее. Кнопки
Telegram — запасной путь, когда разбор неуверенный.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field, replace
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
DOWNLOAD = "download"
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
    DOWNLOAD,
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
        "limit": {
            "type": ["integer", "null"],
            "description": "сколько сериалов просят: «штук 5», «топ-3»",
        },        "mood": {
            "type": ["string", "null"],
            "description": "пожелание словами: «лёгкое на вечер»; пустая строка, если пожелания нет",
        },        "strategy": {
            "type": ["string", "null"],
            "description": "null, если не сказано; иначе model или crew",
        },


    },
    "required": ["intent", "confidence", "show", "season", "number", "limit", "mood", "strategy"],
    "additionalProperties": False,
}

SYSTEM_PROMPT = (
    "Ты разбираешь сообщение пользователя сервиса отслеживания сериалов на русском.\n"
    "Намерения: what_new (что нового вышло), add_show (добавить сериал), "
    "where_to_watch (где смотреть), mark_watched (отметить просмотренным), "
    "unsubscribe (перестать отслеживать), digest (сводка по расписанию на "
    "несколько дней), list (показать список отслеживаемых сериалов и когда "
    "выходит следующий эпизод), recommend (порекомендовать сериал; если человек описал, чего хочет — «лёгкое на вечер» — положи это в mood), "
    "download (найти раздачу серии: «где скачать S02E05» — название и код серии "
    "кладите в show), "
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
    (
        DOWNLOAD,
        re.compile(
            r"(где скачать|найди раздач|раздач\w* для|скачать серию|"
            r"(?:для|серию)\s*s\d{1,4}e\d{1,3}\b|\bs\d{1,4}e\d{1,3}\b)",
            re.I,
        ),
    ),
    (
        RECOMMEND,
        re.compile(r"(посоветуй|порекоменд|что посмотреть|похожие на|похожее на|топ[-\s]?\d)", re.I),
    ),
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
    r"|что за сериал|что за шоу|что такое|расскажи (?:о|про|об)|о чём|о чем"
    r"|где скачать|найди раздач\w*|раздач\w* для|скачать серию)\s+"
    r"(?:для\s+)?(?:сериал\s+)?(.{2,60})",
    re.I,
)
# «2 сезон 5 серию» и «сезон 2, серия 5» — номера стоят с обеих сторон от слова.
# Номера сезона бывают и двухзначные, и «номер года» — у ежедневных сериалов.
_SIMILAR_RE = re.compile(r"похож", re.I)
#: Код серии прямо в тексте: «S02E04», «s2e4». Точнее, чем разбор моделью.
_EPISODE_CODE = re.compile(r"\bs(\d{1,4})e(\d{1,3})\b", re.I)
#: «штук пять», «5 штук», «топ-3», «дай три» — просят разное число.
_COUNT_RE = re.compile(
    r"(?:штук(?:ами|ах|ам)?|топ[-\s]?)\s*(\d{1,2})|(\d{1,2})\s*штук", re.I
)
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
    limit: int | None = None
    #: Пожелание словами: описали, чего хотят, — подбор уходит модели.
    mood: str | None = None
    #: Чем подбирать, если назвали прямо: «посоветуй моделью».
    strategy: str | None = None
    reason: str = ""

    @property
    def confident(self) -> bool:
        return self.name != UNKNOWN and self.confidence >= CONFIDENCE_FLOOR


def parse(text: str, *, client: LLMClient | None = None) -> Intent:
    """Сначала модель, при неудаче — правила. Порядок именно такой.

    Но код серии, номер эпизода и «сколько штук» всегда берутся правилами, даже
    когда интент от модели: модель на «найди раздачи для MobLand S02E04» отвечает
    уверенно, но теряет `S02E04` (считает, что это часть названия). Правила на
    таких деталях точнее, а потерянный код серии потом не восстановить — человек
    получит раздачу соседней серии.
    """
    if client is not None and client.configured:
        try:
            intent = _from_model(text, client)
            if intent.name != UNKNOWN and intent.confidence >= CONFIDENCE_FLOOR:
                return _with_exact_details(text, intent)
        except LLMError:
            pass
    return rules(text)


def _with_exact_details(text: str, intent: Intent) -> Intent:
    """Подставить точные детали из текста поверх ответа модели."""
    code = _EPISODE_CODE.search(text)
    season, number = intent.season, intent.number
    if code is not None:
        season, number = int(code.group(1)), int(code.group(2))
    if season is None:
        season = _season(text)
    if number is None:
        number = _number(text)
    limit = _count(text) or intent.limit
    # Названное словом — факт: «посоветуй моделью» модель в шутку не теряет.
    # Пожелание, наоборот, лучше разбирает модель, правила — только запасной путь.
    strategy = _named_strategy(text) or intent.strategy
    mood = intent.mood or _mood(text)
    show = intent.show
    if code is not None and show and code.group(0).lower() not in show.lower():
        show = f"{show} {code.group(0)}"
    if show is None:
        # Сначала фраза целиком: описание полезнее усечённой головы, а
        # название из реплики («добавь Severance») от полного текста не страдает.
        show = _plain_query(text, intent.name) or _show_after(text, intent.name)
    return replace(
        intent,
        show=show,
        season=season,
        number=number,
        limit=limit,
        mood=mood,
        strategy=strategy,
    )


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
        limit=_as_int(data.get("limit")),
        mood=_as_text(data.get("mood")),
        strategy=_as_strategy(data.get("strategy")),
        reason=reason,
    )


#: Прямой выбор подбора словами: «посоветуй моделью», «через нейросеть».
_STRATEGY_MODEL = re.compile(
    r"\b(?:модел\w*|нейросет\w*|нейронк\w*|ии)\b", re.I
)
_STRATEGY_CREW = re.compile(
    r"\b(?:по авторству|по жанрам|как раньше|обычным подбором)\b", re.I
)
_STRATEGY_WORDS = re.compile(
    r"\b(?:модел\w*|нейросет\w*|нейронк\w*|ии|по авторству|по жанрам|как раньше|"
    r"обычным подбором)\b",
    re.I,
)


def _named_strategy(phrase: str) -> str | None:
    """`model` или `crew`, если человек назвал способ прямо; иначе пусто."""
    if _STRATEGY_MODEL.search(phrase):
        return "model"
    if _STRATEGY_CREW.search(phrase):
        return "crew"
    return None


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
                limit=_count(phrase),
                mood=_mood(phrase),
                strategy=_named_strategy(phrase),
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
    # «похожее на Slow Horses, штук 5» — название кончилось до запятой.
    value = re.split(r"[,;]|\bштук|\bтоп[-\s]|\bдай\b|\bещё\b", match.group(1))[0]
    value = value.strip(" ?!.,«»\"'")
    return value or None


def _season(phrase: str) -> int | None:
    return _first_group(_SEASON.search(phrase))


def _number(phrase: str) -> int | None:
    return _first_group(_NUMBER.search(phrase))


def _count(phrase: str) -> int | None:
    """Сколько просят: «штук 5» или «5 штук». Больше 20 — явно не про сериалы."""
    value = _first_group(_COUNT_RE.search(phrase))
    if value is None or not 1 <= value <= 20:
        return None
    return value


#: Слова-пустышки: «посоветуй что-нибудь» — это не пожелание, а болтовня.
_FILLER = (
    # Предлоги остаются от «через нейросеть»: смысл несут слова стратегии.
    "через",
    "по",
    "с",
    "помощью",
    "что-нибудь",
    "что нибудь",
    "чтото-нибудь",
    "что-то",
    "что-то-нибудь",
    "сериал",
    "сериалы",
    "что-то",
    "да",
)

_MOOD_VERB = re.compile(
    r"(посоветуй|посоветуйте|подбери|порекомендуй)\s+(?!похож)(.*)$", re.I
)


def _mood(phrase: str) -> str | None:
    """Пожелание словами, сказанное без помощи модели.

    «посоветуй что-нибудь лёгкое на вечер» — пожелание есть, «посоветуй
    что-нибудь» — нет, а «посоветуй похожее на X» это не пожелание, а указание
    на сериал: его разбирает show, а не мы.
    """
    match = _MOOD_VERB.search(phrase)
    if match is None:
        return None
    tail = match.group(2)
    # «посоветуй штук 5» — это про количество, а не про настроение.
    tail = _STRATEGY_WORDS.sub(" ", _COUNT_RE.sub(" ", tail))
    words = [word for word in re.findall(r"[\w-]+", tail, re.UNICODE) if word]
    while words and words[0].lower() in _FILLER:
        words.pop(0)
    if not words:
        return None
    if words[0].lower().startswith("похож"):
        return None
    if len(words) > 8:
        return None
    return " ".join(words)


#: Намерения, где человек точно что-то описывает, и пустой ответ модели —
#: не повод спрашивать «какой сериал?».
_QUERY_INTENTS = (ADD_SHOW, WHERE_TO_WATCH, ABOUT, DOWNLOAD)

_LEAD_VERB = re.compile(
    r"^(?:ну\s+)?(?:дай|добавь|добавьте|найди|найдите|найди-нибудь|ищи|ищите|посоветуй|"
    r"порекомендуй|подскажи|подбери|расскажи|что за|что такое|где смотреть|где найти|"
    r"где скачать|найди раздач\w*|раздач\w*)\b",
    re.I,
)


def _plain_query(text: str, name: str) -> str | None:
    """Вся фраза без ведущего глагола — когда модель не заполнила слот.

    Для описаний это единственное, что у нас есть: «добавь сериал про офис, где
    людей разделяют на части» — описание целиком, а не его усечённая голова.
    """
    if name not in _QUERY_INTENTS:
        return None
    body = _LEAD_VERB.sub("", text.strip(), count=1).strip(" ,.:;—-")
    body = re.sub(r"^(?:что-нибудь|что нибудь|сериал|сериалы|фильм|шоу)\b\s*", "", body, flags=re.I)
    return body or None


def _as_strategy(value: Any) -> str | None:
    """Стратегия из ответа модели: только те две, что есть в реестре."""
    name = value.strip().lower() if isinstance(value, str) else ""
    return name if name in ("crew", "model") else None


def _as_text(value: Any) -> str | None:
    """Строка из ответа модели: пустая значит «не названо»."""
    return value.strip() or None if isinstance(value, str) else None


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