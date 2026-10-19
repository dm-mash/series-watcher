"""Модель там, где она действительно нужна: каталог и формулировки.

Четыре вещи, которые человек делает словами, а поиск по подстроке не умеет:

1. описание сериала -> название, по которому искать в каталоге TVmaze;
2. выбор из кандидатов, когда запрос неоднозначный;
3. перевод и сжатие описания эпизода до одной-двух фраз;
4. объяснение рекомендации одной фразой по фактам из ответа API.

Ни одна из них не обязательна: без модели (`None` или ошибка) вызывающий код
получает `None` и работает дальше как раньше. Факты модель не выдумывает —
ей дают то, что пришло из API.
"""

from __future__ import annotations

import re

from typing import Any, Mapping, Sequence

from .llm import LLMClient, LLMError
from .models import ShowRecord

TITLE_SCHEMA: Mapping[str, Any] = {
    "type": "object",
    "properties": {
        "title": {
            "type": "string",
            "description": "название сериала по-английски, как оно есть в каталоге; "
            "пустая строка, если описания недостаточно",
        }
    },
    "required": ["title"],
    "additionalProperties": False,
}

PICK_SCHEMA: Mapping[str, Any] = {
    "type": "object",
    "properties": {
        "id": {
            "type": ["integer", "null"],
            "description": "id выбранного сериала из списка; null, если ни один не подходит",
        },
        "confidence": {"type": "number", "minimum": 0, "maximum": 1},
    },
    "required": ["id", "confidence"],
    "additionalProperties": False,
}

TITLE_PROMPT = (
    "Пользователь описал сериал своими словами. Восстанови его название так, как оно "
    "записано в англоязычном каталоге, и верни только название. Если по описанию "
    "нельзя уверенно понять, какой это сериал, верни пустую строку: лучше пусто, "
    "чем выдуманное название."
)

PICK_PROMPT = (
    "Пользователь хочет добавить сериал. Перед тобой результаты поиска по каталогу. "
    "Выбери тот, о котором говорит пользователь. Если ни один не подходит, верни id=null."
)

SUMMARY_PROMPT = (
    "Сожми описание эпизода до одной-двух фраз на русском и без спойлеров финала. "
    "Не добавляй того, чего в описании нет. Если описания нет, верни пустую строку."
)

ONELINER_PROMPT = (
    "Дай одно предложение на русском о том, о чём этот сериал. Это перевод и сжатие "
    "описания, а не пересказ сюжета и не перечисление героев.\n"
    "Правила:\n"
    "- РОВНО ОДНО предложение, короче 20 слов;\n"
    "- без спойлеров, без имён персонажей, без оценочных прилагательных;\n"
    "- если описания нет, верни пустую строку."
)

EXPLAIN_PROMPT = (
    "Ты объясняешь, почему сериал рекомендуют человеку, который уже смотрит другое.\n"
    "Правила:\n"
    "- приведи РОВНО ОДНО предложение, короче 15 слов;\n"
    "- название сериала и жанры не повторяй — они уже напечатаны отдельно;\n"
    "- бери факт из списка причин и ничего больше: не пиши «известный», «в похожем "
    "стиле», «драматические проекты» и прочее, чего в причинах нет;\n"
    "- отвечай кратко и сухо, как подпись к фото.\n"
    "Пример причины «продюсер: Graham Yost» → «Тот же продюсер, Graham Yost.»\n"
    "Причина «жанры: Drama, Comedy» → «С теми же жанрами: Drama, Comedy.»"
)


def _client(llm: LLMClient | None) -> LLMClient | None:
    return llm if llm is not None and llm.configured else None


def title_from_description(description: str, *, llm: LLMClient | None) -> str | None:
    """«Сериал, где людям режут память» -> название для поиска по каталогу."""
    client = _client(llm)
    if client is None or not description.strip():
        return None
    try:
        data = client.complete_json(
            [
                {"role": "system", "content": TITLE_PROMPT},
                {"role": "user", "content": description},
            ],
            TITLE_SCHEMA,
            name="title",
        )
    except LLMError:
        return None
    if not isinstance(data, Mapping):
        return None
    title = data.get("title")
    return title.strip() if isinstance(title, str) and title.strip() else None


TITLE_PROMPT_TEXT = (
    "Назови сериал по описанию. Ответь только названием, как оно записано "
    "в каталоге, на языке оригинала. Без пояснений, без кавычек, без точки.\n"
    "Описание может быть обрывочным или пересказанным: ищи по тому, что в нём "
    "есть, а не по точному совпадению. Один сериал всегда ближе, чем отказ. "
    "Отвечай НЕТ только если это точно не сериал вовсе."
)


#: Размышляющие модели съедают бюджет на размышление: общего `LLM_MAX_TOKENS`
#: хватало на пустой обрывок вместо названия.
TITLE_MAX_TOKENS = 1500


def describe_title(description: str, *, llm: LLMClient | None) -> str | None:
    """Описание → название сериала. Ошибка модели наружу, а не тишина.

    Формат ответа — обычный текст, а не JSON: принудительная схема у части
    моделей возвращает пустую генерацию, и описание молча превращалось в
    «сериала нет».
    """
    client = _client(llm)
    if client is None or not description or not description.strip():
        return None
    reply = client.complete(
        [
            {"role": "system", "content": TITLE_PROMPT_TEXT},
            {"role": "user", "content": description},
        ],
        max_tokens=TITLE_MAX_TOKENS,
    )
    return _clean_title(reply.text)


def _clean_title(text: str) -> str | None:
    """Из ответа модели — только название, без обвязки и болтовни."""
    line = text.strip().splitlines()[0] if text.strip() else ""
    title = line.strip()
    junk = ('"', "'", "*", "«", "»", "\u201c", "\u201d", ".", ",", ":", ";", "\u2014", "-")
    # Обрезка в несколько проходов: точка в конце прячет за собой закрывающую
    # кавычку, и за один проход она остаётся висеть в названии.
    for _ in range(len(junk)):
        stripped = title
        for char in junk:
            stripped = stripped.strip(char)
        if stripped == title:
            break
        title = stripped
    if not title or len(title) > 80:
        return None
    if len(title.split()) > 6:
        return None
    return title


def pick_show(
    description: str,
    candidates: Sequence[ShowRecord],
    *,
    llm: LLMClient | None,
    floor: float = 0.6,
) -> ShowRecord | None:
    """Выбрать сериал из выдачи поиска, когда запрос неоднозначный."""
    client = _client(llm)
    if client is None or not candidates:
        return None
    listing = "\n".join(
        f"{show.external_id}. {show.title} "
        f"({', '.join(part for part in (show.kind, show.status, show.network) if part)})"
        for show in candidates
    )
    try:
        data = client.complete_json(
            [
                {"role": "system", "content": PICK_PROMPT},
                {"role": "user", "content": f"Запрос: {description}\n\nКандидаты:\n{listing}"},
            ],
            PICK_SCHEMA,
            name="pick",
        )
    except LLMError:
        return None
    if not isinstance(data, Mapping):
        return None
    try:
        confidence = float(data.get("confidence", 0.0))
    except (TypeError, ValueError):
        confidence = 0.0
    chosen = data.get("id")
    if chosen is None:
        return None
    if confidence < floor:
        return None
    key = str(chosen)
    return next((show for show in candidates if show.external_id == key), None)


def short_summary(text: str | None, *, llm: LLMClient | None) -> str | None:
    """Описание эпизода на русском, коротко. Без модели — `None`."""
    client = _client(llm)
    if client is None or not text or not text.strip():
        return None
    try:
        reply = client.complete(
            [
                {"role": "system", "content": SUMMARY_PROMPT},
                {"role": "user", "content": text},
            ]
        )
    except LLMError:
        return None
    cleaned = reply.text.strip().split("\n")[0].strip()
    return cleaned or None


def one_liner(title: str, summary: str | None, *, llm: LLMClient | None) -> str | None:
    """Одно предложение о самом сериале: перевод и сжатие описания TVmaze.

    Это не «почему он вам подходит» — этим занимается `explain`.
    """
    client = _client(llm)
    if client is None or not summary or not summary.strip():
        return None
    try:
        reply = client.complete(
            [
                {"role": "system", "content": ONELINER_PROMPT},
                {"role": "user", "content": f"Сериал: {title}\nОписание: {summary}"},
            ]
        )
    except LLMError:
        return None
    return _one_sentence(reply.text, limit=160)


def explain(title: str, reasons: Sequence[str], *, llm: LLMClient | None) -> str | None:
    """Одна фраза «почему это вам подойдёт» — только из перечисленных причин.

    Лимит токенов намеренно общий (`LLM_MAX_TOKENS`): у reasoning-моделей вроде
    gpt-oss часть бюджета уходит на размышление, и небольшой лимит обрывает
    ответ раньше, чем появится текст.
    """
    client = _client(llm)
    if client is None or not reasons:
        return None
    joined = "; ".join(reasons)
    try:
        reply = client.complete(
            [
                {"role": "system", "content": EXPLAIN_PROMPT},
                {"role": "user", "content": f"Сериал: {title}\nПричины: {joined}"},
            ]
        )
    except LLMError:
        return None
    return _one_sentence(reply.text)


def _one_sentence(text: str, limit: int = 140) -> str | None:
    """Первое предложение и не длиннее лимита: краткость тут не украшение."""
    cleaned = " ".join(text.split())
    for stop in (". ", "! ", "? "):
        if stop in cleaned:
            cleaned = cleaned.split(stop, 1)[0]
            break
    cleaned = cleaned.rstrip(" .!?,;:")
    if not cleaned:
        return None
    if len(cleaned) > limit:
        cleaned = cleaned[: limit - 1].rsplit(" ", 1)[0]
    return cleaned

#: Размышляющие модели съедают на ответ несколько тысяч токенов: общий
#: `LLM_MAX_TOKENS` на список кандидатов не хватает, и подбор заканчивался ничем.
RANK_MAX_TOKENS = 4000

RANK_PROMPT = (
    "Ты отбираешь сериалы из готового списка кандидатов для человека, который уже "
    "что-то смотрит.\n"
    "Правила ответа:\n"
    "- строка вида `<номер> | <причина>`: номер — это номер строки в списке "
    "кандидатов, рядом с названием;\n"
    "- причина — одна фраза НА РУССКОКОМ, до 15 слов, только из карточки кандидата;\n"
    "- не пиши «атмосферный», «легендарный» и прочие оценки: в карточке их нет;\n"
    "- больше строк, чем попросили, не пиши; выбирай только из списка."
)

#: Строка ответа: номер кандидата и причина через вертикальную черту.
_RANK_LINE = re.compile(r"^\s*(\d{1,6})\s*[|)\-.–]\s*(.+?)\s*$")


def rank_candidates(
    candidates: Sequence[ShowRecord],
    *,
    basis: Sequence[ShowRecord],
    limit: int,
    llm: LLMClient | None,
    mood: str | None = None,
) -> list[tuple[ShowRecord, str]]:
    """Попросить модель выбрать сериалы из настоящих кандидатов.

    Отличие от «посоветуй сериалы»: список кандидатов — это то, что реально идёт
    в эфире, а память модели тут не используется вовсе. Всё, что она назвала и
    чего в списке нет, отбрасывается.

    Формат ответа — обычный текст, а не JSON: принудительная схема у части
    моделей возвращает пустую генерацию, и подбор молча заканчивался ничем.
    """
    client = _client(llm)
    if client is None or not candidates:
        return []
    # Кандидаты нумеруются по порядку, а не своими id: настоящие номера сериалов
    # модель охотно подставляет по памяти, и тогда ответ не про то, что в эфире.
    listing = "\n".join(_facts(number, show) for number, show in enumerate(candidates, start=1))
    watched = "\n".join(f"- {show.title} ({', '.join(show.genres)})" for show in basis)
    try:
        reply = client.complete(
            [
                {"role": "system", "content": RANK_PROMPT},
                {
                    "role": "user",
                    "content": (
                        f"Что человек уже смотрит:\n{watched}\n\n"
                        f"Кандидаты:\n{listing}\n\n"
                        f"Выбери {limit}."
                        + (f"\nПожелание: {mood}" if mood else "")
                    ),
                },
            ],
            max_tokens=RANK_MAX_TOKENS,
        )
    except LLMError:
        # Ошибка модели не превращается в пустой список: «модель молчит» и
        # «подходящих нет» — разные вещи, и сказать надо что есть.
        raise
    by_number = {str(number): show for number, show in enumerate(candidates, start=1)}
    picked: list[tuple[ShowRecord, str]] = []
    seen: set[str] = set()
    for line in reply.text.splitlines():
        match = _RANK_LINE.match(line)
        if match is None:
            continue
        show = by_number.get(match.group(1))
        if show is None or show.external_id in seen:
            continue
        seen.add(show.external_id)
        why = _one_sentence(match.group(2), limit=120)
        if why:
            picked.append((show, why))
        if len(picked) >= limit:
            break
    return picked


def _facts(number: int, show: ShowRecord) -> str:
    """Карточка кандидата для модели: только то, что пришло из источника."""
    parts = [f"{number}. {show.title}"]
    details = [
        ", ".join(show.genres) if show.genres else "",
        f"рейтинг {show.rating}" if show.rating is not None else "",
        show.network or "",
        (show.premiered or "")[:4],
        show.status or "",
    ]
    summary = _one_sentence(show.summary or "", limit=200)
    line = f"  ({'; '.join(part for part in details if part)})"
    return f"{parts[0]}{line}\n  {summary}" if summary else f"{parts[0]}{line}"
