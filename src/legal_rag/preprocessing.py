from __future__ import annotations

import re
from functools import lru_cache

GENERIC_TOKEN_RE = re.compile(r"[а-яёa-z0-9]+", re.IGNORECASE)
ARTICLE_REF_RE = re.compile(
    r"\b(?:ст\.?|стать[яи]|стат[ьею])\s*(\d+(?:\.\d+)*(?:-\d+)?)",
    re.IGNORECASE,
)
LAW_REF_RE = re.compile(
    r"\b(?:фз|федераль\w+\s+закон)\s*(?:от\s+[^№]{0,40})?(?:№\s*|-)?(\d+(?:-[а-яёa-z0-9]+)?)?",
    re.IGNORECASE,
)
GK_RF_RE = re.compile(
    r"\b(?:граждан\w+\s+кодекс\w*\s+российской\s+федерации|гк\s*рф|гк\s*российской\s+федерации)\b",
    re.IGNORECASE,
)
DATE_NUMERIC_RE = re.compile(r"\b(\d{1,2})\.(\d{1,2})\.(\d{2,4})\b")
DATE_TEXTUAL_RE = re.compile(
    r"\b(\d{1,2})\s+"
    r"(января|февраля|марта|апреля|мая|июня|июля|августа|сентября|октября|ноября|декабря)"
    r"\s+(\d{4})\s*(?:г\.?|года)?\b",
    re.IGNORECASE,
)
CASE_NUMBER_RE = re.compile(
    r"(?:(?:дел[оа]|uid|уид)\s*)?(?:№|n)\s*([a-zа-я0-9]+(?:[-/][a-zа-я0-9]+)+)",
    re.IGNORECASE,
)
UID_RE = re.compile(
    r"\b(?:uid|уид)\s*([a-zа-я0-9]+(?:[-/][a-zа-я0-9]+)+)\b",
    re.IGNORECASE,
)
LEGAL_NUMERIC_REF_RE = re.compile(r"\b\d+(?:\.\d+)+(?:-\d+)?\b")

TEXTUAL_MONTHS = {
    "января": "01",
    "февраля": "02",
    "марта": "03",
    "апреля": "04",
    "мая": "05",
    "июня": "06",
    "июля": "07",
    "августа": "08",
    "сентября": "09",
    "октября": "10",
    "ноября": "11",
    "декабря": "12",
}

SHORT_LEGAL_TOKENS = {
    "гк",
    "рф",
    "фз",
    "нпф",
    "опс",
    "пфр",
    "снилс",
    "ипк",
    "мфц",
    "uid",
    "уид",
}
SPECIAL_TOKEN_PREFIXES = ("article_", "law_", "code_", "date_", "case_")

RU_STOPWORDS = set(
    "и в во не что он на я с со как а то все она так его но да ты к у же вы за бы по только ее мне было вот "
    "от меня еще нет о из ему когда даже ну ли если уже или ни быть был него до вас уж вам ведь там потом "
    "себя ничего ей может они тут где есть надо ней для мы тебя их чем была сам без чего раз тоже себе под "
    "будет тогда кто этот того потому этого какой ним здесь этом один мой тем чтобы нее были куда зачем всех "
    "при два об другой хоть после над больше тот через эти нас про всего них какая много три эту перед лучше "
    "том такой им более всю между"
    .split()
)


def normalize_text(text: object) -> str:
    normalized = re.sub(r"\s+", " ", str(text)).strip().lower()
    return normalized


def _normalize_year(raw_year: str) -> str:
    year = raw_year.strip()
    if len(year) == 2:
        return f"20{year}" if int(year) < 50 else f"19{year}"
    return year.zfill(4)


def _collect_special_tokens(normalized_text: str) -> list[str]:
    tokens: list[str] = []

    for match in ARTICLE_REF_RE.finditer(normalized_text):
        article_ref = match.group(1)
        tokens.extend(("статья", f"article_{article_ref}"))

    for match in LAW_REF_RE.finditer(normalized_text):
        law_number = match.group(1)
        tokens.append("фз")
        if law_number:
            tokens.append(f"law_{law_number}")

    for _ in GK_RF_RE.finditer(normalized_text):
        tokens.extend(("гк", "рф", "code_гк_рф"))

    for match in DATE_NUMERIC_RE.finditer(normalized_text):
        day, month, year = match.groups()
        tokens.append(f"date_{_normalize_year(year)}-{month.zfill(2)}-{day.zfill(2)}")

    for match in DATE_TEXTUAL_RE.finditer(normalized_text):
        day, month_name, year = match.groups()
        month = TEXTUAL_MONTHS[month_name.lower()]
        tokens.append(f"date_{_normalize_year(year)}-{month}-{day.zfill(2)}")

    for match in CASE_NUMBER_RE.finditer(normalized_text):
        case_number = re.sub(r"\s+", "", match.group(1).lower())
        tokens.append(f"case_{case_number}")

    for match in UID_RE.finditer(normalized_text):
        uid_number = re.sub(r"\s+", "", match.group(1).lower())
        tokens.append(f"uid_{uid_number}")

    for match in LEGAL_NUMERIC_REF_RE.finditer(normalized_text):
        tokens.append(match.group(0))

    return tokens


def _should_keep_generic_token(token: str) -> bool:
    if token in SHORT_LEGAL_TOKENS:
        return True
    return len(token) > 2 and token not in RU_STOPWORDS


def tokenize(text: object) -> list[str]:
    normalized = normalize_text(text)
    generic_tokens = [
        token
        for token in GENERIC_TOKEN_RE.findall(normalized)
        if _should_keep_generic_token(token)
    ]
    return generic_tokens + _collect_special_tokens(normalized)


@lru_cache(maxsize=1)
def get_morph_analyzer():
    from pymorphy3 import MorphAnalyzer

    return MorphAnalyzer()


def lemmatize_tokens(tokens: list[str]) -> list[str]:
    analyzer = get_morph_analyzer()
    lemmas: list[str] = []
    for token in tokens:
        if (
            token in SHORT_LEGAL_TOKENS
            or any(token.startswith(prefix) for prefix in SPECIAL_TOKEN_PREFIXES)
            or any(character.isdigit() for character in token)
        ):
            lemmas.append(token)
            continue
        lemmas.append(analyzer.parse(token)[0].normal_form)
    return lemmas


def tokenize_with_options(
    text: object,
    *,
    use_lemmas: bool = False,
) -> list[str]:
    tokens = tokenize(text)
    if not use_lemmas:
        return tokens
    return lemmatize_tokens(tokens)
