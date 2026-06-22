from __future__ import annotations

import re

TOKEN_RE = re.compile(r"[а-яёa-z0-9]+", re.IGNORECASE)

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


def tokenize(text: object) -> list[str]:
    normalized = normalize_text(text)
    return [
        token
        for token in TOKEN_RE.findall(normalized)
        if len(token) > 2 and token not in RU_STOPWORDS
    ]
