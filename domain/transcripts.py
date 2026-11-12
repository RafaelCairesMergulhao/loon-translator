from __future__ import annotations

import unicodedata
from collections import deque

_PUNCTUATION = ".,!?;:…¿¡\"'“”«»()"


def word_key(word: str) -> str:
    folded = unicodedata.normalize("NFD", word.casefold())
    plain = "".join(char for char in folded if unicodedata.category(char) != "Mn")
    return plain.strip(_PUNCTUATION)


def repeats_previous(previous: str, current: str) -> bool:
    """A frase nova é a anterior de novo, ou o rabo de um loop que o Whisper continuou."""
    previous_words = [word for word in (word_key(word) for word in previous.split()) if word]
    current_words = [word for word in (word_key(word) for word in current.split()) if word]
    if not previous_words or not current_words:
        return False
    if current_words == previous_words:
        return True
    if len(current_words) <= 6 and " ".join(current_words) in " ".join(previous_words):
        return True
    if len(current_words) <= 2 and all(word in previous_words[-8:] for word in current_words):
        return True
    if len(current_words) < 4 or len(current_words) > len(previous_words) + 1:
        return False
    known = set(previous_words)
    shared = sum(1 for word in current_words if word in known)
    return shared / len(current_words) >= 0.85


def same_utterance(left: str, right: str) -> bool:
    """A mesma frase, ignorando acento e pontuação. Serve para reconhecer o próprio áudio de volta."""
    a = [word_key(word) for word in left.split()]
    b = [word_key(word) for word in right.split()]
    a = [word for word in a if word]
    b = [word for word in b if word]
    if not a or not b:
        return False
    return a == b


def stable_prefix(previous: list[str], current: list[str]) -> int:
    """Palavras iniciais em que duas hipóteses seguidas concordam (LocalAgreement-2).

    O Whisper reescreve o fim da frase enquanto ela cresce; o começo que já se
    repetiu em duas transcrições quase nunca muda depois.
    """
    count = 0
    for before, after in zip(previous, current):
        if word_key(before) != word_key(after):
            break
        count += 1
    return count


def drop_repeated_prefix(previous: str, current: str, max_words: int = 4) -> str:
    """Tira do começo de ``current`` as palavras que fecharam ``previous``.

    Uma frase longa demais é cortada com ~100 ms de sobreposição para não engolir
    a sílaba da emenda; às vezes a mesma palavra sai nos dois pedaços.
    """
    tail = [word_key(word) for word in previous.split()[-max_words:]]
    words = current.split()
    head = [word_key(word) for word in words[:max_words]]
    for size in range(min(len(tail), len(head)), 0, -1):
        if tail[-size:] == head[:size]:
            return " ".join(words[size:])
    return current


class RollingTranscript:
    """O que a pessoa disse nos últimos segundos: vira prompt do Whisper na frase seguinte."""

    def __init__(self, seconds: float = 15.0, max_chars: int = 240) -> None:
        self.seconds = seconds
        self.max_chars = max_chars
        self._items: deque[tuple[float, str]] = deque()

    def add(self, text: str, at: float) -> None:
        text = " ".join(text.split())
        if text:
            self._items.append((at, text))
        self._expire(at)

    def text(self, now: float) -> str:
        self._expire(now)
        return self._clip(" ".join(text for _at, text in self._items))

    def already_said(self, text: str, now: float, within: float = 12.0) -> bool:
        """A mesma frase curta já saiu há pouco. Um loop do Whisper não volta a ser falado."""
        words = [word for word in (word_key(word) for word in text.split()) if word]
        if len(words) < 3:
            return False
        key = " ".join(words)
        for at, previous in self._items:
            if now - at > within:
                continue
            earlier = " ".join(word for word in (word_key(word) for word in previous.split()) if word)
            if earlier == key:
                return True
        return False

    def last_phrases(self, now: float, limit: int = 3) -> str:
        """As últimas falas ainda dentro da janela, para o Whisper e para o tradutor."""
        self._expire(now)
        chosen = list(self._items)[-max(1, limit) :]
        return self._clip(" ".join(text for _at, text in chosen))

    def _clip(self, joined: str) -> str:
        if len(joined) <= self.max_chars:
            return joined
        clipped = joined[-self.max_chars :]
        space = clipped.find(" ")
        return clipped[space + 1 :] if space >= 0 else clipped

    def clear(self) -> None:
        self._items.clear()

    def _expire(self, now: float) -> None:
        while self._items and now - self._items[0][0] > self.seconds:
            self._items.popleft()
