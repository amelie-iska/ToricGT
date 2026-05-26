"""Hebrew text utilities for niqqud-aware curation."""

from __future__ import annotations

import re
import unicodedata
from dataclasses import asdict, dataclass


HEBREW_LETTER_RE = re.compile(r"[\u05D0-\u05EA]")
HEBREW_DIACRITIC_RE = re.compile(r"[\u0591-\u05C7]")
NIQQUD_RE = re.compile(r"[\u05B0-\u05BD\u05BF\u05C1-\u05C2\u05C4-\u05C5\u05C7]")


@dataclass(frozen=True)
class HebrewNiqqudStats:
    contains_hebrew: bool
    hebrew_letters: int
    hebrew_diacritics: int
    niqqud_marks: int
    has_niqqud: bool
    has_hebrew_without_niqqud: bool
    niqqud_per_hebrew_letter: float

    def to_dict(self) -> dict[str, bool | int | float]:
        return asdict(self)


def strip_hebrew_diacritics(text: str) -> str:
    """Remove Hebrew combining marks while preserving consonants and spacing."""

    return "".join(ch for ch in unicodedata.normalize("NFD", text) if not HEBREW_DIACRITIC_RE.match(ch))


def hebrew_niqqud_stats(*texts: str) -> HebrewNiqqudStats:
    """Summarize whether Hebrew text is pointed.

    This is deliberately conservative. It detects and preserves existing niqqud;
    it does not invent vowels for unpointed Hebrew.
    """

    letters = 0
    diacritics = 0
    niqqud = 0
    for text in texts:
        if not text:
            continue
        letters += len(HEBREW_LETTER_RE.findall(text))
        diacritics += len(HEBREW_DIACRITIC_RE.findall(text))
        niqqud += len(NIQQUD_RE.findall(text))
    contains = letters > 0
    has_niqqud = niqqud > 0
    return HebrewNiqqudStats(
        contains_hebrew=contains,
        hebrew_letters=letters,
        hebrew_diacritics=diacritics,
        niqqud_marks=niqqud,
        has_niqqud=has_niqqud,
        has_hebrew_without_niqqud=contains and not has_niqqud,
        niqqud_per_hebrew_letter=(niqqud / letters) if letters else 0.0,
    )


def same_hebrew_letters(left: str, right: str) -> bool:
    """Return whether two strings have the same Hebrew consonant sequence."""

    left_letters = "".join(HEBREW_LETTER_RE.findall(strip_hebrew_diacritics(left)))
    right_letters = "".join(HEBREW_LETTER_RE.findall(strip_hebrew_diacritics(right)))
    return bool(left_letters) and left_letters == right_letters


def prefer_pointed_variant(current: str, *candidates: str) -> str:
    """Use a pointed candidate only when it has the same Hebrew consonants."""

    current_stats = hebrew_niqqud_stats(current)
    if not current_stats.contains_hebrew or current_stats.has_niqqud:
        return current
    for candidate in candidates:
        candidate_stats = hebrew_niqqud_stats(candidate)
        if candidate_stats.has_niqqud and same_hebrew_letters(current, candidate):
            return candidate
    return current
