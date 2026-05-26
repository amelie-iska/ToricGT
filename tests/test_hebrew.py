from toricgt.hebrew import hebrew_niqqud_stats, prefer_pointed_variant, strip_hebrew_diacritics


def test_hebrew_niqqud_stats_detects_pointed_and_unpointed_text():
    unpointed = hebrew_niqqud_stats("שלום")
    pointed = hebrew_niqqud_stats("שָׁלוֹם")

    assert unpointed.contains_hebrew
    assert unpointed.has_hebrew_without_niqqud
    assert not unpointed.has_niqqud
    assert pointed.contains_hebrew
    assert pointed.has_niqqud
    assert not pointed.has_hebrew_without_niqqud


def test_prefer_pointed_variant_requires_same_consonants():
    assert prefer_pointed_variant("שלום", "שָׁלוֹם") == "שָׁלוֹם"
    assert prefer_pointed_variant("שלום", "מֶלֶךְ") == "שלום"
    assert strip_hebrew_diacritics("שָׁלוֹם") == "שלום"
