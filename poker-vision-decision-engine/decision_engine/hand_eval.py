from __future__ import annotations

from enum import Enum


class HandCategory(str, Enum):
    PREMIUM = "premium"  # AA, KK, QQ, AK
    STRONG = "strong"  # JJ, TT, AQ, AJ
    MEDIUM = "medium"  # pairs 22-99, suited broadway
    SPECULATIVE = "speculative"  # suited aces (A2s-A9s), suited connectors
    WEAK = "weak"  # everything else


_RANK_VALUE: dict[str, int] = {
    "A": 14,
    "K": 13,
    "Q": 12,
    "J": 11,
    "T": 10,
    "9": 9,
    "8": 8,
    "7": 7,
    "6": 6,
    "5": 5,
    "4": 4,
    "3": 3,
    "2": 2,
}

_BROADWAY: frozenset[int] = frozenset({10, 11, 12, 13, 14})  # T, J, Q, K, A


def _parse_card(card: str) -> tuple[str, str]:
    """Return (rank, suit). Rank is uppercase, suit is lowercase."""
    return card[:-1].upper(), card[-1].lower()


def classify_hand(cards: list[str]) -> HandCategory:
    """Classify a two-card starting hand into a strategic category."""
    r1, s1 = _parse_card(cards[0])
    r2, s2 = _parse_card(cards[1])

    suited = s1 == s2
    v1, v2 = _RANK_VALUE[r1], _RANK_VALUE[r2]
    high, low = max(v1, v2), min(v1, v2)
    pair = v1 == v2

    # --- Premium: AA, KK, QQ, AK (suited or offsuit) ---
    if pair and high >= 12:  # AA, KK, QQ
        return HandCategory.PREMIUM
    if high == 14 and low == 13:  # AK
        return HandCategory.PREMIUM

    # --- Strong: JJ, TT, AQ, AJ (suited or offsuit) ---
    if pair and high in (11, 10):  # JJ, TT
        return HandCategory.STRONG
    if high == 14 and low in (12, 11):  # AQ, AJ
        return HandCategory.STRONG

    # --- Medium: pairs 22-99, suited broadway (e.g. KQs, KJs, QJs, ATs) ---
    if pair:  # 22-99 (higher pairs already handled)
        return HandCategory.MEDIUM
    if suited and high in _BROADWAY and low in _BROADWAY:
        return HandCategory.MEDIUM

    # --- Speculative: suited aces (A2s-A9s), suited connectors ---
    if suited and high == 14:  # A2s through A9s (ATs is broadway → medium above)
        return HandCategory.SPECULATIVE
    if suited and high - low == 1:  # suited connectors: 98s, 87s, 76s, etc.
        return HandCategory.SPECULATIVE

    # --- Weak: everything else ---
    return HandCategory.WEAK
