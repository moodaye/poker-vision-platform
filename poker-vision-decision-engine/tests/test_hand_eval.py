import pytest
from decision_engine.hand_eval import HandCategory, classify_hand


@pytest.mark.parametrize(
    "cards",
    [
        ["Ah", "Ad"],  # AA
        ["Kh", "Kd"],  # KK
        ["Qh", "Qd"],  # QQ
        ["Ah", "Kd"],  # AKo
        ["As", "Kh"],  # AKo (reversed)
        ["Ah", "Ks"],  # AKs
    ],
)
def test_premium_hands(cards: list[str]) -> None:
    assert classify_hand(cards) == HandCategory.PREMIUM


@pytest.mark.parametrize(
    "cards",
    [
        ["Jh", "Jd"],  # JJ
        ["Th", "Td"],  # TT
        ["Ah", "Qd"],  # AQo
        ["Ah", "Qs"],  # AQs
        ["Ah", "Jd"],  # AJo
        ["As", "Jh"],  # AJs
    ],
)
def test_strong_hands(cards: list[str]) -> None:
    assert classify_hand(cards) == HandCategory.STRONG


@pytest.mark.parametrize(
    "cards",
    [
        ["9h", "9d"],  # 99
        ["8h", "8d"],  # 88
        ["2h", "2d"],  # 22
        ["Kh", "Qh"],  # KQs
        ["Kh", "Jh"],  # KJs
        ["Qh", "Jh"],  # QJs
        ["Jh", "Th"],  # JTs
        ["Ah", "Th"],  # ATs (broadway suited)
    ],
)
def test_medium_hands(cards: list[str]) -> None:
    assert classify_hand(cards) == HandCategory.MEDIUM


@pytest.mark.parametrize(
    "cards",
    [
        ["Ah", "9h"],  # A9s
        ["Ah", "2h"],  # A2s
        ["9h", "8h"],  # 98s
        ["6h", "5h"],  # 65s
        ["5c", "4c"],  # 54s
    ],
)
def test_speculative_hands(cards: list[str]) -> None:
    assert classify_hand(cards) == HandCategory.SPECULATIVE


@pytest.mark.parametrize(
    "cards",
    [
        ["7h", "2d"],  # 72o — the classic worst hand
        ["9h", "2d"],  # 92o
        ["Kh", "3d"],  # K3o
        ["Qh", "7d"],  # Q7o
    ],
)
def test_weak_hands(cards: list[str]) -> None:
    assert classify_hand(cards) == HandCategory.WEAK
