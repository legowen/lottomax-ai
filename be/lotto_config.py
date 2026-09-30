"""Shared game constants (no app state, safe to import from any module)."""
from datetime import date
from math import comb

PICK = 7

# Game history (docs/RESEARCH.md §1 and Stage 5 addendum):
#   Era 1: 7/49 weekly (Fridays)           -> 2009-09-25 .. 2019-05-10
#   Era 2: 7/50 twice weekly (Tue/Fri)     -> 2019-05-14 .. 2026-04-10
#   Era 3: 7/52 twice weekly, $6 / 4 lines -> 2026-04-14 ..
ERA2_START_DATE = "2019-05-14"
ERA3_START_DATE = "2026-04-14"
ERA1_POOL, ERA2_POOL, ERA3_POOL = 49, 50, 52
CURRENT_POOL = ERA3_POOL

# Current retail price (verified against public reports of the 2026-04 change);
# update here if the operator changes it again.
TICKET_PRICE = 6.0
LINES_PER_TICKET = 4


def pool_for_date(date_str: str) -> int:
    """Number pool (1..N) in force on a draw date given as YYYY-MM-DD."""
    if date_str >= ERA3_START_DATE:
        return ERA3_POOL
    if date_str >= ERA2_START_DATE:
        return ERA2_POOL
    return ERA1_POOL


def combinations(pool: int = CURRENT_POOL, pick: int = PICK) -> int:
    return comb(pool, pick)


def parse_date(s: str) -> date:
    return date.fromisoformat(s)
