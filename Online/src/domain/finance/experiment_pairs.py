"""Closed A-B, B-C, and A-C finance experiment pair identities."""

from enum import StrEnum, unique
from typing import Final

from src.domain.finance.artifact import ForecastArm


@unique
class ForecastPair(StrEnum):
    A_B = "A-B"
    B_C = "B-C"
    A_C = "A-C"


PAIR_ARMS: Final = {
    ForecastPair.A_B: (ForecastArm.DIRECT, ForecastArm.SEARCH_ONLY),
    ForecastPair.B_C: (ForecastArm.SEARCH_ONLY, ForecastArm.SEARCH_DAG),
    ForecastPair.A_C: (ForecastArm.DIRECT, ForecastArm.SEARCH_DAG),
}


__all__ = ["ForecastPair", "PAIR_ARMS"]
