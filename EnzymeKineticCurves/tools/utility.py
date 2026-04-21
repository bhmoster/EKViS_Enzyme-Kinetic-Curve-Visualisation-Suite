from enum import Enum

import numpy as np


def compute_standard_error(lst):
    return np.std(lst, ddof=1) / np.sqrt(len(lst))


class UtilityUnit(Enum):
    """Provides tools for type safe Unit management."""

    Femto = 0
    Pico = 1
    Nano = 2
    Micro = 3
    Milli = 4
    Molar = 5

    @staticmethod
    def get_text(unit: "UtilityUnit") -> str:
        match unit:
            case UtilityUnit.Femto:
                return "fM"

            case UtilityUnit.Pico:
                return "pM"

            case UtilityUnit.Nano:
                return "nM"

            case UtilityUnit.Micro:
                return "μM"

            case UtilityUnit.Milli:
                return "mM"

            case UtilityUnit.Molar:
                return "M"

    @staticmethod
    def from_text(s: str) -> "UtilityUnit":
        match s:
            case "femto":
                return UtilityUnit.Femto

            case "pico":
                return UtilityUnit.Pico

            case "nano":
                return UtilityUnit.Nano

            case "micro":
                return UtilityUnit.Micro

            case "milli":
                return UtilityUnit.Milli

            case "molar":
                return UtilityUnit.Molar

            case _:
                raise Exception("Could not decice unit.")


class Unit:
    """Provides type safe conversion of units."""

    def __init__(self, num_micro: float):
        self._num_micro = num_micro

    def get_num(self, unit: UtilityUnit) -> float:
        match unit:
            case UtilityUnit.Femto:
                return self._num_micro * 1000**3

            case UtilityUnit.Pico:
                return self._num_micro * 1000**2

            case UtilityUnit.Nano:
                return self._num_micro * 1000

            case UtilityUnit.Micro:
                return self._num_micro

            case UtilityUnit.Milli:
                return self._num_micro / 1000

            case UtilityUnit.Molar:
                return self._num_micro / 1000**2
