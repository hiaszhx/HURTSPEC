from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy.signal import savgol_filter


DEFAULT_INTERFERENCE_RANGES = [
    (1300.0, 1500.0),
    (1800.0, 2000.0),
    (1950.0, 2050.0),
]

DEFAULT_FINGERPRINT_RANGES = [
    (400.0, 1800.0),
]


@dataclass
class BandSelectionConfig:
    mode: str = "remove_interference"  # remove_interference or fingerprint
    interference_ranges: list[tuple[float, float]] = field(
        default_factory=lambda: list(DEFAULT_INTERFERENCE_RANGES)
    )
    fingerprint_ranges: list[tuple[float, float]] = field(
        default_factory=lambda: list(DEFAULT_FINGERPRINT_RANGES)
    )


@dataclass
class SGConfig:
    enabled: bool = True
    window_length: int = 15
    polyorder: int = 2


def build_band_mask(wave_grid: np.ndarray, config: BandSelectionConfig) -> np.ndarray:
    mask = np.ones_like(wave_grid, dtype=bool)

    if config.mode == "remove_interference":
        for low, high in config.interference_ranges:
            mask &= ~((wave_grid >= low) & (wave_grid <= high))
    elif config.mode == "fingerprint":
        mask = np.zeros_like(wave_grid, dtype=bool)
        for low, high in config.fingerprint_ranges:
            mask |= (wave_grid >= low) & (wave_grid <= high)
    else:
        raise ValueError(f"Unknown band selection mode: {config.mode}")

    if int(np.sum(mask)) < 8:
        raise ValueError("Band selection left too few points.")

    return mask


def apply_sg_filter(X: np.ndarray, config: SGConfig) -> np.ndarray:
    if not config.enabled:
        return X

    n_features = X.shape[1]
    window = int(config.window_length)

    if window % 2 == 0:
        window += 1

    if window > n_features:
        window = n_features if n_features % 2 == 1 else n_features - 1

    if window <= config.polyorder:
        window = config.polyorder + 3
        if window % 2 == 0:
            window += 1
        if window > n_features:
            return X

    return savgol_filter(X, window_length=window, polyorder=config.polyorder, axis=1)


def apply_snv(X: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    mean = np.mean(X, axis=1, keepdims=True)
    std = np.std(X, axis=1, keepdims=True)
    std = np.where(std < eps, 1.0, std)
    return (X - mean) / std
