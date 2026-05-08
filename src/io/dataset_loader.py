from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
import re
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd


FILENAME_TS_PATTERN = re.compile(r"_(\d{8})_(\d{6,9})\.csv$", re.IGNORECASE)


@dataclass
class SampleRecord:
    class_name: str
    file_path: Path
    file_name: str
    date_str: str
    time_str: str
    timestamp: datetime
    seconds_in_day: float
    wave: np.ndarray
    intensity: np.ndarray


@dataclass
class LoadedDataset:
    samples: list[SampleRecord]

    def class_names(self) -> list[str]:
        return sorted({s.class_name for s in self.samples})

    def to_index_frame(self) -> pd.DataFrame:
        rows = []
        for s in self.samples:
            rows.append(
                {
                    "class_name": s.class_name,
                    "file_name": s.file_name,
                    "file_path": str(s.file_path),
                    "date": s.date_str,
                    "time": s.time_str,
                    "seconds_in_day": s.seconds_in_day,
                    "wave_min": float(np.min(s.wave)),
                    "wave_max": float(np.max(s.wave)),
                    "n_points": int(s.wave.shape[0]),
                }
            )
        return pd.DataFrame(rows)


class DatasetLoadingError(RuntimeError):
    pass


def parse_timestamp_from_filename(file_name: str) -> tuple[str, str, datetime, float]:
    match = FILENAME_TS_PATTERN.search(file_name)
    if match is None:
        raise DatasetLoadingError(
            f"Cannot parse date/time from file name: {file_name}. "
            "Expected suffix _YYYYMMDD_HHmmssXXX.csv"
        )

    date_str, time_str = match.group(1), match.group(2)

    if len(time_str) < 6:
        raise DatasetLoadingError(f"Invalid time token in file name: {file_name}")

    hour = int(time_str[0:2])
    minute = int(time_str[2:4])
    second = int(time_str[4:6])
    fraction_text = time_str[6:] if len(time_str) > 6 else "0"

    if len(fraction_text) > 6:
        fraction_text = fraction_text[:6]

    microsecond = int(fraction_text.ljust(6, "0"))

    try:
        timestamp = datetime(
            year=int(date_str[0:4]),
            month=int(date_str[4:6]),
            day=int(date_str[6:8]),
            hour=hour,
            minute=minute,
            second=second,
            microsecond=microsecond,
        )
    except ValueError as exc:
        raise DatasetLoadingError(f"Invalid date/time in file name: {file_name}") from exc

    seconds_in_day = (
        hour * 3600
        + minute * 60
        + second
        + microsecond / 1_000_000.0
    )

    return date_str, time_str, timestamp, seconds_in_day


def _read_single_csv(csv_path: Path) -> tuple[np.ndarray, np.ndarray]:
    try:
        frame = pd.read_csv(csv_path)
    except Exception as exc:
        raise DatasetLoadingError(f"Failed to read CSV: {csv_path}") from exc

    if frame.shape[1] < 2:
        raise DatasetLoadingError(f"CSV needs at least 2 columns: {csv_path}")

    wave = frame.iloc[:, 0].to_numpy(dtype=float)
    intensity = frame.iloc[:, 1].to_numpy(dtype=float)

    if wave.size < 8:
        raise DatasetLoadingError(f"Too few rows in CSV: {csv_path}")

    order = np.argsort(wave)
    wave = wave[order]
    intensity = intensity[order]

    return wave, intensity


def discover_category_dirs(root_dir: Path) -> list[Path]:
    if not root_dir.exists() or not root_dir.is_dir():
        raise DatasetLoadingError(f"Input directory does not exist: {root_dir}")

    category_dirs: list[Path] = []
    for p in root_dir.iterdir():
        if not p.is_dir() or p.name.startswith("."):
            continue

        has_matching_csv = any(FILENAME_TS_PATTERN.search(csv.name) for csv in p.rglob("*.csv"))
        if has_matching_csv:
            category_dirs.append(p)

    if not category_dirs:
        raise DatasetLoadingError(f"No category directories found under: {root_dir}")

    return sorted(category_dirs)


def iter_csv_files_under(folder: Path) -> Iterable[Path]:
    for path in folder.rglob("*.csv"):
        if path.is_file():
            yield path


def load_dataset(root_dir: str | Path) -> LoadedDataset:
    root = Path(root_dir).resolve()
    category_dirs = discover_category_dirs(root)

    samples: list[SampleRecord] = []

    for category_dir in category_dirs:
        class_name = category_dir.name
        csv_files = sorted(iter_csv_files_under(category_dir))
        for csv_path in csv_files:
            if FILENAME_TS_PATTERN.search(csv_path.name) is None:
                continue

            date_str, time_str, ts, sec = parse_timestamp_from_filename(csv_path.name)
            wave, intensity = _read_single_csv(csv_path)

            sample = SampleRecord(
                class_name=class_name,
                file_path=csv_path,
                file_name=csv_path.name,
                date_str=date_str,
                time_str=time_str,
                timestamp=ts,
                seconds_in_day=sec,
                wave=wave,
                intensity=intensity,
            )
            samples.append(sample)

    if not samples:
        raise DatasetLoadingError(f"No CSV files found under: {root}")

    unique_classes = sorted({s.class_name for s in samples})
    if len(unique_classes) < 2:
        raise DatasetLoadingError(
            "Need at least 2 classes for classification. "
            f"Found classes: {', '.join(unique_classes)}"
        )

    return LoadedDataset(samples=samples)
