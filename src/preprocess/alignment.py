from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from src.io.dataset_loader import LoadedDataset


@dataclass
class AlignedDataset:
    X: np.ndarray
    y: np.ndarray
    class_names: list[str]
    wave_grid: np.ndarray
    seconds_in_day: np.ndarray
    sample_names: list[str]
    sample_paths: list[str]


class AlignmentError(RuntimeError):
    pass


def _estimate_step(all_waves: list[np.ndarray]) -> float:
    diffs = []
    for w in all_waves:
        d = np.diff(w)
        d = d[d > 0]
        if d.size:
            diffs.append(np.median(d))

    if not diffs:
        raise AlignmentError("Cannot estimate wave step from samples.")

    step = float(np.median(np.array(diffs, dtype=float)))
    if not np.isfinite(step) or step <= 0:
        raise AlignmentError("Estimated invalid wave step.")

    return step


def build_common_wave_grid(dataset: LoadedDataset) -> np.ndarray:
    mins = [float(np.min(s.wave)) for s in dataset.samples]
    maxs = [float(np.max(s.wave)) for s in dataset.samples]

    start = max(mins)
    end = min(maxs)

    if start >= end:
        raise AlignmentError(
            "No common wave range across all samples. "
            f"Computed range [{start}, {end}]"
        )

    step = _estimate_step([s.wave for s in dataset.samples])
    n_points = int(np.floor((end - start) / step)) + 1

    if n_points < 16:
        raise AlignmentError(
            "Common wave range too narrow after alignment. "
            f"Only {n_points} points available."
        )

    grid = start + np.arange(n_points, dtype=float) * step
    return grid


def align_dataset(dataset: LoadedDataset) -> AlignedDataset:
    class_names = dataset.class_names()
    class_to_int = {c: i for i, c in enumerate(class_names)}

    wave_grid = build_common_wave_grid(dataset)

    X_list = []
    y_list = []
    sec_list = []
    sample_names = []
    sample_paths = []

    for sample in dataset.samples:
        intensity_interp = np.interp(wave_grid, sample.wave, sample.intensity)
        X_list.append(intensity_interp)
        y_list.append(class_to_int[sample.class_name])
        sec_list.append(sample.seconds_in_day)
        sample_names.append(sample.file_name)
        sample_paths.append(str(sample.file_path))

    X = np.vstack(X_list).astype(float)
    y = np.array(y_list, dtype=int)
    seconds = np.array(sec_list, dtype=float)

    return AlignedDataset(
        X=X,
        y=y,
        class_names=class_names,
        wave_grid=wave_grid,
        seconds_in_day=seconds,
        sample_names=sample_names,
        sample_paths=sample_paths,
    )
