from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass, replace
from datetime import datetime
import json
from pathlib import Path
import re
import sys
import tomllib
from typing import Any, Sequence

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.cross_decomposition import PLSRegression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    cohen_kappa_score,
    confusion_matrix,
    f1_score,
    matthews_corrcoef,
    precision_recall_fscore_support,
    precision_score,
    recall_score,
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import torch
from torch import nn
from torch.optim import Adam

from src.io.dataset_loader import load_dataset
from src.preprocess.alignment import align_dataset
from src.preprocess.filters import apply_snv


@dataclass
class BaselineConfig:
    config_path: str
    input_root: str
    output_root: str
    s3prl_repo: str
    upstream: str
    upstreams: list[str]
    test_size: float
    random_state: int
    batch_size: int
    device: str
    classifier_epochs: int
    classifier_lr: float
    classifier_weight_decay: float
    classifier_head_type: str
    classifier_hidden_dim: int
    classifier_dropout: float
    classifier_label_smoothing_enabled: bool
    classifier_label_smoothing: float
    classifier_prototype_dim: int
    classifier_prototype_temperature: float
    classifier_supcon_enabled: bool
    classifier_supcon_weight: float
    classifier_supcon_temperature: float
    preprocess: "SpectralPreprocessConfig"
    outputs: "OutputConfig"


class BaselineError(RuntimeError):
    pass


@dataclass
class TrainEvalResult:
    y_test: np.ndarray
    y_test_pred: np.ndarray
    cm_test: np.ndarray
    y_all_pred: np.ndarray
    y_all_logits: np.ndarray
    cm_all: np.ndarray
    metrics_test: dict
    metrics_all: dict
    report_test: pd.DataFrame
    report_all: pd.DataFrame
    complexity: dict
    history: pd.DataFrame
    test_indices: np.ndarray
    checkpoint: dict


@dataclass
class PLSCalibrationResult:
    X: np.ndarray
    metadata: dict
    state: dict


@dataclass
class SNVStepConfig:
    enabled: bool = True
    eps: float = 1e-12


@dataclass
class WaveletDriftStepConfig:
    enabled: bool = False
    wavelet: str = "db6"
    level: int = 4
    mode: str = "symmetric"
    approximation_scale: float = 0.0


@dataclass
class PLSStepConfig:
    enabled: bool = True
    components: int = 0


@dataclass
class SegmentNormalizeStepConfig:
    enabled: bool = False
    ranges: list[tuple[float, float]] | None = None
    method: str = "zscore"
    eps: float = 1e-12


@dataclass
class BandSelectionStepConfig:
    enabled: bool = False
    method: str = "none"
    fusion_mode: str = "dual"
    manual_ranges: list[tuple[float, float]] | None = None
    top_k: int = 0
    top_ratio: float = 0.25
    min_bands: int = 16
    pls_components: int = 0
    lasso_alpha: float = 0.05
    cars_iterations: int = 40
    cars_sample_ratio: float = 0.8
    ga_population: int = 24
    ga_generations: int = 30
    ga_crossover_rate: float = 0.8
    ga_mutation_rate: float = 0.08
    ga_elite_count: int = 2
    iwoa_population: int = 24
    iwoa_iterations: int = 30
    iwoa_b: float = 1.0
    iwoa_mutation_rate: float = 0.05
    epochs: int = 200
    lr: float = 0.01
    weight_decay: float = 0.0001
    hidden_dim: int = 64
    temperature: float = 1.0
    sparsity_lambda: float = 0.001


@dataclass
class SpectralPreprocessConfig:
    enabled: bool
    order: list[str]
    snv: SNVStepConfig
    wavelet: WaveletDriftStepConfig
    pls: PLSStepConfig
    segment_normalize: SegmentNormalizeStepConfig
    band_selection: BandSelectionStepConfig


@dataclass
class OutputConfig:
    save_embeddings: bool = True
    save_preprocessed_spectra: bool = True
    save_sample_index: bool = True


@dataclass
class SpectralPreprocessResult:
    X: np.ndarray
    metadata: dict
    state: dict
    selected_band_features: np.ndarray | None = None


class SmallMLPHead(nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        num_classes: int,
        dropout: float,
    ) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(p=dropout),
            nn.Linear(hidden_dim, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class LinearHead(nn.Module):
    def __init__(self, input_dim: int, num_classes: int) -> None:
        super().__init__()
        self.net = nn.Linear(input_dim, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class PrototypeHead(nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        prototype_dim: int,
        num_classes: int,
        dropout: float,
        temperature: float,
    ) -> None:
        super().__init__()
        self.temperature = max(float(temperature), 1e-6)
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(p=dropout),
            nn.Linear(hidden_dim, prototype_dim),
        )
        self.prototypes = nn.Parameter(torch.empty(num_classes, prototype_dim))
        nn.init.xavier_uniform_(self.prototypes)

    def forward(
        self,
        x: torch.Tensor,
        return_features: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        features = self.encoder(x)
        features_norm = nn.functional.normalize(features, dim=1)
        prototypes_norm = nn.functional.normalize(self.prototypes, dim=1)
        logits = features_norm @ prototypes_norm.t() / self.temperature
        if return_features:
            return logits, features_norm
        return logits


class SupervisedContrastiveLoss(nn.Module):
    def __init__(self, temperature: float = 0.1) -> None:
        super().__init__()
        self.temperature = max(float(temperature), 1e-6)

    def forward(self, features: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        if features.ndim != 2:
            raise BaselineError(f"SupCon features must be 2D, got shape {tuple(features.shape)}")

        features = nn.functional.normalize(features, dim=1)
        labels = labels.view(-1, 1)
        positive_mask = torch.eq(labels, labels.t()).to(features.device, dtype=features.dtype)
        self_mask = torch.eye(features.shape[0], device=features.device, dtype=features.dtype)
        positive_mask = positive_mask * (1.0 - self_mask)
        positive_counts = positive_mask.sum(dim=1)

        if not torch.any(positive_counts > 0):
            return features.new_zeros(())

        logits = features @ features.t() / self.temperature
        logits = logits - torch.max(logits, dim=1, keepdim=True).values.detach()
        logits_mask = 1.0 - self_mask
        exp_logits = torch.exp(logits) * logits_mask
        log_prob = logits - torch.log(exp_logits.sum(dim=1, keepdim=True).clamp_min(1e-12))

        mean_log_prob_pos = (positive_mask * log_prob).sum(dim=1) / positive_counts.clamp_min(1.0)
        valid = positive_counts > 0
        return -mean_log_prob_pos[valid].mean()


def _safe_name(text: str) -> str:
    safe = re.sub(r"[^A-Za-z0-9_.-]+", "_", text.strip())
    return safe.strip("._") or "upstream"


def _create_output_dir(base_dir: Path, upstream: str) -> Path:
    base_dir.mkdir(parents=True, exist_ok=True)
    run_id = f"{_safe_name(upstream)}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    out_dir = base_dir / run_id

    suffix = 1
    while out_dir.exists():
        out_dir = base_dir / f"{run_id}_{suffix:02d}"
        suffix += 1

    out_dir.mkdir(parents=True, exist_ok=False)
    return out_dir


def _create_multi_output_dir(base_dir: Path) -> Path:
    base_dir.mkdir(parents=True, exist_ok=True)
    run_id = f"multi_upstream_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    out_dir = base_dir / run_id

    suffix = 1
    while out_dir.exists():
        out_dir = base_dir / f"{run_id}_{suffix:02d}"
        suffix += 1

    out_dir.mkdir(parents=True, exist_ok=False)
    return out_dir


def _log_progress(prefix: str, message: str) -> None:
    label = f"{prefix} " if prefix else ""
    print(f"{label}{message}", flush=True)


def _import_s3prl_upstream(repo_path: Path):
    if not repo_path.exists():
        raise BaselineError(f"s3prl repo path not found: {repo_path}")

    sys.path.insert(0, str(repo_path.resolve()))

    try:
        from s3prl.nn import S3PRLUpstream  # type: ignore
    except Exception as exc:
        raise BaselineError(
            "Failed to import S3PRLUpstream. Install dependencies first, e.g. "
            "pip install -e ./s3prl-main"
        ) from exc

    return S3PRLUpstream


def _to_pseudo_wave(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float32)
    x = x - float(np.mean(x))
    std = float(np.std(x))
    if std < 1e-8:
        std = 1.0
    x = x / std
    max_abs = float(np.max(np.abs(x)))
    if max_abs > 0:
        x = x / max_abs
    return x.astype(np.float32)


def _mean_pool_by_length(hidden: torch.Tensor, lengths: torch.Tensor) -> np.ndarray:
    pooled = []
    hidden_np = hidden.detach().cpu().numpy()
    lengths_np = lengths.detach().cpu().numpy()

    for i in range(hidden_np.shape[0]):
        t = int(lengths_np[i])
        t = max(1, min(t, hidden_np.shape[1]))
        pooled.append(hidden_np[i, :t, :].mean(axis=0))

    return np.stack(pooled, axis=0)


def _count_model_params(model: nn.Module) -> dict:
    total = int(sum(p.numel() for p in model.parameters()))
    trainable = int(sum(p.numel() for p in model.parameters() if p.requires_grad))
    return {
        "total_params": total,
        "trainable_params": trainable,
        "frozen_params": int(total - trainable),
    }


def _resolve_mlp_hidden_dim(input_dim: int, requested_hidden_dim: int) -> int:
    if requested_hidden_dim > 0:
        return int(requested_hidden_dim)
    return int(min(256, max(32, input_dim // 2)))


def _estimate_mlp_head_forward_flops_per_sample(
    input_dim: int,
    hidden_dim: int,
    num_classes: int,
) -> int:
    linear_1 = 2 * input_dim * hidden_dim + hidden_dim
    gelu = 8 * hidden_dim
    linear_2 = 2 * hidden_dim * num_classes + num_classes
    return int(linear_1 + gelu + linear_2)


def _estimate_prototype_head_forward_flops_per_sample(
    input_dim: int,
    hidden_dim: int,
    prototype_dim: int,
    num_classes: int,
) -> int:
    encoder = _estimate_mlp_head_forward_flops_per_sample(
        input_dim=input_dim,
        hidden_dim=hidden_dim,
        num_classes=prototype_dim,
    )
    feature_norm = 4 * prototype_dim
    prototype_norm = num_classes * 4 * prototype_dim
    cosine_logits = 2 * prototype_dim * num_classes
    return int(encoder + feature_norm + prototype_norm + cosine_logits)


def _estimate_classifier_head_forward_flops_per_sample(
    head_type: str,
    input_dim: int,
    hidden_dim: int,
    num_classes: int,
    prototype_dim: int | None = None,
) -> int:
    if head_type == "linear":
        return int(2 * input_dim * num_classes + num_classes)
    if head_type == "prototype":
        return _estimate_prototype_head_forward_flops_per_sample(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            prototype_dim=int(prototype_dim or hidden_dim),
            num_classes=num_classes,
        )
    return _estimate_mlp_head_forward_flops_per_sample(
        input_dim=input_dim,
        hidden_dim=hidden_dim,
        num_classes=num_classes,
    )


def _estimate_upstream_forward_flops_per_sample(
    model: nn.Module,
    sample_wave: np.ndarray,
    device: torch.device,
) -> dict:
    try:
        activities = [torch.profiler.ProfilerActivity.CPU]
        if device.type == "cuda":
            activities.append(torch.profiler.ProfilerActivity.CUDA)

        wav = torch.from_numpy(sample_wave[np.newaxis, :]).to(device)
        wav_len = torch.tensor([sample_wave.shape[0]], dtype=torch.long, device=device)

        with torch.no_grad():
            with torch.profiler.profile(activities=activities, with_flops=True) as prof:
                _ = model(wav, wav_len)

        total_flops = 0
        for evt in prof.key_averages():
            evt_flops = getattr(evt, "flops", 0)
            if evt_flops:
                total_flops += int(evt_flops)

        if total_flops <= 0:
            return {
                "upstream_forward_flops_per_sample": None,
                "upstream_flops_note": "torch.profiler returned 0 FLOPs",
            }

        return {
            "upstream_forward_flops_per_sample": int(total_flops),
            "upstream_flops_note": "estimated by torch.profiler",
        }
    except Exception as exc:
        return {
            "upstream_forward_flops_per_sample": None,
            "upstream_flops_note": f"unavailable: {exc}",
        }


def _compute_classification_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    class_names: Sequence[str],
) -> tuple[dict, pd.DataFrame]:
    labels = np.arange(len(class_names))
    precision_cls, recall_cls, f1_cls, support_cls = precision_recall_fscore_support(
        y_true,
        y_pred,
        labels=labels,
        zero_division=0,
    )

    report = pd.DataFrame(
        {
            "class_id": labels,
            "class_name": list(class_names),
            "precision": precision_cls,
            "recall": recall_cls,
            "f1": f1_cls,
            "support": support_cls,
        }
    )

    metrics = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "precision_macro": float(precision_score(y_true, y_pred, average="macro", zero_division=0)),
        "precision_micro": float(precision_score(y_true, y_pred, average="micro", zero_division=0)),
        "precision_weighted": float(precision_score(y_true, y_pred, average="weighted", zero_division=0)),
        "recall_macro": float(recall_score(y_true, y_pred, average="macro", zero_division=0)),
        "recall_micro": float(recall_score(y_true, y_pred, average="micro", zero_division=0)),
        "recall_weighted": float(recall_score(y_true, y_pred, average="weighted", zero_division=0)),
        "f1_macro": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "f1_micro": float(f1_score(y_true, y_pred, average="micro", zero_division=0)),
        "f1_weighted": float(f1_score(y_true, y_pred, average="weighted", zero_division=0)),
        "mcc": float(matthews_corrcoef(y_true, y_pred)),
        "cohen_kappa": float(cohen_kappa_score(y_true, y_pred)),
        "n_samples": int(len(y_true)),
    }

    return metrics, report


def _make_split_indices(
    y: np.ndarray,
    test_size: float,
    random_state: int,
) -> tuple[np.ndarray, np.ndarray]:
    all_indices = np.arange(len(y), dtype=int)
    train_idx, test_idx = train_test_split(
        all_indices,
        test_size=test_size,
        random_state=random_state,
        stratify=y,
    )
    return np.asarray(train_idx, dtype=int), np.asarray(test_idx, dtype=int)


def _one_hot_labels(y: np.ndarray, n_classes: int) -> np.ndarray:
    y = np.asarray(y, dtype=int)
    one_hot = np.zeros((len(y), n_classes), dtype=np.float32)
    one_hot[np.arange(len(y)), y] = 1.0
    return one_hot


def _resolve_pls_components(
    requested_components: int,
    n_train: int,
    n_features: int,
) -> int:
    max_components = min(
        max(1, n_train - 1),
        max(1, n_features),
    )

    if requested_components <= 0:
        return min(10, max_components)

    if requested_components > max_components:
        raise BaselineError(
            "pls_components is too large for this train split. "
            f"Requested {requested_components}, maximum is {max_components}."
        )

    return int(requested_components)


def apply_pls_spectral_calibration(
    X: np.ndarray,
    y: np.ndarray,
    train_indices: np.ndarray,
    class_names: Sequence[str],
    requested_components: int,
) -> PLSCalibrationResult:
    X = np.asarray(X, dtype=np.float32)
    y = np.asarray(y, dtype=int)
    train_indices = np.asarray(train_indices, dtype=int)

    if X.ndim != 2:
        raise BaselineError(f"Expected 2D spectral matrix, got shape {X.shape}")

    n_classes = len(class_names)
    n_components = _resolve_pls_components(
        requested_components=requested_components,
        n_train=len(train_indices),
        n_features=X.shape[1],
    )

    X_train = X[train_indices]
    y_train = y[train_indices]
    Y_train = _one_hot_labels(y_train, n_classes=n_classes)

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_all_scaled = scaler.transform(X)

    pls = PLSRegression(n_components=n_components, scale=False)
    pls.fit(X_train_scaled, Y_train)

    X_scores = pls.transform(X_all_scaled)
    X_reconstructed_scaled = X_scores @ pls.x_loadings_.T
    X_calibrated = scaler.inverse_transform(X_reconstructed_scaled).astype(np.float32)

    reconstruction_rmse = float(np.sqrt(np.mean((X - X_calibrated) ** 2)))
    train_reconstruction_rmse = float(
        np.sqrt(np.mean((X_train - X_calibrated[train_indices]) ** 2))
    )

    metadata = {
        "enabled": True,
        "method": "PLSRegression PLS-DA spectral projection/reconstruction",
        "requested_components": int(requested_components),
        "n_components": int(n_components),
        "n_train": int(len(train_indices)),
        "n_features": int(X.shape[1]),
        "n_classes": int(n_classes),
        "target_encoding": "one-hot class labels",
        "fit_scope": "train split only",
        "reconstruction_rmse_all": reconstruction_rmse,
        "reconstruction_rmse_train": train_reconstruction_rmse,
    }

    state = {
        "enabled": True,
        "method": "PLSRegression PLS-DA spectral projection/reconstruction",
        "n_components": int(n_components),
        "n_features": int(X.shape[1]),
        "scaler_mean": scaler.mean_.astype(np.float32),
        "scaler_scale": scaler.scale_.astype(np.float32),
        "x_rotations": pls.x_rotations_.astype(np.float32),
        "x_loadings": pls.x_loadings_.astype(np.float32),
    }

    return PLSCalibrationResult(X=X_calibrated, metadata=metadata, state=state)


@dataclass
class BandSelectionResult:
    X: np.ndarray
    wave_grid: np.ndarray
    metadata: dict
    state: dict


class _LearnableBandSelector(nn.Module):
    def __init__(
        self,
        n_features: int,
        n_classes: int,
        hidden_dim: int,
        mode: str,
        temperature: float,
    ) -> None:
        super().__init__()
        self.mode = mode
        self.temperature = max(float(temperature), 1e-4)
        self.band_logits = nn.Parameter(torch.zeros(n_features))

        if hidden_dim > 0:
            self.classifier = nn.Sequential(
                nn.Linear(n_features, hidden_dim),
                nn.GELU(),
                nn.Linear(hidden_dim, n_classes),
            )
        else:
            self.classifier = nn.Linear(n_features, n_classes)

    def band_weights(self) -> torch.Tensor:
        if self.mode == "band_attention":
            return torch.softmax(self.band_logits / self.temperature, dim=0) * self.band_logits.numel()
        return torch.sigmoid(self.band_logits / self.temperature)

    def regularization(self) -> torch.Tensor:
        weights = self.band_weights()
        if self.mode == "band_attention":
            probs = weights / weights.sum().clamp_min(1e-8)
            entropy = -(probs * torch.log(probs.clamp_min(1e-8))).sum()
            return entropy / max(np.log(float(weights.numel())), 1e-8)
        return weights.mean()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.classifier(x * self.band_weights())


def _resolve_selected_band_count(n_features: int, config: BandSelectionStepConfig) -> int:
    if n_features < 1:
        raise BaselineError("Band selection received an empty feature matrix.")

    if config.top_k > 0:
        count = int(config.top_k)
    else:
        ratio = float(config.top_ratio)
        if not 0.0 < ratio <= 1.0:
            raise BaselineError("band_selection.top_ratio must be > 0 and <= 1.")
        count = int(round(n_features * ratio))

    count = max(int(config.min_bands), count)
    count = max(1, min(int(n_features), count))
    return count


def _top_band_indices(scores: np.ndarray, n_select: int) -> np.ndarray:
    scores = np.asarray(scores, dtype=np.float64)
    scores = np.nan_to_num(scores, nan=0.0, posinf=0.0, neginf=0.0)
    if scores.ndim != 1:
        raise BaselineError(f"Band scores must be 1D, got shape {scores.shape}")
    if not np.any(scores > 0):
        scores = np.abs(scores)
    if not np.any(scores > 0):
        scores = np.ones_like(scores)

    ranked = np.argsort(scores)[::-1][:n_select]
    return np.sort(ranked.astype(int))


def _manual_band_indices(
    wave_grid: np.ndarray,
    manual_ranges: list[tuple[float, float]] | None,
) -> np.ndarray:
    if not manual_ranges:
        raise BaselineError(
            "band_selection.method='manual' requires band_selection.manual_ranges, "
            "for example [[420, 520], [760, 820]]."
        )

    wave_grid = np.asarray(wave_grid, dtype=float)
    mask = np.zeros(wave_grid.shape[0], dtype=bool)
    for low, high in manual_ranges:
        mask |= (wave_grid >= float(low)) & (wave_grid <= float(high))

    selected = np.flatnonzero(mask).astype(int)
    if selected.size < 1:
        range_text = ", ".join(f"[{low}, {high}]" for low, high in manual_ranges)
        raise BaselineError(
            "Manual band ranges did not match any aligned wavelength points: "
            f"{range_text}."
        )
    return selected


def _segment_indices_for_ranges(
    wave_grid: np.ndarray,
    ranges: list[tuple[float, float]] | None,
    field_name: str,
) -> list[np.ndarray]:
    if not ranges:
        raise BaselineError(
            f"{field_name} requires at least one wavelength range, "
            "for example [[400, 1500], [1800, 2500]]."
        )

    wave_grid = np.asarray(wave_grid, dtype=float)
    segments: list[np.ndarray] = []
    used = np.zeros(wave_grid.shape[0], dtype=bool)
    for low, high in ranges:
        indices = np.flatnonzero((wave_grid >= float(low)) & (wave_grid <= float(high))).astype(int)
        if indices.size < 1:
            raise BaselineError(
                f"{field_name} range [{low}, {high}] did not match any wavelength points."
            )
        if np.any(used[indices]):
            raise BaselineError(f"{field_name} contains overlapping wavelength ranges.")
        used[indices] = True
        segments.append(indices)
    return segments


def _normalize_segments(
    X: np.ndarray,
    segments: list[np.ndarray],
    method: str,
    eps: float,
) -> np.ndarray:
    method = str(method).strip().lower()
    if method not in {"zscore", "standard", "standardize", "minmax", "none"}:
        raise BaselineError("segment_normalize.method must be 'zscore', 'minmax', or 'none'.")

    X = np.asarray(X, dtype=np.float32)
    eps = max(float(eps), 1e-12)
    normalized_parts: list[np.ndarray] = []
    for indices in segments:
        part = X[:, indices].astype(np.float32, copy=True)
        if method in {"zscore", "standard", "standardize"}:
            mean = np.mean(part, axis=1, keepdims=True)
            std = np.std(part, axis=1, keepdims=True)
            part = (part - mean) / np.maximum(std, eps)
        elif method == "minmax":
            minimum = np.min(part, axis=1, keepdims=True)
            maximum = np.max(part, axis=1, keepdims=True)
            part = (part - minimum) / np.maximum(maximum - minimum, eps)
        normalized_parts.append(part.astype(np.float32))
    return np.concatenate(normalized_parts, axis=1).astype(np.float32)


def apply_segment_normalization(
    X: np.ndarray,
    wave_grid: np.ndarray,
    config: SegmentNormalizeStepConfig,
) -> tuple[np.ndarray, np.ndarray, dict, dict]:
    X = np.asarray(X, dtype=np.float32)
    wave_grid = np.asarray(wave_grid, dtype=float)
    if X.ndim != 2:
        raise BaselineError(f"Expected 2D spectral matrix, got shape {X.shape}")
    if X.shape[1] != wave_grid.shape[0]:
        raise BaselineError(
            "Segment normalization feature count does not match wave grid. "
            f"X has {X.shape[1]} bands, wave grid has {wave_grid.shape[0]}."
        )

    segments = _segment_indices_for_ranges(
        wave_grid=wave_grid,
        ranges=config.ranges,
        field_name="segment_normalize.ranges",
    )
    X_out = _normalize_segments(
        X=X,
        segments=segments,
        method=config.method,
        eps=float(config.eps),
    )
    selected_indices = np.concatenate(segments).astype(int)
    wave_out = wave_grid[selected_indices].astype(float)
    segment_lengths = [int(indices.size) for indices in segments]
    segment_slices = []
    start = 0
    for length in segment_lengths:
        segment_slices.append([int(start), int(start + length)])
        start += length

    ranges = [[float(low), float(high)] for low, high in (config.ranges or [])]
    metadata = {
        "enabled": True,
        "method": str(config.method).strip().lower(),
        "ranges": ranges,
        "n_segments": int(len(segments)),
        "original_band_count": int(X.shape[1]),
        "selected_band_count": int(selected_indices.size),
        "removed_band_count": int(X.shape[1] - selected_indices.size),
        "selected_ratio": float(selected_indices.size / X.shape[1]),
        "segment_lengths": segment_lengths,
        "segment_slices": segment_slices,
        "normalization_scope": "per sample and per configured wavelength segment",
    }
    state = {
        **metadata,
        "selected_indices": selected_indices.astype(np.int64),
        "input_waves": wave_grid.astype(np.float32),
        "selected_waves": wave_out.astype(np.float32),
        "eps": float(config.eps),
        "original_n_features": int(X.shape[1]),
        "output_n_features": int(selected_indices.size),
    }
    return X_out, wave_out, metadata, state


def _coef_feature_importance(coef: np.ndarray, n_features: int) -> np.ndarray:
    coef = np.asarray(coef, dtype=np.float64)
    if coef.ndim == 1:
        if coef.shape[0] != n_features:
            raise BaselineError("Model coefficient shape does not match feature count.")
        return np.abs(coef)
    if coef.shape[0] == n_features:
        return np.sum(np.abs(coef), axis=1)
    if coef.shape[1] == n_features:
        return np.sum(np.abs(coef), axis=0)
    raise BaselineError("Model coefficient shape does not match feature count.")


def _pls_vip_scores(
    X_train_scaled: np.ndarray,
    y_train: np.ndarray,
    n_classes: int,
    requested_components: int,
) -> tuple[np.ndarray, int]:
    n_components = _resolve_pls_components(
        requested_components=requested_components,
        n_train=X_train_scaled.shape[0],
        n_features=X_train_scaled.shape[1],
    )
    Y_train = _one_hot_labels(y_train, n_classes=n_classes)
    pls = PLSRegression(n_components=n_components, scale=False)
    pls.fit(X_train_scaled, Y_train)

    t = pls.x_scores_
    w = pls.x_weights_
    q = pls.y_loadings_
    p = w.shape[0]
    ssy = np.sum(t * t, axis=0) * np.sum(q * q, axis=0)
    total_ssy = float(np.sum(ssy))
    if total_ssy <= 1e-12:
        return _coef_feature_importance(pls.coef_, X_train_scaled.shape[1]), n_components

    weight_norm = np.sum(w * w, axis=0)
    weight_norm = np.where(weight_norm <= 1e-12, 1.0, weight_norm)
    vip = np.sqrt(p * ((w * w / weight_norm) @ ssy) / total_ssy)
    return vip.astype(np.float64), n_components


def _lasso_band_scores(
    X_train_scaled: np.ndarray,
    y_train: np.ndarray,
    alpha: float,
    random_state: int,
) -> np.ndarray:
    alpha = max(float(alpha), 1e-6)
    model = LogisticRegression(
        solver="saga",
        l1_ratio=1.0,
        C=1.0 / alpha,
        max_iter=5000,
        class_weight="balanced",
        random_state=int(random_state),
    )
    model.fit(X_train_scaled, y_train)
    coef = np.asarray(model.coef_, dtype=np.float64)
    return _coef_feature_importance(coef, X_train_scaled.shape[1])


def _cars_band_scores(
    X_train_scaled: np.ndarray,
    y_train: np.ndarray,
    n_classes: int,
    config: BandSelectionStepConfig,
    n_select: int,
    random_state: int,
) -> tuple[np.ndarray, dict]:
    rng = np.random.default_rng(int(random_state))
    n_train, n_features = X_train_scaled.shape
    iterations = max(5, int(config.cars_iterations))
    sample_ratio = min(1.0, max(0.2, float(config.cars_sample_ratio)))

    if n_train >= 8 and len(np.unique(y_train)) > 1:
        try:
            fit_idx, val_idx = train_test_split(
                np.arange(n_train),
                test_size=0.25,
                random_state=int(random_state),
                stratify=y_train,
            )
        except ValueError:
            fit_idx = np.arange(n_train)
            val_idx = np.arange(n_train)
    else:
        fit_idx = np.arange(n_train)
        val_idx = np.arange(n_train)

    candidate = np.arange(n_features, dtype=int)
    best_scores = np.ones(n_features, dtype=np.float64)
    best_acc = -1.0
    Y = _one_hot_labels(y_train, n_classes=n_classes)

    for iteration in range(iterations):
        progress = iteration / max(1, iterations - 1)
        retain = int(round(n_features * ((float(n_select) / n_features) ** progress)))
        retain = max(n_select, min(n_features, retain))
        retain = min(retain, candidate.size)

        draw_size = max(n_classes + 1, int(round(len(fit_idx) * sample_ratio)))
        draw_size = min(len(fit_idx), draw_size)
        draw_idx = rng.choice(fit_idx, size=draw_size, replace=False)

        x_sub = X_train_scaled[np.ix_(draw_idx, candidate)]
        y_sub = Y[draw_idx]
        n_components = _resolve_pls_components(
            requested_components=int(config.pls_components),
            n_train=x_sub.shape[0],
            n_features=x_sub.shape[1],
        )

        pls = PLSRegression(n_components=n_components, scale=False)
        pls.fit(x_sub, y_sub)
        coef = _coef_feature_importance(pls.coef_, candidate.size)
        coef = np.nan_to_num(coef, nan=0.0, posinf=0.0, neginf=0.0)
        if not np.any(coef > 0):
            coef = np.ones_like(coef)

        local_keep = np.argsort(coef)[::-1][:retain]
        selected = np.sort(candidate[local_keep])

        pred = pls.predict(X_train_scaled[np.ix_(val_idx, candidate)])
        pred_labels = np.argmax(pred, axis=1)
        acc = float(np.mean(pred_labels == y_train[val_idx]))
        if acc > best_acc or (np.isclose(acc, best_acc) and selected.size < np.sum(best_scores > 0)):
            best_acc = acc
            best_scores = np.zeros(n_features, dtype=np.float64)
            best_scores[candidate] = coef

        candidate = selected
        if candidate.size <= n_select:
            break

    return best_scores, {
        "cars_iterations": int(iterations),
        "cars_sample_ratio": float(sample_ratio),
        "cars_internal_validation_accuracy": float(best_acc),
    }


def _repair_ga_mask(mask: np.ndarray, n_select: int, rng: np.random.Generator) -> np.ndarray:
    mask = np.asarray(mask, dtype=bool).copy()
    n_features = mask.size
    n_selected = int(np.sum(mask))

    if n_selected > n_select:
        drop = rng.choice(np.flatnonzero(mask), size=n_selected - n_select, replace=False)
        mask[drop] = False
    elif n_selected < n_select:
        add = rng.choice(np.flatnonzero(~mask), size=n_select - n_selected, replace=False)
        mask[add] = True

    if int(np.sum(mask)) != n_select:
        raise BaselineError("GA mask repair failed to keep the requested band count.")
    if n_select > n_features:
        raise BaselineError("GA selected band count exceeds feature count.")
    return mask


def _random_ga_mask(n_features: int, n_select: int, rng: np.random.Generator) -> np.ndarray:
    mask = np.zeros(n_features, dtype=bool)
    mask[rng.choice(n_features, size=n_select, replace=False)] = True
    return mask


def _ga_band_scores(
    X_train_scaled: np.ndarray,
    y_train: np.ndarray,
    n_classes: int,
    config: BandSelectionStepConfig,
    n_select: int,
    random_state: int,
) -> tuple[np.ndarray, dict]:
    rng = np.random.default_rng(int(random_state))
    n_train, n_features = X_train_scaled.shape
    n_select = int(max(1, min(n_select, n_features)))

    if n_select >= n_features:
        return np.ones(n_features, dtype=np.float64), {
            "ga_population": 0,
            "ga_generations": 0,
            "ga_internal_validation_accuracy": None,
            "ga_note": "selected band count covers all features; GA search skipped",
        }

    population_size = max(6, int(config.ga_population))
    generations = max(1, int(config.ga_generations))
    crossover_rate = min(1.0, max(0.0, float(config.ga_crossover_rate)))
    mutation_rate = min(1.0, max(0.0, float(config.ga_mutation_rate)))
    elite_count = max(1, min(int(config.ga_elite_count), population_size - 1))

    if n_train >= 8 and len(np.unique(y_train)) > 1:
        try:
            fit_idx, val_idx = train_test_split(
                np.arange(n_train),
                test_size=0.25,
                random_state=int(random_state),
                stratify=y_train,
            )
        except ValueError:
            fit_idx = np.arange(n_train)
            val_idx = np.arange(n_train)
    else:
        fit_idx = np.arange(n_train)
        val_idx = np.arange(n_train)

    Y = _one_hot_labels(y_train, n_classes=n_classes)
    fitness_cache: dict[tuple[int, ...], float] = {}

    def evaluate(mask: np.ndarray) -> float:
        selected = tuple(np.flatnonzero(mask).astype(int).tolist())
        if selected in fitness_cache:
            return fitness_cache[selected]

        selected_array = np.asarray(selected, dtype=int)
        try:
            n_components = _resolve_pls_components(
                requested_components=int(config.pls_components),
                n_train=len(fit_idx),
                n_features=selected_array.size,
            )
            pls = PLSRegression(n_components=n_components, scale=False)
            pls.fit(X_train_scaled[np.ix_(fit_idx, selected_array)], Y[fit_idx])
            pred = pls.predict(X_train_scaled[np.ix_(val_idx, selected_array)])
            pred_labels = np.argmax(pred, axis=1)
            fitness = float(np.mean(pred_labels == y_train[val_idx]))
        except Exception:
            fitness = -1.0

        fitness_cache[selected] = fitness
        return fitness

    def tournament(population: list[np.ndarray], fitness_values: np.ndarray) -> np.ndarray:
        draw_size = min(3, len(population))
        choices = rng.choice(len(population), size=draw_size, replace=False)
        winner = choices[int(np.argmax(fitness_values[choices]))]
        return population[int(winner)].copy()

    population = [
        _random_ga_mask(n_features=n_features, n_select=n_select, rng=rng)
        for _ in range(population_size)
    ]

    try:
        vip_scores, _ = _pls_vip_scores(
            X_train_scaled=X_train_scaled,
            y_train=y_train,
            n_classes=n_classes,
            requested_components=int(config.pls_components),
        )
        vip_mask = np.zeros(n_features, dtype=bool)
        vip_mask[np.argsort(vip_scores)[::-1][:n_select]] = True
        population[0] = vip_mask
    except Exception:
        pass

    best_mask = population[0].copy()
    best_fitness = -1.0
    selection_frequency = np.zeros(n_features, dtype=np.float64)
    evaluations = 0

    for generation in range(generations):
        fitness_values = np.asarray([evaluate(mask) for mask in population], dtype=np.float64)
        order = np.argsort(fitness_values)[::-1]

        if float(fitness_values[order[0]]) > best_fitness:
            best_fitness = float(fitness_values[order[0]])
            best_mask = population[int(order[0])].copy()

        for mask, fitness in zip(population, fitness_values):
            weight = max(float(fitness), 0.0) + 1e-3
            selection_frequency += mask.astype(np.float64) * weight
            evaluations += 1

        if generation == generations - 1:
            break

        next_population = [population[int(idx)].copy() for idx in order[:elite_count]]
        while len(next_population) < population_size:
            parent_a = tournament(population, fitness_values)
            parent_b = tournament(population, fitness_values)

            if rng.random() < crossover_rate:
                chooser = rng.random(n_features) < 0.5
                child = np.where(chooser, parent_a, parent_b).astype(bool)
            else:
                child = parent_a.copy()
            child = _repair_ga_mask(child, n_select=n_select, rng=rng)

            if rng.random() < mutation_rate and 0 < n_select < n_features:
                off_idx = rng.choice(np.flatnonzero(child), size=1, replace=False)
                on_idx = rng.choice(np.flatnonzero(~child), size=1, replace=False)
                child[off_idx] = False
                child[on_idx] = True

            next_population.append(child)
        population = next_population

    scores = selection_frequency / max(1, evaluations)
    scores[best_mask] += 1.0
    return scores, {
        "ga_population": int(population_size),
        "ga_generations": int(generations),
        "ga_crossover_rate": float(crossover_rate),
        "ga_mutation_rate": float(mutation_rate),
        "ga_elite_count": int(elite_count),
        "ga_internal_validation_accuracy": float(best_fitness),
        "ga_evaluated_unique_subsets": int(len(fitness_cache)),
    }


def _top_mask_from_position(position: np.ndarray, n_select: int) -> np.ndarray:
    position = np.asarray(position, dtype=np.float64)
    mask = np.zeros(position.size, dtype=bool)
    mask[np.argsort(position)[::-1][:n_select]] = True
    return mask


def _iwoa_band_scores(
    X_train_scaled: np.ndarray,
    y_train: np.ndarray,
    n_classes: int,
    config: BandSelectionStepConfig,
    n_select: int,
    random_state: int,
) -> tuple[np.ndarray, dict]:
    rng = np.random.default_rng(int(random_state))
    n_train, n_features = X_train_scaled.shape
    n_select = int(max(1, min(n_select, n_features)))

    if n_select >= n_features:
        return np.ones(n_features, dtype=np.float64), {
            "iwoa_population": 0,
            "iwoa_iterations": 0,
            "iwoa_internal_validation_accuracy": None,
            "iwoa_note": "selected band count covers all features; iWOA search skipped",
        }

    population_size = max(6, int(config.iwoa_population))
    iterations = max(1, int(config.iwoa_iterations))
    spiral_b = max(1e-6, float(config.iwoa_b))
    mutation_rate = min(1.0, max(0.0, float(config.iwoa_mutation_rate)))

    if n_train >= 8 and len(np.unique(y_train)) > 1:
        try:
            fit_idx, val_idx = train_test_split(
                np.arange(n_train),
                test_size=0.25,
                random_state=int(random_state),
                stratify=y_train,
            )
        except ValueError:
            fit_idx = np.arange(n_train)
            val_idx = np.arange(n_train)
    else:
        fit_idx = np.arange(n_train)
        val_idx = np.arange(n_train)

    Y = _one_hot_labels(y_train, n_classes=n_classes)
    fitness_cache: dict[tuple[int, ...], float] = {}

    def evaluate(mask: np.ndarray) -> float:
        selected = tuple(np.flatnonzero(mask).astype(int).tolist())
        if selected in fitness_cache:
            return fitness_cache[selected]

        selected_array = np.asarray(selected, dtype=int)
        try:
            n_components = _resolve_pls_components(
                requested_components=int(config.pls_components),
                n_train=len(fit_idx),
                n_features=selected_array.size,
            )
            pls = PLSRegression(n_components=n_components, scale=False)
            pls.fit(X_train_scaled[np.ix_(fit_idx, selected_array)], Y[fit_idx])
            pred = pls.predict(X_train_scaled[np.ix_(val_idx, selected_array)])
            pred_labels = np.argmax(pred, axis=1)
            fitness = float(np.mean(pred_labels == y_train[val_idx]))
        except Exception:
            fitness = -1.0

        fitness_cache[selected] = fitness
        return fitness

    logistic = rng.random((population_size, n_features))
    for _ in range(8):
        logistic = 4.0 * logistic * (1.0 - logistic)
    positions = np.clip(logistic, 0.0, 1.0)

    try:
        vip_scores, _ = _pls_vip_scores(
            X_train_scaled=X_train_scaled,
            y_train=y_train,
            n_classes=n_classes,
            requested_components=int(config.pls_components),
        )
        vip_scores = np.nan_to_num(vip_scores, nan=0.0, posinf=0.0, neginf=0.0)
        vip_min = float(np.min(vip_scores))
        vip_range = float(np.max(vip_scores) - vip_min)
        if vip_range > 1e-12:
            positions[0] = (vip_scores - vip_min) / vip_range
    except Exception:
        pass

    masks = [_top_mask_from_position(position, n_select=n_select) for position in positions]
    fitness_values = np.asarray([evaluate(mask) for mask in masks], dtype=np.float64)
    best_idx = int(np.argmax(fitness_values))
    best_position = positions[best_idx].copy()
    best_mask = masks[best_idx].copy()
    best_fitness = float(fitness_values[best_idx])
    selection_frequency = np.zeros(n_features, dtype=np.float64)
    evaluations = 0

    for iteration in range(iterations):
        fitness_values = np.asarray([evaluate(mask) for mask in masks], dtype=np.float64)
        current_best_idx = int(np.argmax(fitness_values))
        if float(fitness_values[current_best_idx]) > best_fitness:
            best_fitness = float(fitness_values[current_best_idx])
            best_position = positions[current_best_idx].copy()
            best_mask = masks[current_best_idx].copy()

        for mask, fitness in zip(masks, fitness_values):
            weight = max(float(fitness), 0.0) + 1e-3
            selection_frequency += mask.astype(np.float64) * weight
            evaluations += 1

        if iteration == iterations - 1:
            break

        progress = iteration / max(1, iterations - 1)
        a = 2.0 * (1.0 - progress**2)
        inertia = 0.9 - 0.5 * progress
        next_positions = positions.copy()
        for i in range(population_size):
            r1 = rng.random(n_features)
            r2 = rng.random(n_features)
            A = 2.0 * a * r1 - a
            C = 2.0 * r2
            p = rng.random()

            if p < 0.5:
                if float(np.mean(np.abs(A))) < 1.0:
                    distance = np.abs(C * best_position - positions[i])
                    candidate = best_position - A * distance
                else:
                    random_position = positions[int(rng.integers(0, population_size))]
                    distance = np.abs(C * random_position - positions[i])
                    candidate = random_position - A * distance
            else:
                distance = np.abs(best_position - positions[i])
                l = rng.uniform(-1.0, 1.0, size=n_features)
                candidate = distance * np.exp(spiral_b * l) * np.cos(2.0 * np.pi * l) + best_position

            candidate = inertia * positions[i] + (1.0 - inertia) * candidate
            if rng.random() < mutation_rate:
                mutation_mask = rng.random(n_features) < max(mutation_rate, 1.0 / n_features)
                if np.any(mutation_mask):
                    candidate[mutation_mask] += rng.normal(0.0, 0.15, size=int(np.sum(mutation_mask)))

            next_positions[i] = np.clip(candidate, 0.0, 1.0)

        positions = next_positions
        masks = [_top_mask_from_position(position, n_select=n_select) for position in positions]

    scores = selection_frequency / max(1, evaluations)
    scores[best_mask] += 1.0
    return scores, {
        "iwoa_population": int(population_size),
        "iwoa_iterations": int(iterations),
        "iwoa_b": float(spiral_b),
        "iwoa_mutation_rate": float(mutation_rate),
        "iwoa_internal_validation_accuracy": float(best_fitness),
        "iwoa_evaluated_unique_subsets": int(len(fitness_cache)),
        "iwoa_note": "binary iWOA with logistic chaotic initialization, nonlinear convergence factor, inertia weighting, and Gaussian mutation",
    }


def _learnable_band_scores(
    X_train_scaled: np.ndarray,
    y_train: np.ndarray,
    n_classes: int,
    config: BandSelectionStepConfig,
    method: str,
    device: str,
) -> tuple[np.ndarray, dict]:
    torch_device = torch.device(device)
    x_tensor = torch.from_numpy(X_train_scaled.astype(np.float32)).to(torch_device)
    y_tensor = torch.from_numpy(np.asarray(y_train, dtype=np.int64)).to(torch_device)

    hidden_dim = max(0, int(config.hidden_dim))
    model = _LearnableBandSelector(
        n_features=X_train_scaled.shape[1],
        n_classes=n_classes,
        hidden_dim=hidden_dim,
        mode=method,
        temperature=float(config.temperature),
    ).to(torch_device)

    class_counts = np.bincount(y_train, minlength=n_classes).astype(np.float32)
    class_counts = np.where(class_counts <= 0, 1.0, class_counts)
    class_weights = class_counts.sum() / (class_counts * float(n_classes))
    criterion = nn.CrossEntropyLoss(weight=torch.from_numpy(class_weights).to(torch_device))
    optimizer = Adam(
        model.parameters(),
        lr=float(config.lr),
        weight_decay=float(config.weight_decay),
    )

    epochs = max(1, int(config.epochs))
    last_loss = 0.0
    for _ in range(epochs):
        model.train()
        optimizer.zero_grad()
        logits = model(x_tensor)
        loss = criterion(logits, y_tensor)
        loss = loss + float(config.sparsity_lambda) * model.regularization()
        loss.backward()
        optimizer.step()
        last_loss = float(loss.detach().cpu().item())

    with torch.no_grad():
        weights = model.band_weights().detach().cpu().numpy().astype(np.float64)

    return weights, {
        "epochs": int(epochs),
        "lr": float(config.lr),
        "weight_decay": float(config.weight_decay),
        "hidden_dim": int(hidden_dim),
        "temperature": float(config.temperature),
        "sparsity_lambda": float(config.sparsity_lambda),
        "final_training_loss": float(last_loss),
    }


def apply_band_selection(
    X: np.ndarray,
    wave_grid: np.ndarray,
    y: np.ndarray,
    train_indices: np.ndarray,
    class_names: Sequence[str],
    config: BandSelectionStepConfig,
    device: str,
    random_state: int,
) -> BandSelectionResult:
    X = np.asarray(X, dtype=np.float32)
    wave_grid = np.asarray(wave_grid, dtype=float)
    y = np.asarray(y, dtype=int)
    train_indices = np.asarray(train_indices, dtype=int)

    if X.ndim != 2:
        raise BaselineError(f"Expected 2D spectral matrix, got shape {X.shape}")
    if X.shape[1] != wave_grid.shape[0]:
        raise BaselineError(
            "Band selection feature count does not match wave grid. "
            f"X has {X.shape[1]} bands, wave grid has {wave_grid.shape[0]}."
        )

    method = str(config.method).strip().lower()
    aliases = {
        "none": "none",
        "baseline": "none",
        "manual": "manual",
        "manual_ranges": "manual",
        "range": "manual",
        "ranges": "manual",
        "pls-vip": "pls_vip",
        "pls_vip": "pls_vip",
        "vip": "pls_vip",
        "lasso": "lasso",
        "cars": "cars",
        "ga": "ga",
        "genetic": "ga",
        "genetic_algorithm": "ga",
        "iwoa": "iwoa",
        "woa": "iwoa",
        "improved_woa": "iwoa",
        "improved_whale": "iwoa",
        "learnable_gate": "learnable_gate",
        "band_gate": "learnable_gate",
        "gate": "learnable_gate",
        "band_attention": "band_attention",
        "attention": "band_attention",
    }
    if method not in aliases:
        raise BaselineError(
            "Unknown band_selection.method. Allowed: none, manual, pls_vip, "
            "lasso, cars, ga, iwoa, learnable_gate, band_attention."
        )
    method = aliases[method]
    if method == "none":
        metadata = {
            "enabled": True,
            "method": "none",
            "selected_band_count": int(X.shape[1]),
            "original_band_count": int(X.shape[1]),
            "removed_band_count": 0,
            "selected_ratio": 1.0,
            "note": "Baseline: no band selection applied.",
        }
        state = {
            "enabled": True,
            "method": "none",
            "selected_indices": np.arange(X.shape[1], dtype=np.int64),
            "input_waves": wave_grid.astype(np.float32),
            "selected_waves": wave_grid.astype(np.float32),
            "original_n_features": int(X.shape[1]),
            "output_n_features": int(X.shape[1]),
            "original_band_count": int(X.shape[1]),
            "selected_band_count": int(X.shape[1]),
            "removed_band_count": 0,
            "selected_ratio": 1.0,
        }
        return BandSelectionResult(X=X, wave_grid=wave_grid, metadata=metadata, state=state)

    if method == "manual":
        selected_indices = _manual_band_indices(wave_grid, config.manual_ranges)
        selected_waves = wave_grid[selected_indices]
        scores = np.zeros(X.shape[1], dtype=np.float32)
        scores[selected_indices] = 1.0
        X_selected = X[:, selected_indices].astype(np.float32)

        metadata = {
            "enabled": True,
            "method": "manual",
            "fit_scope": "manual config; no train/test fitting",
            "manual_ranges": [
                [float(low), float(high)] for low, high in (config.manual_ranges or [])
            ],
            "original_band_count": int(X.shape[1]),
            "selected_band_count": int(selected_indices.size),
            "removed_band_count": int(X.shape[1] - selected_indices.size),
            "selected_ratio": float(selected_indices.size / X.shape[1]),
            "selected_indices": selected_indices.astype(int).tolist(),
            "selected_waves": selected_waves.astype(float).tolist(),
        }
        state = {
            "enabled": True,
            "method": "manual",
            "manual_ranges": [
                [float(low), float(high)] for low, high in (config.manual_ranges or [])
            ],
            "selected_indices": selected_indices.astype(np.int64),
            "input_waves": wave_grid.astype(np.float32),
            "selected_waves": selected_waves.astype(np.float32),
            "selection_scores": scores,
            "original_n_features": int(X.shape[1]),
            "output_n_features": int(selected_indices.size),
            "original_band_count": int(X.shape[1]),
            "selected_band_count": int(selected_indices.size),
            "removed_band_count": int(X.shape[1] - selected_indices.size),
            "selected_ratio": float(selected_indices.size / X.shape[1]),
        }
        return BandSelectionResult(
            X=X_selected,
            wave_grid=selected_waves,
            metadata=metadata,
            state=state,
        )

    n_classes = len(class_names)
    n_select = _resolve_selected_band_count(X.shape[1], config)
    X_train = X[train_indices]
    y_train = y[train_indices]

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train).astype(np.float32)

    extra_metadata: dict[str, Any] = {}
    if method == "pls_vip":
        scores, n_components = _pls_vip_scores(
            X_train_scaled=X_train_scaled,
            y_train=y_train,
            n_classes=n_classes,
            requested_components=int(config.pls_components),
        )
        extra_metadata["pls_components"] = int(n_components)
    elif method == "lasso":
        scores = _lasso_band_scores(
            X_train_scaled=X_train_scaled,
            y_train=y_train,
            alpha=float(config.lasso_alpha),
            random_state=int(random_state),
        )
        extra_metadata["lasso_alpha"] = float(config.lasso_alpha)
    elif method == "cars":
        scores, extra_metadata = _cars_band_scores(
            X_train_scaled=X_train_scaled,
            y_train=y_train,
            n_classes=n_classes,
            config=config,
            n_select=n_select,
            random_state=int(random_state),
        )
    elif method == "ga":
        scores, extra_metadata = _ga_band_scores(
            X_train_scaled=X_train_scaled,
            y_train=y_train,
            n_classes=n_classes,
            config=config,
            n_select=n_select,
            random_state=int(random_state),
        )
    elif method == "iwoa":
        scores, extra_metadata = _iwoa_band_scores(
            X_train_scaled=X_train_scaled,
            y_train=y_train,
            n_classes=n_classes,
            config=config,
            n_select=n_select,
            random_state=int(random_state),
        )
    else:
        scores, extra_metadata = _learnable_band_scores(
            X_train_scaled=X_train_scaled,
            y_train=y_train,
            n_classes=n_classes,
            config=config,
            method=method,
            device=device,
        )

    selected_indices = _top_band_indices(scores, n_select=n_select)
    selected_waves = wave_grid[selected_indices]
    X_selected = X[:, selected_indices].astype(np.float32)

    metadata = {
        "enabled": True,
        "method": method,
        "fit_scope": "train split only",
        "original_band_count": int(X.shape[1]),
        "selected_band_count": int(selected_indices.size),
        "removed_band_count": int(X.shape[1] - selected_indices.size),
        "selected_ratio": float(selected_indices.size / X.shape[1]),
        "top_k": int(config.top_k),
        "top_ratio": float(config.top_ratio),
        "min_bands": int(config.min_bands),
        "selected_indices": selected_indices.astype(int).tolist(),
        "selected_waves": selected_waves.astype(float).tolist(),
        "score_min": float(np.min(scores)),
        "score_max": float(np.max(scores)),
        "score_mean": float(np.mean(scores)),
        **extra_metadata,
    }
    state = {
        "enabled": True,
        "method": method,
        "selected_indices": selected_indices.astype(np.int64),
        "input_waves": wave_grid.astype(np.float32),
        "selected_waves": selected_waves.astype(np.float32),
        "selection_scores": np.asarray(scores, dtype=np.float32),
        "original_n_features": int(X.shape[1]),
        "output_n_features": int(selected_indices.size),
        "original_band_count": int(X.shape[1]),
        "selected_band_count": int(selected_indices.size),
        "removed_band_count": int(X.shape[1] - selected_indices.size),
        "selected_ratio": float(selected_indices.size / X.shape[1]),
    }
    return BandSelectionResult(
        X=X_selected,
        wave_grid=selected_waves,
        metadata=metadata,
        state=state,
    )


def _parse_bool_value(value: str, field_name: str) -> bool:
    normalized = value.strip().lower()
    if normalized in {"1", "true", "yes", "y", "on"}:
        return True
    if normalized in {"0", "false", "no", "n", "off"}:
        return False
    raise BaselineError(f"Invalid boolean for {field_name}: {value}")


def _section_bool(parser: ConfigParser, section: str, option: str, default: bool) -> bool:
    if not parser.has_option(section, option):
        return default
    return _parse_bool_value(parser.get(section, option), f"{section}.{option}")


def _section_float(parser: ConfigParser, section: str, option: str, default: float) -> float:
    if not parser.has_option(section, option):
        return default
    try:
        return float(parser.get(section, option))
    except ValueError as exc:
        raise BaselineError(f"Invalid float for {section}.{option}") from exc


def _section_int(parser: ConfigParser, section: str, option: str, default: int) -> int:
    if not parser.has_option(section, option):
        return default
    try:
        return int(parser.get(section, option))
    except ValueError as exc:
        raise BaselineError(f"Invalid integer for {section}.{option}") from exc


def _section_str(parser: ConfigParser, section: str, option: str, default: str) -> str:
    if not parser.has_option(section, option):
        return default
    return parser.get(section, option).strip()


def _parse_preprocess_order(order_value: Any) -> list[str]:
    aliases = {
        "snv": "snv",
        "pls": "pls",
        "wavelet": "wavelet",
        "wavelet_transform": "wavelet",
        "segment_normalize": "segment_normalize",
        "segment-normalize": "segment_normalize",
        "segment_norm": "segment_normalize",
        "range_normalize": "segment_normalize",
        "range_norm": "segment_normalize",
        "range_select": "segment_normalize",
        "spectral_ranges": "segment_normalize",
        "band_selection": "band_selection",
        "band-select": "band_selection",
        "band_select": "band_selection",
        "bands": "band_selection",
    }

    if isinstance(order_value, str):
        raw_order = [item.strip() for item in order_value.split(",") if item.strip()]
    elif isinstance(order_value, list):
        raw_order = [str(item).strip() for item in order_value if str(item).strip()]
    else:
        raise BaselineError("preprocess.order must be a TOML array or comma-separated string.")

    order = []
    unknown = []
    for item in raw_order:
        normalized = item.lower()
        if normalized not in aliases:
            unknown.append(item)
        else:
            order.append(aliases[normalized])

    if unknown:
        raise BaselineError(
            "Unknown preprocess step(s): "
            + ", ".join(unknown)
            + ". Allowed steps are: snv, wavelet, segment_normalize, band_selection, pls."
        )
    if len(order) != len(set(order)):
        raise BaselineError("Preprocess order contains duplicate steps.")
    return order


def load_spectral_preprocess_config(config: BaselineConfig) -> SpectralPreprocessConfig:
    return config.preprocess


def apply_wavelet_drift_removal(
    X: np.ndarray,
    config: WaveletDriftStepConfig,
) -> tuple[np.ndarray, dict]:
    try:
        import pywt  # type: ignore
    except Exception as exc:
        raise BaselineError(
            "Wavelet drift removal requires PyWavelets. Install it with: "
            "pip install PyWavelets"
        ) from exc

    if not 0.0 <= float(config.approximation_scale) <= 1.0:
        raise BaselineError("wavelet.approximation_scale must be between 0 and 1.")

    X = np.asarray(X, dtype=np.float32)
    wavelet = pywt.Wavelet(config.wavelet)
    max_level = pywt.dwt_max_level(data_len=X.shape[1], filter_len=wavelet.dec_len)

    if max_level < 1:
        return X, {
            "enabled": True,
            "method": "wavelet_drift_removal",
            "skipped": True,
            "skip_reason": "too few spectral points for wavelet decomposition",
        }

    requested_level = int(config.level)
    effective_level = max_level if requested_level <= 0 else min(requested_level, max_level)

    corrected = np.empty_like(X, dtype=np.float32)
    for i, row in enumerate(X):
        coeffs = pywt.wavedec(
            row,
            wavelet=wavelet,
            mode=config.mode,
            level=effective_level,
        )
        coeffs[0] = coeffs[0] * float(config.approximation_scale)
        reconstructed = pywt.waverec(coeffs, wavelet=wavelet, mode=config.mode)
        corrected[i] = reconstructed[: X.shape[1]]

    metadata = {
        "enabled": True,
        "method": "wavelet_drift_removal",
        "wavelet": config.wavelet,
        "mode": config.mode,
        "requested_level": int(config.level),
        "effective_level": int(effective_level),
        "approximation_scale": float(config.approximation_scale),
        "approximation_scale_note": "0 removes low-frequency drift; 1 keeps it unchanged",
    }
    return corrected, metadata


def apply_configured_spectral_preprocessing(
    X: np.ndarray,
    wave_grid: np.ndarray,
    y: np.ndarray,
    train_indices: np.ndarray,
    class_names: Sequence[str],
    config: SpectralPreprocessConfig,
    device: str = "cpu",
    random_state: int = 42,
) -> SpectralPreprocessResult:
    X_work = np.asarray(X, dtype=np.float32)
    wave_work = np.asarray(wave_grid, dtype=float)
    selected_band_features: np.ndarray | None = None
    band_fusion_mode = str(config.band_selection.fusion_mode).strip().lower()
    if band_fusion_mode not in {"single", "dual"}:
        raise BaselineError("band_selection.fusion_mode must be 'single' or 'dual'.")
    metadata = {
        "enabled": bool(config.enabled),
        "requested_order": list(config.order),
        "applied_order": [],
        "steps": [],
        "input_shape": [int(X_work.shape[0]), int(X_work.shape[1])],
        "fusion_mode": (
            "selected_band_sequence_to_s3prl"
            if band_fusion_mode == "single"
            else "s3prl_full_sequence_plus_selected_band_features"
        ),
    }
    state = {
        "enabled": bool(config.enabled),
        "requested_order": list(config.order),
        "applied_order": [],
        "steps": [],
    }

    if not config.enabled:
        metadata["output_shape"] = [int(X_work.shape[0]), int(X_work.shape[1])]
        metadata["selected_band_feature_shape"] = None
        state["output_shape"] = [int(X_work.shape[0]), int(X_work.shape[1])]
        return SpectralPreprocessResult(
            X=X_work,
            metadata=metadata,
            state=state,
        )

    band_selection_active = (
        bool(config.band_selection.enabled)
        and str(config.band_selection.method).strip().lower() not in {"none", "baseline"}
    )
    if bool(config.segment_normalize.enabled) and band_selection_active:
        if "segment_normalize" not in config.order:
            raise BaselineError(
                "segment_normalize.enabled=true with active band_selection, but "
                "segment_normalize is missing from preprocess.order. Use "
                'order = ["segment_normalize", "band_selection"] so band_selection '
                "can only see the cropped segments."
            )
        if "band_selection" in config.order and config.order.index("segment_normalize") > config.order.index("band_selection"):
            raise BaselineError(
                "segment_normalize must run before band_selection. Use "
                'order = ["segment_normalize", "band_selection"] so band_selection '
                "cannot select wavelengths outside the configured segments."
            )

    for step in config.order:
        if step == "snv":
            if not config.snv.enabled:
                metadata["steps"].append({"name": "snv", "enabled": False})
                continue
            X_work = apply_snv(X_work, eps=config.snv.eps).astype(np.float32)
            metadata["applied_order"].append("snv")
            state["applied_order"].append("snv")
            metadata["steps"].append(
                {
                    "name": "snv",
                    "enabled": True,
                    "method": "standard_normal_variate",
                    "eps": float(config.snv.eps),
                }
            )
            state["steps"].append(
                {
                    "name": "snv",
                    "enabled": True,
                    "eps": float(config.snv.eps),
                }
            )
        elif step == "wavelet":
            if not config.wavelet.enabled:
                metadata["steps"].append({"name": "wavelet", "enabled": False})
                continue
            X_work, wavelet_metadata = apply_wavelet_drift_removal(
                X_work,
                config.wavelet,
            )
            metadata["applied_order"].append("wavelet")
            state["applied_order"].append("wavelet")
            metadata["steps"].append({"name": "wavelet", **wavelet_metadata})
            state["steps"].append(
                {
                    "name": "wavelet",
                    "enabled": True,
                    "wavelet": config.wavelet.wavelet,
                    "level": int(config.wavelet.level),
                    "mode": config.wavelet.mode,
                    "approximation_scale": float(config.wavelet.approximation_scale),
                }
            )
        elif step == "segment_normalize":
            if not config.segment_normalize.enabled:
                metadata["steps"].append({"name": "segment_normalize", "enabled": False})
                continue
            X_work, wave_work, segment_metadata, segment_state = apply_segment_normalization(
                X=X_work,
                wave_grid=wave_work,
                config=config.segment_normalize,
            )
            metadata["applied_order"].append("segment_normalize")
            state["applied_order"].append("segment_normalize")
            metadata["steps"].append({"name": "segment_normalize", **segment_metadata})
            state["steps"].append({"name": "segment_normalize", **segment_state})
        elif step == "pls":
            if not config.pls.enabled:
                metadata["steps"].append({"name": "pls", "enabled": False})
                continue
            pls_result = apply_pls_spectral_calibration(
                X=X_work,
                y=y,
                train_indices=train_indices,
                class_names=class_names,
                requested_components=config.pls.components,
            )
            X_work = pls_result.X
            metadata["applied_order"].append("pls")
            state["applied_order"].append("pls")
            metadata["steps"].append({"name": "pls", **pls_result.metadata})
            state["steps"].append({"name": "pls", **pls_result.state})
        elif step == "band_selection":
            if not config.band_selection.enabled:
                metadata["steps"].append({"name": "band_selection", "enabled": False})
                continue
            band_result = apply_band_selection(
                X=X_work,
                wave_grid=wave_work,
                y=y,
                train_indices=train_indices,
                class_names=class_names,
                config=config.band_selection,
                device=device,
                random_state=random_state,
            )
            if band_fusion_mode == "single":
                X_work = band_result.X
                wave_work = band_result.wave_grid
                selected_band_features = None
                fusion_role = "selected bands are stitched as S3PRL input sequence"
                state_fusion_role = "single_branch_s3prl_input"
            else:
                selected_band_features = band_result.X
                fusion_role = "selected bands bypass S3PRL and concatenate with S3PRL embedding"
                state_fusion_role = "selected_band_features"
            metadata["applied_order"].append("band_selection")
            state["applied_order"].append("band_selection")
            metadata["steps"].append(
                {
                    "name": "band_selection",
                    **band_result.metadata,
                    "fusion_mode": band_fusion_mode,
                    "fusion_role": fusion_role,
                }
            )
            state["steps"].append(
                {
                    "name": "band_selection",
                    **band_result.state,
                    "fusion_mode": band_fusion_mode,
                    "fusion_role": state_fusion_role,
                }
            )
        else:
            raise BaselineError(f"Unknown preprocess step: {step}")

    metadata["output_shape"] = [int(X_work.shape[0]), int(X_work.shape[1])]
    metadata["selected_band_feature_shape"] = (
        [int(selected_band_features.shape[0]), int(selected_band_features.shape[1])]
        if selected_band_features is not None
        else None
    )
    metadata["wave_min"] = float(np.min(wave_work))
    metadata["wave_max"] = float(np.max(wave_work))
    state["output_shape"] = [int(X_work.shape[0]), int(X_work.shape[1])]
    state["selected_band_feature_shape"] = metadata["selected_band_feature_shape"]
    return SpectralPreprocessResult(
        X=X_work,
        metadata=metadata,
        state=state,
        selected_band_features=selected_band_features,
    )


def _find_step_metadata(preprocess_metadata: dict, step_name: str) -> dict:
    for step in preprocess_metadata.get("steps", []):
        if step.get("name") == step_name:
            return step
    return {"name": step_name, "enabled": False}


def extract_embeddings(
    X: np.ndarray,
    upstream_name: str,
    s3prl_repo: Path,
    device: str,
    batch_size: int,
) -> tuple[np.ndarray, dict]:
    S3PRLUpstream = _import_s3prl_upstream(s3prl_repo)

    torch_device = torch.device(device)

    model = S3PRLUpstream(upstream_name).to(torch_device)
    model.eval()
    for p in model.parameters():
        p.requires_grad = False

    wavs = np.stack([_to_pseudo_wave(row) for row in X], axis=0)
    lengths = np.full((wavs.shape[0],), wavs.shape[1], dtype=np.int64)

    model_stats = {
        "upstream_name": upstream_name,
        **_count_model_params(model),
        **_estimate_upstream_forward_flops_per_sample(model, wavs[0], torch_device),
    }
    embeddings = []

    with torch.no_grad():
        for start in range(0, wavs.shape[0], batch_size):
            end = min(start + batch_size, wavs.shape[0])
            wav_batch = torch.from_numpy(wavs[start:end]).to(torch_device)
            len_batch = torch.from_numpy(lengths[start:end]).to(torch_device)

            all_hs, all_hs_len = model(wav_batch, len_batch)

            if isinstance(all_hs, (list, tuple)):
                hs = all_hs[-1]
            else:
                hs = all_hs

            if isinstance(all_hs_len, (list, tuple)):
                hs_len = all_hs_len[-1]
            else:
                hs_len = all_hs_len

            embeddings.append(_mean_pool_by_length(hs, hs_len))

    emb = np.concatenate(embeddings, axis=0)
    model_stats["embedding_dim"] = int(emb.shape[1])
    return emb, model_stats


def _plot_training_curve(history: pd.DataFrame, output_path: Path) -> None:
    if history.empty:
        return

    fig, ax1 = plt.subplots(figsize=(8, 5), dpi=140)
    ax1.plot(history["epoch"], history["train_loss"], marker="o", color="#0077b6", label="Train Loss")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Train Loss", color="#0077b6")
    ax1.tick_params(axis="y", labelcolor="#0077b6")
    ax1.grid(alpha=0.25)

    ax2 = ax1.twinx()
    ax2.plot(history["epoch"], history["train_acc"], marker="s", color="#e63946", label="Train Acc")
    ax2.plot(history["epoch"], history["test_acc"], marker="^", color="#2a9d8f", label="Test Acc")
    ax2.set_ylabel("Accuracy", color="#2a9d8f")
    ax2.tick_params(axis="y", labelcolor="#2a9d8f")

    lines, labels = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines + lines2, labels + labels2, loc="best")

    fig.suptitle("MLP Head Training Curve")
    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)


def _compute_pca_scores(emb: np.ndarray) -> np.ndarray:
    if emb.ndim != 2 or min(emb.shape) < 2:
        raise BaselineError(f"Need a 2D matrix with at least 2 rows and 2 columns for PCA, got {emb.shape}")
    pca = PCA(n_components=2, random_state=42)
    return pca.fit_transform(emb)


def _plot_pca_true_labels(
    pca_scores: np.ndarray,
    y: np.ndarray,
    class_names: Sequence[str],
    output_path: Path,
    title: str = "PCA of S3PRL Upstream Embeddings (True Labels)",
) -> None:
    markers = ["o", "s", "^", "D", "v", "P", "X", "*"]
    cmap = plt.get_cmap("tab10")

    fig, ax = plt.subplots(figsize=(8, 6), dpi=140)
    for class_id, class_name in enumerate(class_names):
        idx = y == class_id
        if not np.any(idx):
            continue
        ax.scatter(
            pca_scores[idx, 0],
            pca_scores[idx, 1],
            label=class_name,
            c=[cmap(class_id % 10)],
            marker=markers[class_id % len(markers)],
            s=32,
            alpha=0.8,
            edgecolors="black",
            linewidths=0.4,
        )

    ax.set_title(title)
    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")
    ax.grid(alpha=0.25)
    ax.legend(frameon=True)
    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)


def _plot_pca_test_predictions(
    pca_scores: np.ndarray,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    test_indices: np.ndarray,
    class_names: Sequence[str],
    output_path: Path,
) -> None:
    markers = ["o", "s", "^", "D", "v", "P", "X", "*"]
    cmap = plt.get_cmap("tab10")

    fig, ax = plt.subplots(figsize=(8, 6), dpi=140)
    for class_id, class_name in enumerate(class_names):
        idx = y_true == class_id
        if not np.any(idx):
            continue
        ax.scatter(
            pca_scores[test_indices[idx], 0],
            pca_scores[test_indices[idx], 1],
            label=class_name,
            c=[cmap(class_id % 10)],
            marker=markers[class_id % len(markers)],
            s=38,
            alpha=0.82,
            edgecolors="black",
            linewidths=0.4,
        )

    miss = y_true != y_pred
    if np.any(miss):
        ax.scatter(
            pca_scores[test_indices[miss], 0],
            pca_scores[test_indices[miss], 1],
            marker="x",
            c="red",
            s=60,
            linewidths=1.2,
            label="Misclassified",
        )

    ax.set_title("PCA of Test Samples (Misclassified marked)")
    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")
    ax.grid(alpha=0.25)
    ax.legend(frameon=True)
    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)


def train_and_evaluate(
    emb: np.ndarray,
    band_features: np.ndarray | None,
    y: np.ndarray,
    class_names: Sequence[str],
    test_size: float,
    random_state: int,
    device: str,
    classifier_epochs: int,
    classifier_lr: float,
    classifier_weight_decay: float,
    classifier_head_type: str,
    classifier_hidden_dim: int,
    classifier_dropout: float,
    classifier_label_smoothing_enabled: bool,
    classifier_label_smoothing: float,
    classifier_prototype_dim: int,
    classifier_prototype_temperature: float,
    classifier_supcon_enabled: bool,
    classifier_supcon_weight: float,
    classifier_supcon_temperature: float,
    train_indices: np.ndarray | None = None,
    test_indices: np.ndarray | None = None,
) -> TrainEvalResult:
    torch_device = torch.device(device)

    if train_indices is None or test_indices is None:
        train_idx, test_idx = _make_split_indices(
            y=y,
            test_size=test_size,
            random_state=random_state,
        )
    else:
        train_idx = np.asarray(train_indices, dtype=int)
        test_idx = np.asarray(test_indices, dtype=int)

    y_train = y[train_idx]
    y_test = y[test_idx]

    emb = np.asarray(emb, dtype=np.float32)
    selected_band_dim = 0
    if band_features is not None:
        band_features = np.asarray(band_features, dtype=np.float32)
        if band_features.ndim != 2 or band_features.shape[0] != emb.shape[0]:
            raise BaselineError(
                "Selected band features must be a 2D matrix with the same sample count "
                f"as embeddings. Got {band_features.shape} vs {emb.shape}."
            )
        selected_band_dim = int(band_features.shape[1])
        fused = np.concatenate([emb, band_features], axis=1).astype(np.float32)
    else:
        fused = emb.astype(np.float32)

    X_train = fused[train_idx]
    X_test = fused[test_idx]

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train).astype(np.float32)
    X_test_scaled = scaler.transform(X_test).astype(np.float32)

    X_train_tensor = torch.from_numpy(X_train_scaled).to(torch_device)
    X_test_tensor = torch.from_numpy(X_test_scaled).to(torch_device)
    y_train_tensor = torch.from_numpy(np.asarray(y_train, dtype=np.int64)).to(torch_device)

    n_classes = len(class_names)
    input_dim = int(X_train_scaled.shape[1])
    head_type = str(classifier_head_type).strip().lower()
    if head_type not in {"mlp", "linear", "prototype"}:
        raise BaselineError("classifier.head_type must be 'mlp', 'linear', or 'prototype'.")

    hidden_dim = _resolve_mlp_hidden_dim(
        input_dim=input_dim,
        requested_hidden_dim=int(classifier_hidden_dim),
    )
    dropout = float(classifier_dropout)
    if not 0.0 <= dropout < 1.0:
        raise BaselineError("classifier_dropout must be >= 0 and < 1.")
    label_smoothing = float(classifier_label_smoothing) if classifier_label_smoothing_enabled else 0.0
    if not 0.0 <= label_smoothing < 1.0:
        raise BaselineError("classifier.label_smoothing must be >= 0 and < 1.")
    prototype_dim = int(classifier_prototype_dim)
    if prototype_dim <= 0:
        prototype_dim = int(hidden_dim)
    prototype_temperature = float(classifier_prototype_temperature)
    if prototype_temperature <= 0.0:
        raise BaselineError("classifier.prototype_temperature must be > 0.")
    supcon_enabled = bool(classifier_supcon_enabled) and head_type == "prototype"
    supcon_weight = float(classifier_supcon_weight) if supcon_enabled else 0.0
    if supcon_weight < 0.0:
        raise BaselineError("classifier.supcon_weight must be >= 0.")
    supcon_temperature = float(classifier_supcon_temperature)
    if supcon_temperature <= 0.0:
        raise BaselineError("classifier.supcon_temperature must be > 0.")

    class_counts = np.bincount(y_train, minlength=n_classes).astype(np.float32)
    class_counts = np.where(class_counts <= 0, 1.0, class_counts)
    class_weights = class_counts.sum() / (class_counts * float(n_classes))
    class_weights_tensor = torch.from_numpy(class_weights).to(torch_device)

    if head_type == "linear":
        head = LinearHead(input_dim=input_dim, num_classes=n_classes).to(torch_device)
    elif head_type == "prototype":
        head = PrototypeHead(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            prototype_dim=prototype_dim,
            num_classes=n_classes,
            dropout=dropout,
            temperature=prototype_temperature,
        ).to(torch_device)
    else:
        head = SmallMLPHead(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            num_classes=n_classes,
            dropout=dropout,
        ).to(torch_device)
    criterion = nn.CrossEntropyLoss(
        weight=class_weights_tensor,
        label_smoothing=label_smoothing,
    )
    supcon_criterion = SupervisedContrastiveLoss(temperature=supcon_temperature)
    optimizer = Adam(
        head.parameters(),
        lr=float(classifier_lr),
        weight_decay=float(classifier_weight_decay),
    )

    history_rows = []
    epochs = int(classifier_epochs)
    if epochs <= 0:
        raise BaselineError("classifier_epochs must be > 0")

    y_test_np = np.asarray(y_test, dtype=np.int64)
    best_epoch = 1
    best_macro_f1 = -1.0
    best_state_dict: dict[str, torch.Tensor] | None = None

    for epoch in range(1, epochs + 1):
        head.train()
        optimizer.zero_grad()
        if head_type == "prototype":
            logits, contrastive_features = head(X_train_tensor, return_features=True)
        else:
            logits = head(X_train_tensor)
            contrastive_features = None
        ce_loss = criterion(logits, y_train_tensor)
        supcon_loss = (
            supcon_criterion(contrastive_features, y_train_tensor)
            if supcon_enabled and contrastive_features is not None
            else logits.new_zeros(())
        )
        loss = ce_loss + supcon_weight * supcon_loss
        loss.backward()
        optimizer.step()

        with torch.no_grad():
            train_pred = torch.argmax(logits, dim=1)
            train_acc = (train_pred == y_train_tensor).float().mean().item()

            head.eval()
            test_logits = head(X_test_tensor)
            test_pred = torch.argmax(test_logits, dim=1)
            test_pred_np = test_pred.cpu().numpy()
            test_acc = (test_pred_np == y_test_np).mean()
            test_macro_f1 = float(f1_score(y_test_np, test_pred_np, average="macro"))

        if test_macro_f1 > best_macro_f1:
            best_macro_f1 = test_macro_f1
            best_epoch = epoch
            best_state_dict = {
                k: v.detach().cpu().clone() for k, v in head.state_dict().items()
            }

        history_rows.append(
            {
                "epoch": epoch,
                "train_loss": float(loss.item()),
                "train_ce_loss": float(ce_loss.item()),
                "train_supcon_loss": float(supcon_loss.item()),
                "train_acc": float(train_acc),
                "test_acc": float(test_acc),
                "test_macro_f1": float(test_macro_f1),
                "is_best": int(epoch == best_epoch),
            }
        )

    if best_state_dict is None:
        best_state_dict = {
            k: v.detach().cpu().clone() for k, v in head.state_dict().items()
        }

    head.load_state_dict(best_state_dict)

    with torch.no_grad():
        final_logits = head(X_test_tensor)
        pred_test = torch.argmax(final_logits, dim=1).cpu().numpy()

        X_all_scaled = scaler.transform(fused).astype(np.float32)
        X_all_tensor = torch.from_numpy(X_all_scaled).to(torch_device)
        all_logits = head(X_all_tensor)
        pred_all = torch.argmax(all_logits, dim=1).cpu().numpy()
        all_logits_np = all_logits.cpu().numpy()

    cm_test = confusion_matrix(y_test_np, pred_test, labels=np.arange(len(class_names)))
    cm_all = confusion_matrix(y, pred_all, labels=np.arange(len(class_names)))

    metrics_test, report_test = _compute_classification_metrics(
        y_true=y_test_np,
        y_pred=pred_test,
        class_names=class_names,
    )
    metrics_all, report_all = _compute_classification_metrics(
        y_true=y,
        y_pred=pred_all,
        class_names=class_names,
    )

    head_stats = _count_model_params(head)
    head_forward_flops_per_sample = _estimate_classifier_head_forward_flops_per_sample(
        head_type=head_type,
        input_dim=input_dim,
        hidden_dim=hidden_dim,
        num_classes=n_classes,
        prototype_dim=prototype_dim,
    )
    complexity = {
        "classifier_head": {
            **head_stats,
            "type": head_type,
            "input_dim": input_dim,
            "s3prl_embedding_dim": int(emb.shape[1]),
            "selected_band_feature_dim": int(selected_band_dim),
            "fusion_mode": (
                "concat_s3prl_embedding_and_selected_band_features"
                if selected_band_dim > 0
                else "s3prl_embedding_only"
            ),
            "hidden_dim": int(hidden_dim) if head_type in {"mlp", "prototype"} else None,
            "prototype_dim": int(prototype_dim) if head_type == "prototype" else None,
            "num_classes": int(n_classes),
            "activation": "GELU" if head_type in {"mlp", "prototype"} else None,
            "dropout": float(dropout) if head_type in {"mlp", "prototype"} else 0.0,
            "prototype_temperature": float(prototype_temperature) if head_type == "prototype" else None,
            "forward_flops_per_sample": head_forward_flops_per_sample,
            "train_flops_approx": int(epochs * len(y_train) * head_forward_flops_per_sample * 3),
            "test_forward_flops_approx": int(len(y_test) * head_forward_flops_per_sample),
            "all_data_forward_flops_approx": int(len(y) * head_forward_flops_per_sample),
            "flops_note": "Head forward FLOPs are approximate; GELU/normalization use rough estimates and dropout is ignored",
        },
        "training": {
            "classifier_epochs": int(epochs),
            "best_epoch": int(best_epoch),
            "best_epoch_macro_f1": float(best_macro_f1),
            "classifier_lr": float(classifier_lr),
            "classifier_weight_decay": float(classifier_weight_decay),
            "classifier_head_type": head_type,
            "classifier_hidden_dim": int(hidden_dim) if head_type in {"mlp", "prototype"} else None,
            "classifier_dropout": float(dropout) if head_type in {"mlp", "prototype"} else 0.0,
            "classifier_label_smoothing_enabled": bool(classifier_label_smoothing_enabled),
            "classifier_label_smoothing": float(label_smoothing),
            "classifier_prototype_dim": int(prototype_dim) if head_type == "prototype" else None,
            "classifier_prototype_temperature": (
                float(prototype_temperature) if head_type == "prototype" else None
            ),
            "classifier_supcon_enabled": bool(supcon_enabled),
            "classifier_supcon_weight": float(supcon_weight),
            "classifier_supcon_temperature": float(supcon_temperature) if supcon_enabled else None,
            "n_train": int(len(y_train)),
            "n_test": int(len(y_test)),
            "n_all": int(len(y)),
            "s3prl_embedding_dim": int(emb.shape[1]),
            "selected_band_feature_dim": int(selected_band_dim),
            "fusion_mode": (
                "concat_s3prl_embedding_and_selected_band_features"
                if selected_band_dim > 0
                else "s3prl_embedding_only"
            ),
        },
    }

    history = pd.DataFrame(history_rows)
    checkpoint = {
        "state_dict": best_state_dict,
        "head_type": head_type,
        "input_dim": int(input_dim),
        "s3prl_embedding_dim": int(emb.shape[1]),
        "selected_band_feature_dim": int(selected_band_dim),
        "feature_fusion": {
            "mode": (
                "concat_s3prl_embedding_and_selected_band_features"
                if selected_band_dim > 0
                else "s3prl_embedding_only"
            ),
            "s3prl_embedding_dim": int(emb.shape[1]),
            "selected_band_feature_dim": int(selected_band_dim),
            "input_order": ["s3prl_embedding", "selected_band_features"]
            if selected_band_dim > 0
            else ["s3prl_embedding"],
        },
        "hidden_dim": int(hidden_dim) if head_type in {"mlp", "prototype"} else None,
        "prototype_dim": int(prototype_dim) if head_type == "prototype" else None,
        "num_classes": int(n_classes),
        "activation": "GELU" if head_type in {"mlp", "prototype"} else None,
        "dropout": float(dropout) if head_type in {"mlp", "prototype"} else 0.0,
        "prototype_temperature": float(prototype_temperature) if head_type == "prototype" else None,
        "label_smoothing": float(label_smoothing),
        "supcon_enabled": bool(supcon_enabled),
        "supcon_weight": float(supcon_weight),
        "supcon_temperature": float(supcon_temperature) if supcon_enabled else None,
        "class_names": list(class_names),
        "best_epoch": int(best_epoch),
        "best_epoch_macro_f1": float(best_macro_f1),
        "scaler_mean": scaler.mean_.astype(np.float32),
        "scaler_scale": scaler.scale_.astype(np.float32),
    }

    return TrainEvalResult(
        y_test=y_test_np,
        y_test_pred=pred_test,
        cm_test=cm_test,
        y_all_pred=pred_all,
        y_all_logits=all_logits_np,
        cm_all=cm_all,
        metrics_test=metrics_test,
        metrics_all=metrics_all,
        report_test=report_test,
        report_all=report_all,
        complexity=complexity,
        history=history,
        test_indices=test_idx,
        checkpoint=checkpoint,
    )

def _save_confusion_matrix_plot(
    cm: np.ndarray,
    class_names: Sequence[str],
    output_path: Path,
    title: str,
) -> None:
    fig, ax = plt.subplots(figsize=(7, 6), dpi=140)
    im = ax.imshow(cm, interpolation="nearest", cmap="Blues")
    ax.figure.colorbar(im, ax=ax)

    ax.set(
        xticks=np.arange(len(class_names)),
        yticks=np.arange(len(class_names)),
        xticklabels=class_names,
        yticklabels=class_names,
        ylabel="True Label",
        xlabel="Predicted Label",
        title=title,
    )

    plt.setp(ax.get_xticklabels(), rotation=25, ha="right", rotation_mode="anchor")

    thresh = cm.max() / 2.0 if cm.size else 0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(
                j,
                i,
                format(cm[i, j], "d"),
                ha="center",
                va="center",
                color="white" if cm[i, j] > thresh else "black",
            )

    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)


def _normalize_for_heatmap(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    values = np.nan_to_num(values, nan=0.0, posinf=0.0, neginf=0.0)
    if values.size == 0:
        return values
    low = float(np.min(values))
    high = float(np.max(values))
    if high - low <= 1e-12:
        return np.ones_like(values, dtype=np.float64)
    return (values - low) / (high - low)


def _contiguous_index_runs(indices: np.ndarray) -> list[tuple[int, int]]:
    indices = np.sort(np.unique(np.asarray(indices, dtype=int)))
    if indices.size == 0:
        return []

    runs: list[tuple[int, int]] = []
    start = int(indices[0])
    prev = int(indices[0])
    for value in indices[1:]:
        current = int(value)
        if current == prev + 1:
            prev = current
            continue
        runs.append((start, prev))
        start = current
        prev = current
    runs.append((start, prev))
    return runs


def _plot_band_selection_heatmap(
    X: np.ndarray,
    wave_grid: np.ndarray,
    selected_indices: np.ndarray,
    selection_scores: np.ndarray,
    output_path: Path,
    method: str,
) -> None:
    X = np.asarray(X, dtype=np.float32)
    wave_grid = np.asarray(wave_grid, dtype=float)
    selected_indices = np.asarray(selected_indices, dtype=int)
    selection_scores = np.asarray(selection_scores, dtype=float)

    if X.ndim != 2 or wave_grid.ndim != 1 or X.shape[1] != wave_grid.shape[0]:
        raise BaselineError("Cannot plot band selection heatmap: invalid spectral shape.")
    if selection_scores.size != wave_grid.size:
        selection_scores = np.zeros(wave_grid.size, dtype=float)
        selection_scores[selected_indices] = 1.0

    selected_mask = np.zeros(wave_grid.size, dtype=bool)
    selected_mask[selected_indices] = True

    mean_spectrum = np.mean(X, axis=0)
    p10 = np.percentile(X, 10, axis=0)
    p90 = np.percentile(X, 90, axis=0)
    selected_line = np.where(selected_mask, mean_spectrum, np.nan)
    runs = _contiguous_index_runs(selected_indices)

    with plt.rc_context(
        {
            "font.family": "DejaVu Serif",
            "axes.linewidth": 0.9,
            "axes.edgecolor": "black",
            "axes.labelsize": 10.5,
            "axes.titlesize": 11.5,
            "xtick.labelsize": 9.5,
            "ytick.labelsize": 9.5,
            "figure.dpi": 160,
            "savefig.dpi": 300,
            "legend.fontsize": 8.5,
        }
    ):
        fig, ax = plt.subplots(figsize=(9.0, 4.6), constrained_layout=True)

        for run_idx, (start, end) in enumerate(runs):
            left = float(wave_grid[start])
            right = float(wave_grid[end])
            if start > 0:
                left = (float(wave_grid[start - 1]) + left) / 2.0
            if end < wave_grid.size - 1:
                right = (right + float(wave_grid[end + 1])) / 2.0
            ax.axvspan(
                left,
                right,
                color="#9ECAE1",
                alpha=0.22,
                linewidth=0,
                label="Selected range" if run_idx == 0 else None,
                zorder=0,
            )

        ax.fill_between(
            wave_grid,
            p10,
            p90,
            color="#BDBDBD",
            alpha=0.18,
            linewidth=0,
            label="10-90 percentile",
            zorder=1,
        )
        ax.plot(
            wave_grid,
            mean_spectrum,
            color="#1F4E79",
            linewidth=1.55,
            label="Mean spectrum",
            zorder=2,
        )
        ax.plot(
            wave_grid,
            selected_line,
            color="#B2182B",
            linewidth=2.35,
            alpha=0.95,
            solid_capstyle="round",
            zorder=3,
        )

        y_min, y_max = ax.get_ylim()
        rug_y = y_min + 0.028 * (y_max - y_min)
        ax.vlines(
            wave_grid[selected_mask],
            y_min,
            rug_y,
            color="#B2182B",
            linewidth=0.55,
            alpha=0.45,
            zorder=5,
        )
        ax.set_ylim(y_min, y_max)

        ax.set_xlabel("Wavelength / band position")
        ax.set_ylabel("Intensity")
        ax.set_title(
            f"Band selection on mean spectrum ({method}; {int(selected_mask.sum())}/{wave_grid.size})"
        )
        ax.grid(axis="y", linestyle="--", linewidth=0.45, alpha=0.32)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.margins(x=0.01)
        handles, labels = ax.get_legend_handles_labels()
        by_label = dict(zip(labels, handles))
        ax.legend(by_label.values(), by_label.keys(), frameon=False, loc="upper right")
        fig.savefig(output_path)
        plt.close(fig)


def run_baseline(
    config: BaselineConfig,
    output_dir: Path | None = None,
    progress_prefix: str = "",
) -> Path:
    if output_dir is None:
        output_dir = _create_output_dir(Path(config.output_root), config.upstream)
    else:
        output_dir.mkdir(parents=True, exist_ok=True)
    figures_dir = output_dir / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)

    _log_progress(progress_prefix, "Loading and aligning dataset...")
    loaded = load_dataset(config.input_root)
    aligned = align_dataset(loaded)
    train_idx, test_idx = _make_split_indices(
        y=aligned.y,
        test_size=config.test_size,
        random_state=config.random_state,
    )

    _log_progress(progress_prefix, "Applying spectral preprocessing...")
    preprocess_config = load_spectral_preprocess_config(config)
    preprocess_result = apply_configured_spectral_preprocessing(
        X=aligned.X,
        wave_grid=aligned.wave_grid,
        y=aligned.y,
        train_indices=train_idx,
        class_names=aligned.class_names,
        config=preprocess_config,
        device=config.device,
        random_state=config.random_state,
    )
    X_for_embedding = preprocess_result.X
    selected_band_features = preprocess_result.selected_band_features
    pls_metadata = _find_step_metadata(preprocess_result.metadata, "pls")
    band_selection_metadata = _find_step_metadata(preprocess_result.metadata, "band_selection")

    _log_progress(progress_prefix, f"Extracting embeddings with upstream={config.upstream}...")
    emb, upstream_stats = extract_embeddings(
        X=X_for_embedding,
        upstream_name=config.upstream,
        s3prl_repo=Path(config.s3prl_repo),
        device=config.device,
        batch_size=config.batch_size,
    )

    _log_progress(progress_prefix, f"Training classifier head={config.classifier_head_type}...")
    result = train_and_evaluate(
        emb=emb,
        band_features=selected_band_features,
        y=aligned.y,
        class_names=aligned.class_names,
        test_size=config.test_size,
        random_state=config.random_state,
        device=config.device,
        classifier_epochs=config.classifier_epochs,
        classifier_lr=config.classifier_lr,
        classifier_weight_decay=config.classifier_weight_decay,
        classifier_head_type=config.classifier_head_type,
        classifier_hidden_dim=config.classifier_hidden_dim,
        classifier_dropout=config.classifier_dropout,
        classifier_label_smoothing_enabled=config.classifier_label_smoothing_enabled,
        classifier_label_smoothing=config.classifier_label_smoothing,
        classifier_prototype_dim=config.classifier_prototype_dim,
        classifier_prototype_temperature=config.classifier_prototype_temperature,
        classifier_supcon_enabled=config.classifier_supcon_enabled,
        classifier_supcon_weight=config.classifier_supcon_weight,
        classifier_supcon_temperature=config.classifier_supcon_temperature,
        train_indices=train_idx,
        test_indices=test_idx,
    )

    _log_progress(progress_prefix, "Saving metrics, CSV files, and figures...")
    input_pca_scores = _compute_pca_scores(aligned.X)
    upstream_pca_scores = _compute_pca_scores(emb)
    classifier_logit_pca_scores = _compute_pca_scores(result.y_all_logits)
    preprocessed_pca_scores = None
    if preprocess_result.metadata.get("applied_order"):
        preprocessed_pca_scores = _compute_pca_scores(X_for_embedding)

    pd.DataFrame(
        {
            "y_true": result.y_test,
            "y_pred": result.y_test_pred,
            "y_true_name": [aligned.class_names[int(v)] for v in result.y_test],
            "y_pred_name": [aligned.class_names[int(v)] for v in result.y_test_pred],
        }
    ).to_csv(output_dir / "predictions_test.csv", index=False)

    pd.DataFrame(
        {
            "y_true": aligned.y,
            "y_pred": result.y_all_pred,
            "y_true_name": [aligned.class_names[int(v)] for v in aligned.y],
            "y_pred_name": [aligned.class_names[int(v)] for v in result.y_all_pred],
        }
    ).to_csv(output_dir / "predictions_all.csv", index=False)

    pd.DataFrame(result.cm_test, index=aligned.class_names, columns=aligned.class_names).to_csv(
        output_dir / "confusion_matrix_test.csv"
    )

    pd.DataFrame(result.cm_all, index=aligned.class_names, columns=aligned.class_names).to_csv(
        output_dir / "confusion_matrix_all.csv"
    )

    _save_confusion_matrix_plot(
        cm=result.cm_test,
        class_names=aligned.class_names,
        output_path=figures_dir / "confusion_matrix_test.png",
        title="S3PRL Baseline Confusion Matrix (Test Split)",
    )

    _save_confusion_matrix_plot(
        cm=result.cm_all,
        class_names=aligned.class_names,
        output_path=figures_dir / "confusion_matrix_all.png",
        title="S3PRL Baseline Confusion Matrix (All Data)",
    )

    _plot_pca_true_labels(
        pca_scores=input_pca_scores,
        y=aligned.y,
        class_names=aligned.class_names,
        output_path=figures_dir / "pca_input_spectra.png",
        title="PCA of Input Spectra (True Labels)",
    )

    if preprocessed_pca_scores is not None:
        _plot_pca_true_labels(
            pca_scores=preprocessed_pca_scores,
            y=aligned.y,
            class_names=aligned.class_names,
            output_path=figures_dir / "pca_preprocessed_spectra.png",
            title="PCA of Preprocessed Spectra (True Labels)",
        )

    _plot_pca_true_labels(
        pca_scores=upstream_pca_scores,
        y=aligned.y,
        class_names=aligned.class_names,
        output_path=figures_dir / "pca_true_labels.png",
        title="PCA of S3PRL Upstream Embeddings (True Labels)",
    )

    _plot_pca_true_labels(
        pca_scores=classifier_logit_pca_scores,
        y=aligned.y,
        class_names=aligned.class_names,
        output_path=figures_dir / "pca_classifier_logits.png",
        title="PCA of Classifier Head Logits (True Labels)",
    )

    _plot_pca_test_predictions(
        pca_scores=upstream_pca_scores,
        y_true=np.asarray(result.y_test, dtype=int),
        y_pred=np.asarray(result.y_test_pred, dtype=int),
        test_indices=np.asarray(result.test_indices, dtype=int),
        class_names=aligned.class_names,
        output_path=figures_dir / "pca_test_predictions.png",
    )

    result.history.to_csv(output_dir / "training_history.csv", index=False)
    _plot_training_curve(result.history, figures_dir / "training_curve.png")

    torch.save(result.checkpoint, output_dir / "best_classifier_head.pt")
    torch.save(preprocess_result.state, output_dir / "preprocess_state.pt")
    pd.DataFrame({"wave": aligned.wave_grid}).to_csv(output_dir / "wave_grid.csv", index=False)

    result.report_test.to_csv(output_dir / "classification_report_test.csv", index=False)
    result.report_all.to_csv(output_dir / "classification_report_all.csv", index=False)

    if config.outputs.save_embeddings:
        pd.DataFrame(emb).to_csv(output_dir / "embeddings.csv", index=False)
    if config.outputs.save_preprocessed_spectra:
        pd.DataFrame(X_for_embedding).to_csv(output_dir / "preprocessed_spectra.csv", index=False)
    if selected_band_features is not None:
        pd.DataFrame(selected_band_features).to_csv(output_dir / "selected_band_features.csv", index=False)
    if config.outputs.save_sample_index:
        loaded.to_index_frame().to_csv(output_dir / "sample_index.csv", index=False)
    band_selection_state = None
    segment_normalize_state = None
    for step_state in preprocess_result.state.get("steps", []):
        if step_state.get("name") == "band_selection":
            band_selection_state = step_state
        elif step_state.get("name") == "segment_normalize":
            segment_normalize_state = step_state

    band_summary = {
        "enabled": bool(band_selection_metadata.get("enabled", False)),
        "method": str(band_selection_metadata.get("method", "not_applied")),
        "original_band_count": int(
            band_selection_metadata.get("original_band_count", aligned.X.shape[1])
        ),
        "selected_band_count": int(
            band_selection_metadata.get("selected_band_count", X_for_embedding.shape[1])
        ),
        "removed_band_count": int(
            band_selection_metadata.get(
                "removed_band_count",
                aligned.X.shape[1] - X_for_embedding.shape[1],
            )
        ),
        "selected_ratio": float(
            band_selection_metadata.get(
                "selected_ratio",
                X_for_embedding.shape[1] / max(1, aligned.X.shape[1]),
            )
        ),
        "fit_scope": band_selection_metadata.get("fit_scope", "not_applied"),
    }
    pd.DataFrame([band_summary]).to_csv(output_dir / "band_selection_summary.csv", index=False)
    with open(output_dir / "band_selection_summary.json", "w", encoding="utf-8") as f:
        json.dump(band_summary, f, indent=2, ensure_ascii=False)

    if band_selection_state is not None and band_selection_state.get("method") != "none":
        selected_indices = np.asarray(band_selection_state["selected_indices"], dtype=int)
        selected_waves = np.asarray(band_selection_state["selected_waves"], dtype=float)
        original_selected_indices = selected_indices
        candidate_original_indices = None
        scores = np.asarray(band_selection_state.get("selection_scores", []), dtype=float)
        input_waves = np.asarray(
            band_selection_state.get("input_waves", aligned.wave_grid[: scores.size]),
            dtype=float,
        )

        if segment_normalize_state is not None:
            segment_indices = np.asarray(
                segment_normalize_state.get("selected_indices", []),
                dtype=int,
            )
            selected_in_segment = (
                selected_indices.size == 0
                or (np.min(selected_indices) >= 0 and np.max(selected_indices) < segment_indices.size)
            )
            if (
                segment_indices.size >= max(input_waves.size, scores.size)
                and selected_in_segment
            ):
                candidate_original_indices = segment_indices[: max(input_waves.size, scores.size)]
                original_selected_indices = segment_indices[selected_indices]

        pd.DataFrame(
            {
                "selected_rank": np.arange(1, selected_indices.size + 1),
                "band_index": selected_indices,
                "original_band_index": original_selected_indices,
                "wave": selected_waves,
            }
        ).to_csv(output_dir / "selected_bands.csv", index=False)
        if scores.size:
            score_waves = input_waves if input_waves.size == scores.size else aligned.wave_grid[: scores.size]
            score_original_indices = (
                candidate_original_indices[: scores.size]
                if candidate_original_indices is not None and candidate_original_indices.size >= scores.size
                else np.arange(scores.size, dtype=int)
            )
            pd.DataFrame(
                {
                    "band_index": np.arange(scores.size, dtype=int),
                    "original_band_index": score_original_indices,
                    "wave": score_waves,
                    "score": scores,
                    "selected": np.isin(np.arange(scores.size), selected_indices),
                }
            ).to_csv(output_dir / "band_selection_scores.csv", index=False)

        plot_scores = scores
        plot_selected_indices = selected_indices
        if input_waves.shape[0] != aligned.X.shape[1] and candidate_original_indices is not None:
            plot_scores = np.zeros(aligned.X.shape[1], dtype=float)
            if scores.size:
                plot_scores[candidate_original_indices[: scores.size]] = scores
            plot_selected_indices = original_selected_indices

        _plot_band_selection_heatmap(
            X=aligned.X,
            wave_grid=aligned.wave_grid,
            selected_indices=plot_selected_indices,
            selection_scores=plot_scores,
            output_path=figures_dir / "band_selection_heatmap.png",
            method=str(band_selection_state.get("method", "band_selection")),
        )

    complexity = {
        "preprocess_pipeline": preprocess_result.metadata,
        "pls_calibration": pls_metadata,
        "band_selection": band_selection_metadata,
        "upstream": upstream_stats,
        **result.complexity,
        "total_params": {
            "upstream_total_params": int(upstream_stats["total_params"]),
            "classifier_head_total_params": int(result.complexity["classifier_head"]["total_params"]),
            "combined_total_params": int(
                upstream_stats["total_params"] + result.complexity["classifier_head"]["total_params"]
            ),
            "combined_trainable_params": int(result.complexity["classifier_head"]["trainable_params"]),
        },
    }

    metrics_combined = {
        "test": result.metrics_test,
        "all_data": result.metrics_all,
    }

    artifact_files = sorted(
        str(path.relative_to(output_dir)).replace("\\", "/")
        for path in output_dir.rglob("*")
        if path.is_file()
    )
    artifact_files.append("run_summary.json")

    run_summary = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "config": asdict(config),
        "dataset": {
            "input_root": config.input_root,
            "n_samples": int(aligned.X.shape[0]),
            "n_classes": int(len(aligned.class_names)),
            "class_names": list(aligned.class_names),
            "aligned_shape": [int(aligned.X.shape[0]), int(aligned.X.shape[1])],
            "preprocessed_shape": [int(X_for_embedding.shape[0]), int(X_for_embedding.shape[1])],
            "s3prl_input_shape": [int(X_for_embedding.shape[0]), int(X_for_embedding.shape[1])],
            "selected_band_feature_shape": (
                [int(selected_band_features.shape[0]), int(selected_band_features.shape[1])]
                if selected_band_features is not None
                else None
            ),
            "n_train": int(len(train_idx)),
            "n_test": int(len(test_idx)),
        },
        "metrics": metrics_combined,
        "complexity": complexity,
        "preprocess_pipeline": preprocess_result.metadata,
        "pls_calibration": pls_metadata,
        "band_selection": band_selection_metadata,
        "band_selection_summary": band_summary,
        "artifacts": artifact_files,
    }

    with open(output_dir / "run_summary.json", "w", encoding="utf-8") as f:
        json.dump(run_summary, f, indent=2, ensure_ascii=False)

    _log_progress(progress_prefix, f"Finished. Test f1_macro={result.metrics_test['f1_macro']:.4f}")
    return output_dir


def _section(data: dict[str, Any], name: str) -> dict[str, Any]:
    value = data.get(name, {})
    if not isinstance(value, dict):
        raise BaselineError(f"Config section [{name}] must be a table.")
    return value


def _resolve_config_path(value: str, base_dir: Path) -> str:
    path = Path(value)
    if not path.is_absolute():
        path = base_dir / path
    return str(path.resolve())


def _resolve_device(value: str) -> str:
    value = value.strip().lower()
    if value == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    if value not in {"cpu", "cuda"}:
        raise BaselineError("s3prl.device must be 'auto', 'cpu', or 'cuda'.")
    if value == "cuda" and not torch.cuda.is_available():
        raise BaselineError("s3prl.device is 'cuda', but CUDA is not available.")
    return value


def _parse_upstreams(value: Any) -> list[str]:
    if isinstance(value, str):
        upstreams = [value.strip()]
    elif isinstance(value, list):
        upstreams = [str(item).strip() for item in value]
    else:
        raise BaselineError("s3prl.upstream must be a string or a list of strings.")

    upstreams = [item for item in upstreams if item]
    if not upstreams:
        raise BaselineError("Missing required config value: s3prl.upstream")
    if len(upstreams) != len(set(upstreams)):
        raise BaselineError("s3prl.upstream contains duplicate model names.")
    return upstreams


def _parse_range_list(value: Any, field_name: str) -> list[tuple[float, float]] | None:
    if value is None:
        return None
    if not isinstance(value, list):
        raise BaselineError(f"{field_name} must be a list of [low, high] pairs.")

    ranges: list[tuple[float, float]] = []
    for item in value:
        if not isinstance(item, list) or len(item) != 2:
            raise BaselineError(f"{field_name} must contain only [low, high] pairs.")
        low = float(item[0])
        high = float(item[1])
        if low >= high:
            raise BaselineError(f"{field_name} contains invalid range [{low}, {high}].")
        ranges.append((low, high))
    return ranges


def load_config(config_path: str | Path) -> BaselineConfig:
    path = Path(config_path).resolve()
    if not path.exists():
        raise BaselineError(f"Config file not found: {path}")

    with open(path, "rb") as f:
        data = tomllib.load(f)

    base_dir = path.parent
    paths = _section(data, "paths")
    s3prl = _section(data, "s3prl")
    split = _section(data, "split")
    classifier = _section(data, "classifier")
    preprocess = _section(data, "preprocess")
    outputs = _section(data, "outputs")

    upstreams = _parse_upstreams(s3prl.get("upstream", ""))
    upstream = upstreams[0]

    order = _parse_preprocess_order(
        preprocess.get(
            "order",
            ["snv", "wavelet", "pls"],
        )
    )

    snv = _section(data, "snv")
    wavelet = _section(data, "wavelet")
    pls = _section(data, "pls")
    segment_normalize = _section(data, "segment_normalize")
    band_selection = _section(data, "band_selection")

    preprocess_config = SpectralPreprocessConfig(
        enabled=bool(preprocess.get("enabled", True)),
        order=order,
        snv=SNVStepConfig(
            enabled=bool(snv.get("enabled", True)),
            eps=float(snv.get("eps", 1e-12)),
        ),
        wavelet=WaveletDriftStepConfig(
            enabled=bool(wavelet.get("enabled", True)),
            wavelet=str(wavelet.get("wavelet", "db6")),
            level=int(wavelet.get("level", 4)),
            mode=str(wavelet.get("mode", "symmetric")),
            approximation_scale=float(wavelet.get("approximation_scale", 0.0)),
        ),
        pls=PLSStepConfig(
            enabled=bool(pls.get("enabled", True)),
            components=int(pls.get("components", 0)),
        ),
        segment_normalize=SegmentNormalizeStepConfig(
            enabled=bool(segment_normalize.get("enabled", False)),
            ranges=_parse_range_list(
                segment_normalize.get("ranges"),
                "segment_normalize.ranges",
            ),
            method=str(segment_normalize.get("method", "zscore")).strip().lower(),
            eps=float(segment_normalize.get("eps", 1e-12)),
        ),
        band_selection=BandSelectionStepConfig(
            enabled=bool(band_selection.get("enabled", False)),
            method=str(band_selection.get("method", "none")).strip().lower(),
            fusion_mode=str(band_selection.get("fusion_mode", "dual")).strip().lower(),
            manual_ranges=_parse_range_list(
                band_selection.get("manual_ranges"),
                "band_selection.manual_ranges",
            ),
            top_k=int(band_selection.get("top_k", 0)),
            top_ratio=float(band_selection.get("top_ratio", 0.25)),
            min_bands=int(band_selection.get("min_bands", 16)),
            pls_components=int(band_selection.get("pls_components", 0)),
            lasso_alpha=float(band_selection.get("lasso_alpha", 0.05)),
            cars_iterations=int(band_selection.get("cars_iterations", 40)),
            cars_sample_ratio=float(band_selection.get("cars_sample_ratio", 0.8)),
            ga_population=int(band_selection.get("ga_population", 24)),
            ga_generations=int(band_selection.get("ga_generations", 30)),
            ga_crossover_rate=float(band_selection.get("ga_crossover_rate", 0.8)),
            ga_mutation_rate=float(band_selection.get("ga_mutation_rate", 0.08)),
            ga_elite_count=int(band_selection.get("ga_elite_count", 2)),
            iwoa_population=int(band_selection.get("iwoa_population", 24)),
            iwoa_iterations=int(band_selection.get("iwoa_iterations", 30)),
            iwoa_b=float(band_selection.get("iwoa_b", 1.0)),
            iwoa_mutation_rate=float(band_selection.get("iwoa_mutation_rate", 0.05)),
            epochs=int(band_selection.get("epochs", 200)),
            lr=float(band_selection.get("lr", 0.01)),
            weight_decay=float(band_selection.get("weight_decay", 0.0001)),
            hidden_dim=int(band_selection.get("hidden_dim", 64)),
            temperature=float(band_selection.get("temperature", 1.0)),
            sparsity_lambda=float(band_selection.get("sparsity_lambda", 0.001)),
        ),
    )

    return BaselineConfig(
        config_path=str(path),
        input_root=_resolve_config_path(str(paths.get("input_root", "data")), base_dir),
        output_root=_resolve_config_path(str(paths.get("output_root", "output_s3prl")), base_dir),
        s3prl_repo=_resolve_config_path(str(paths.get("s3prl_repo", "s3prl-main")), base_dir),
        upstream=upstream,
        upstreams=upstreams,
        test_size=float(split.get("test_size", 0.2)),
        random_state=int(split.get("random_state", 42)),
        batch_size=int(s3prl.get("batch_size", 8)),
        device=_resolve_device(str(s3prl.get("device", "auto"))),
        classifier_epochs=int(classifier.get("epochs", 30)),
        classifier_lr=float(classifier.get("lr", 1e-3)),
        classifier_weight_decay=float(classifier.get("weight_decay", 1e-4)),
        classifier_head_type=str(classifier.get("head_type", "mlp")).strip().lower(),
        classifier_hidden_dim=int(classifier.get("hidden_dim", 256)),
        classifier_dropout=float(classifier.get("dropout", 0.3)),
        classifier_label_smoothing_enabled=bool(classifier.get("label_smoothing_enabled", False)),
        classifier_label_smoothing=float(classifier.get("label_smoothing", 0.0)),
        classifier_prototype_dim=int(classifier.get("prototype_dim", 128)),
        classifier_prototype_temperature=float(classifier.get("prototype_temperature", 0.1)),
        classifier_supcon_enabled=bool(classifier.get("supcon_enabled", False)),
        classifier_supcon_weight=float(classifier.get("supcon_weight", 0.1)),
        classifier_supcon_temperature=float(classifier.get("supcon_temperature", 0.1)),
        preprocess=preprocess_config,
        outputs=OutputConfig(
            save_embeddings=bool(outputs.get("save_embeddings", True)),
            save_preprocessed_spectra=bool(outputs.get("save_preprocessed_spectra", True)),
            save_sample_index=bool(outputs.get("save_sample_index", True)),
        ),
    )


def _extract_multi_model_row(
    upstream: str,
    model_dir: Path,
    summary: dict[str, Any],
) -> dict[str, Any]:
    test_metrics = summary["metrics"]["test"]
    all_metrics = summary["metrics"]["all_data"]
    complexity = summary.get("complexity", {})
    upstream_complexity = complexity.get("upstream", {})
    head_complexity = complexity.get("classifier_head", {})
    total_params = complexity.get("total_params", {})
    training = complexity.get("training", {})
    dataset = summary.get("dataset", {})

    test_f1 = float(test_metrics["f1_macro"])
    all_data_f1 = float(all_metrics["f1_macro"])
    combined_params = int(total_params.get("combined_total_params", 0) or 0)
    upstream_flops = upstream_complexity.get("upstream_forward_flops_per_sample")
    n_test = int(test_metrics.get("n_samples", dataset.get("n_test", 0)) or 0)
    test_accuracy = float(test_metrics["accuracy"])

    return {
        "status": "ok",
        "upstream": upstream,
        "test_f1_macro": test_f1,
        "test_accuracy": test_accuracy,
        "test_balanced_accuracy": float(test_metrics["balanced_accuracy"]),
        "test_precision_macro": float(test_metrics["precision_macro"]),
        "test_recall_macro": float(test_metrics["recall_macro"]),
        "test_mcc": float(test_metrics["mcc"]),
        "test_error_count": int(round((1.0 - test_accuracy) * n_test)),
        "all_data_f1_macro": all_data_f1,
        "all_data_accuracy": float(all_metrics["accuracy"]),
        "all_data_test_f1_gap": float(all_data_f1 - test_f1),
        "best_epoch": int(training.get("best_epoch", 0) or 0),
        "best_epoch_macro_f1": float(training.get("best_epoch_macro_f1", np.nan)),
        "embedding_dim": int(upstream_complexity.get("embedding_dim", 0) or 0),
        "upstream_params": int(total_params.get("upstream_total_params", 0) or 0),
        "classifier_head_params": int(total_params.get("classifier_head_total_params", 0) or 0),
        "combined_params": combined_params,
        "combined_params_m": float(combined_params / 1_000_000.0) if combined_params else np.nan,
        "upstream_gflops_per_sample": (
            float(upstream_flops) / 1_000_000_000.0 if upstream_flops is not None else np.nan
        ),
        "classifier_kflops_per_sample": float(
            (head_complexity.get("forward_flops_per_sample", 0) or 0) / 1_000.0
        ),
        "f1_per_million_params": (
            float(test_f1 / (combined_params / 1_000_000.0)) if combined_params else np.nan
        ),
        "output_dir": str(model_dir),
        "error": "",
    }


def _plot_multi_upstream_metrics(rows: list[dict[str, Any]], output_path: Path) -> None:
    if not rows:
        return

    plot_rows = sorted(rows, key=lambda row: float(row["test_f1_macro"]), reverse=True)
    labels = [row["upstream"] for row in plot_rows]
    metric_specs = [
        ("test_f1_macro", "Test Macro F1"),
        ("test_accuracy", "Test Accuracy"),
        ("test_balanced_accuracy", "Test Balanced Acc."),
    ]

    x = np.arange(len(labels))
    width = 0.24
    colors = ["#0072B2", "#009E73", "#D55E00"]

    with plt.rc_context(
        {
            "font.family": "DejaVu Serif",
            "axes.linewidth": 0.9,
            "axes.edgecolor": "black",
            "axes.labelsize": 10,
            "axes.titlesize": 11,
            "xtick.labelsize": 8,
            "ytick.labelsize": 9,
            "legend.fontsize": 8,
            "figure.dpi": 160,
            "savefig.dpi": 300,
        }
    ):
        fig, ax = plt.subplots(figsize=(9.2, 4.8))
        for idx, (key, label) in enumerate(metric_specs):
            values = [float(row[key]) for row in plot_rows]
            bars = ax.bar(
                x + (idx - 1) * width,
                values,
                width=width,
                label=label,
                color=colors[idx],
                edgecolor="black",
                linewidth=0.6,
            )
            for bar, value in zip(bars, values):
                ax.text(
                    bar.get_x() + bar.get_width() / 2,
                    value + 0.008,
                    f"{value:.3f}",
                    ha="center",
                    va="bottom",
                    fontsize=7,
                    rotation=90,
                )

        ax.set_ylabel("Score")
        ax.set_title("Upstream Model Classification Performance")
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=25, ha="right")
        ax.set_ylim(0.0, min(1.08, max(1.0, max(float(row["test_f1_macro"]) for row in plot_rows) + 0.12)))
        ax.grid(axis="y", linestyle="--", linewidth=0.5, alpha=0.45)
        ax.legend(frameon=False, ncols=3, loc="upper center", bbox_to_anchor=(0.5, 1.14))
        fig.tight_layout()
        fig.savefig(output_path)
        plt.close(fig)


def _plot_multi_upstream_complexity(rows: list[dict[str, Any]], output_path: Path) -> None:
    if not rows:
        return

    plot_rows = sorted(rows, key=lambda row: float(row["combined_params_m"]))
    labels = [row["upstream"] for row in plot_rows]
    params_m = [float(row["combined_params_m"]) for row in plot_rows]
    gflops = [float(row["upstream_gflops_per_sample"]) for row in plot_rows]
    f1_scores = [float(row["test_f1_macro"]) for row in plot_rows]

    x = np.arange(len(labels))
    with plt.rc_context(
        {
            "font.family": "DejaVu Serif",
            "axes.linewidth": 0.9,
            "axes.edgecolor": "black",
            "axes.labelsize": 10,
            "axes.titlesize": 11,
            "xtick.labelsize": 8,
            "ytick.labelsize": 9,
            "legend.fontsize": 8,
            "figure.dpi": 160,
            "savefig.dpi": 300,
        }
    ):
        fig, axes = plt.subplots(1, 2, figsize=(11, 4.5), sharex=True)

        axes[0].bar(
            x,
            params_m,
            color="#0072B2",
            edgecolor="black",
            linewidth=0.6,
        )
        axes[0].set_ylabel("Parameters (M)")
        axes[0].set_title("Model Size")
        axes[0].grid(axis="y", linestyle="--", linewidth=0.5, alpha=0.45)

        axes[1].bar(
            x,
            gflops,
            color="#D55E00",
            edgecolor="black",
            linewidth=0.6,
        )
        axes[1].set_ylabel("Upstream GFLOPs / sample")
        axes[1].set_title("Forward Compute")
        axes[1].grid(axis="y", linestyle="--", linewidth=0.5, alpha=0.45)

        for ax in axes:
            ax.set_xticks(x)
            ax.set_xticklabels(labels, rotation=30, ha="right")

        ax_f1 = axes[1].twinx()
        ax_f1.plot(
            x,
            f1_scores,
            color="black",
            marker="o",
            linewidth=1.3,
            markersize=4,
            label="Test Macro F1",
        )
        ax_f1.set_ylabel("Test Macro F1")
        ax_f1.set_ylim(0.0, 1.0)
        ax_f1.legend(frameon=False, loc="upper right")

        fig.suptitle("Upstream Model Size, Compute, and Accuracy Trade-off", y=1.02, fontsize=12)
        fig.tight_layout()
        fig.savefig(output_path)
        plt.close(fig)


def run_multi_upstream(config: BaselineConfig) -> Path:
    parent_dir = _create_multi_output_dir(Path(config.output_root))
    figures_dir = parent_dir / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)
    total = len(config.upstreams)
    _log_progress("", f"Multi-upstream run started: {total} models")
    _log_progress("", f"Output folder: {parent_dir}")

    rows: list[dict[str, Any]] = []
    summaries: list[dict[str, Any]] = []

    for idx, upstream in enumerate(config.upstreams, start=1):
        prefix = f"[{idx}/{total} {upstream}]"
        model_dir = parent_dir / _safe_name(upstream)
        model_config = replace(config, upstream=upstream)

        _log_progress(prefix, "Started")
        try:
            run_baseline(
                model_config,
                output_dir=model_dir,
                progress_prefix=prefix,
            )
            summary_path = model_dir / "run_summary.json"
            with open(summary_path, "r", encoding="utf-8") as f:
                summary = json.load(f)

            row = _extract_multi_model_row(
                upstream=upstream,
                model_dir=model_dir,
                summary=summary,
            )
            rows.append(row)
            summaries.append(summary)
            _log_progress(
                prefix,
                f"Completed: test_f1_macro={row['test_f1_macro']:.4f}, "
                f"accuracy={row['test_accuracy']:.4f}",
            )
        except Exception as exc:
            rows.append(
                {
                    "status": "failed",
                    "upstream": upstream,
                    "test_f1_macro": np.nan,
                    "test_accuracy": np.nan,
                    "test_balanced_accuracy": np.nan,
                    "test_precision_macro": np.nan,
                    "test_recall_macro": np.nan,
                    "test_mcc": np.nan,
                    "test_error_count": np.nan,
                    "all_data_f1_macro": np.nan,
                    "all_data_accuracy": np.nan,
                    "all_data_test_f1_gap": np.nan,
                    "best_epoch": np.nan,
                    "best_epoch_macro_f1": np.nan,
                    "embedding_dim": np.nan,
                    "upstream_params": np.nan,
                    "classifier_head_params": np.nan,
                    "combined_params": np.nan,
                    "combined_params_m": np.nan,
                    "upstream_gflops_per_sample": np.nan,
                    "classifier_kflops_per_sample": np.nan,
                    "f1_per_million_params": np.nan,
                    "output_dir": str(model_dir),
                    "error": str(exc),
                }
            )
            _log_progress(prefix, f"Failed: {exc}")

        pd.DataFrame(rows).to_csv(parent_dir / "model_comparison.csv", index=False)

    ok_rows = [row for row in rows if row["status"] == "ok"]
    if not ok_rows:
        with open(parent_dir / "best_model.json", "w", encoding="utf-8") as f:
            json.dump(
                {
                    "status": "failed",
                    "message": "No upstream model completed successfully.",
                    "models": rows,
                },
                f,
                indent=2,
                ensure_ascii=False,
            )
        raise BaselineError(f"No upstream model completed successfully. See {parent_dir}")

    _log_progress("", "Drawing multi-model comparison figures...")
    _plot_multi_upstream_metrics(ok_rows, figures_dir / "model_metrics_bar.png")
    _plot_multi_upstream_complexity(ok_rows, figures_dir / "model_complexity_bar.png")
    pd.DataFrame(rows).to_csv(parent_dir / "model_comparison.csv", index=False)

    best_row = max(
        ok_rows,
        key=lambda row: (
            float(row["test_f1_macro"]),
            float(row["test_accuracy"]),
            float(row["test_balanced_accuracy"]),
        ),
    )
    best_summary = next(
        summary
        for summary in summaries
        if summary["config"]["upstream"] == best_row["upstream"]
    )
    best_payload = {
        "status": "ok",
        "selection_metric": "test_f1_macro",
        "tie_breakers": ["test_accuracy", "test_balanced_accuracy"],
        "best_model": best_row,
        "best_result": best_summary,
        "all_models": rows,
    }

    with open(parent_dir / "best_model.json", "w", encoding="utf-8") as f:
        json.dump(best_payload, f, indent=2, ensure_ascii=False)

    with open(parent_dir / "run_summary.json", "w", encoding="utf-8") as f:
        json.dump(
            {
                "created_at": datetime.now().isoformat(timespec="seconds"),
                "config": asdict(config),
                "mode": "multi_upstream",
                "selection_metric": "test_f1_macro",
                "best_upstream": best_row["upstream"],
                "best_model": best_row,
                "models": rows,
                "artifacts": [
                    "model_comparison.csv",
                    "best_model.json",
                    "run_summary.json",
                    "figures/model_metrics_bar.png",
                    "figures/model_complexity_bar.png",
                ],
            },
            f,
            indent=2,
            ensure_ascii=False,
        )

    _log_progress(
        "",
        f"Best upstream: {best_row['upstream']} "
        f"(test_f1_macro={best_row['test_f1_macro']:.4f}, "
        f"accuracy={best_row['test_accuracy']:.4f})",
    )
    return parent_dir


def parse_args() -> BaselineConfig:
    parser = argparse.ArgumentParser(
        description="S3PRL baseline classification for spectral CSV dataset"
    )
    parser.add_argument(
        "--config",
        default="config.toml",
        help="TOML config file. All run parameters are read from this file.",
    )
    args = parser.parse_args()
    return load_config(args.config)


def main() -> int:
    cfg = parse_args()
    if len(cfg.upstreams) > 1:
        out_dir = run_multi_upstream(cfg)
    else:
        prefix = f"[1/1 {cfg.upstream}]"
        _log_progress(prefix, "Single-upstream run started")
        out_dir = run_baseline(cfg, progress_prefix=prefix)
    print(f"S3PRL baseline finished. Output: {out_dir}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
