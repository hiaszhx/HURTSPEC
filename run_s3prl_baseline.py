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
class SpectralPreprocessConfig:
    enabled: bool
    order: list[str]
    snv: SNVStepConfig
    wavelet: WaveletDriftStepConfig
    pls: PLSStepConfig


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


def _estimate_classifier_head_forward_flops_per_sample(
    head_type: str,
    input_dim: int,
    hidden_dim: int,
    num_classes: int,
) -> int:
    if head_type == "linear":
        return int(2 * input_dim * num_classes + num_classes)
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
            + ". Allowed steps are: snv, wavelet, pls."
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
) -> SpectralPreprocessResult:
    X_work = np.asarray(X, dtype=np.float32)
    wave_work = np.asarray(wave_grid, dtype=float)
    metadata = {
        "enabled": bool(config.enabled),
        "requested_order": list(config.order),
        "applied_order": [],
        "steps": [],
        "input_shape": [int(X_work.shape[0]), int(X_work.shape[1])],
    }
    state = {
        "enabled": bool(config.enabled),
        "requested_order": list(config.order),
        "applied_order": [],
        "steps": [],
    }

    if not config.enabled:
        metadata["output_shape"] = [int(X_work.shape[0]), int(X_work.shape[1])]
        state["output_shape"] = [int(X_work.shape[0]), int(X_work.shape[1])]
        return SpectralPreprocessResult(X=X_work, metadata=metadata, state=state)

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
        else:
            raise BaselineError(f"Unknown preprocess step: {step}")

    metadata["output_shape"] = [int(X_work.shape[0]), int(X_work.shape[1])]
    metadata["wave_min"] = float(np.min(wave_work))
    metadata["wave_max"] = float(np.max(wave_work))
    state["output_shape"] = [int(X_work.shape[0]), int(X_work.shape[1])]
    return SpectralPreprocessResult(X=X_work, metadata=metadata, state=state)


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

    X_train = emb[train_idx]
    X_test = emb[test_idx]

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train).astype(np.float32)
    X_test_scaled = scaler.transform(X_test).astype(np.float32)

    X_train_tensor = torch.from_numpy(X_train_scaled).to(torch_device)
    X_test_tensor = torch.from_numpy(X_test_scaled).to(torch_device)
    y_train_tensor = torch.from_numpy(np.asarray(y_train, dtype=np.int64)).to(torch_device)

    n_classes = len(class_names)
    input_dim = int(X_train_scaled.shape[1])
    head_type = str(classifier_head_type).strip().lower()
    if head_type not in {"mlp", "linear"}:
        raise BaselineError("classifier.head_type must be 'mlp' or 'linear'.")

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

    class_counts = np.bincount(y_train, minlength=n_classes).astype(np.float32)
    class_counts = np.where(class_counts <= 0, 1.0, class_counts)
    class_weights = class_counts.sum() / (class_counts * float(n_classes))
    class_weights_tensor = torch.from_numpy(class_weights).to(torch_device)

    if head_type == "linear":
        head = LinearHead(input_dim=input_dim, num_classes=n_classes).to(torch_device)
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
        logits = head(X_train_tensor)
        loss = criterion(logits, y_train_tensor)
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

        X_all_scaled = scaler.transform(emb).astype(np.float32)
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
    )
    complexity = {
        "classifier_head": {
            **head_stats,
            "type": head_type,
            "input_dim": input_dim,
            "hidden_dim": int(hidden_dim) if head_type == "mlp" else None,
            "num_classes": int(n_classes),
            "activation": "GELU" if head_type == "mlp" else None,
            "dropout": float(dropout) if head_type == "mlp" else 0.0,
            "forward_flops_per_sample": head_forward_flops_per_sample,
            "train_flops_approx": int(epochs * len(y_train) * head_forward_flops_per_sample * 3),
            "test_forward_flops_approx": int(len(y_test) * head_forward_flops_per_sample),
            "all_data_forward_flops_approx": int(len(y) * head_forward_flops_per_sample),
            "flops_note": "MLP forward FLOPs are approximate; GELU uses a rough estimate and dropout is ignored",
        },
        "training": {
            "classifier_epochs": int(epochs),
            "best_epoch": int(best_epoch),
            "best_epoch_macro_f1": float(best_macro_f1),
            "classifier_lr": float(classifier_lr),
            "classifier_weight_decay": float(classifier_weight_decay),
            "classifier_head_type": head_type,
            "classifier_hidden_dim": int(hidden_dim) if head_type == "mlp" else None,
            "classifier_dropout": float(dropout) if head_type == "mlp" else 0.0,
            "classifier_label_smoothing_enabled": bool(classifier_label_smoothing_enabled),
            "classifier_label_smoothing": float(label_smoothing),
            "n_train": int(len(y_train)),
            "n_test": int(len(y_test)),
            "n_all": int(len(y)),
        },
    }

    history = pd.DataFrame(history_rows)
    checkpoint = {
        "state_dict": best_state_dict,
        "head_type": head_type,
        "input_dim": int(input_dim),
        "hidden_dim": int(hidden_dim) if head_type == "mlp" else None,
        "num_classes": int(n_classes),
        "activation": "GELU" if head_type == "mlp" else None,
        "dropout": float(dropout) if head_type == "mlp" else 0.0,
        "label_smoothing": float(label_smoothing),
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
    )
    X_for_embedding = preprocess_result.X
    pls_metadata = _find_step_metadata(preprocess_result.metadata, "pls")

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
    if config.outputs.save_sample_index:
        loaded.to_index_frame().to_csv(output_dir / "sample_index.csv", index=False)

    complexity = {
        "preprocess_pipeline": preprocess_result.metadata,
        "pls_calibration": pls_metadata,
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
            "n_train": int(len(train_idx)),
            "n_test": int(len(test_idx)),
        },
        "metrics": metrics_combined,
        "complexity": complexity,
        "preprocess_pipeline": preprocess_result.metadata,
        "pls_calibration": pls_metadata,
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
