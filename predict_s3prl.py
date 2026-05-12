from __future__ import annotations

import argparse
from datetime import datetime
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt

from run_s3prl_baseline import (
    BaselineError,
    LinearHead,
    PrototypeHead,
    SmallMLPHead,
    WaveletDriftStepConfig,
    _compute_classification_metrics,
    _save_confusion_matrix_plot,
    _safe_name,
    apply_wavelet_drift_removal,
    extract_embeddings,
)
from src.io.dataset_loader import LoadedDataset, load_dataset
from src.preprocess.alignment import align_dataset
from src.preprocess.filters import apply_snv


def _torch_load(path: Path, map_location: torch.device) -> Any:
    try:
        return torch.load(path, map_location=map_location, weights_only=False)
    except TypeError:
        return torch.load(path, map_location=map_location)


def _resolve_device(device_text: str) -> str:
    value = device_text.strip().lower()
    if value == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    if value not in {"cpu", "cuda"}:
        raise BaselineError("--device must be auto, cpu, or cuda.")
    if value == "cuda" and not torch.cuda.is_available():
        raise BaselineError("CUDA was requested but is not available.")
    return value


def _collect_model_dirs(path: Path) -> list[Path]:
    path = path.resolve()
    if (path / "best_classifier_head.pt").exists():
        return [path]

    best_path = path / "best_model.json"
    if not best_path.exists():
        raise BaselineError(
            f"Cannot find best_classifier_head.pt or best_model.json under: {path}"
        )

    ordered_names: list[str] = []
    with open(best_path, "r", encoding="utf-8") as f:
        payload = json.load(f)
    for row in payload.get("all_models", []):
        upstream = row.get("upstream")
        if upstream:
            ordered_names.append(_safe_name(str(upstream)))

    candidates = []
    seen = set()
    for name in ordered_names:
        candidate = path / name
        if (candidate / "best_classifier_head.pt").exists():
            candidates.append(candidate.resolve())
            seen.add(candidate.resolve())

    for candidate in sorted(path.iterdir()):
        resolved = candidate.resolve()
        if (
            candidate.is_dir()
            and resolved not in seen
            and (candidate / "best_classifier_head.pt").exists()
        ):
            candidates.append(resolved)

    if candidates:
        return candidates

    raise BaselineError(f"Could not resolve model subfolders from: {best_path}")


def _default_prediction_root(model_path: Path) -> Path:
    resolved = model_path.resolve()
    for candidate in [resolved, *resolved.parents]:
        if candidate.name == "output_s3prl":
            return candidate.parent / "output_s3prl_predict"
    return resolved.parent / "output_s3prl_predict"


def _create_prediction_run_dir(base_dir: Path, source_name: str, input_name: str) -> Path:
    base_dir.mkdir(parents=True, exist_ok=True)
    run_id = (
        f"predict_{_safe_name(source_name)}_{_safe_name(input_name)}_"
        f"{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    )
    out_dir = base_dir / run_id
    suffix = 1
    while out_dir.exists():
        out_dir = base_dir / f"{run_id}_{suffix:02d}"
        suffix += 1
    out_dir.mkdir(parents=True, exist_ok=False)
    return out_dir


def _align_to_wave_grid(dataset: LoadedDataset, wave_grid: np.ndarray) -> tuple[np.ndarray, list[str], list[str], list[str]]:
    X_list = []
    sample_names = []
    sample_paths = []
    class_names = []

    for sample in dataset.samples:
        X_list.append(np.interp(wave_grid, sample.wave, sample.intensity))
        sample_names.append(sample.file_name)
        sample_paths.append(str(sample.file_path))
        class_names.append(sample.class_name)

    return (
        np.vstack(X_list).astype(float),
        sample_names,
        sample_paths,
        class_names,
    )


def _find_step(state: dict[str, Any], name: str) -> dict[str, Any] | None:
    for step in state.get("steps", []):
        if step.get("name") == name:
            return step
    return None


def _apply_pls_state(X: np.ndarray, step: dict[str, Any]) -> np.ndarray:
    mean = np.asarray(step["scaler_mean"], dtype=np.float32)
    scale = np.asarray(step["scaler_scale"], dtype=np.float32)
    x_rotations = np.asarray(step["x_rotations"], dtype=np.float32)
    x_loadings = np.asarray(step["x_loadings"], dtype=np.float32)

    if X.shape[1] != mean.shape[0]:
        raise BaselineError(
            "PLS preprocessing state feature count does not match input spectra. "
            f"Expected {mean.shape[0]}, got {X.shape[1]}."
        )

    X_scaled = (np.asarray(X, dtype=np.float32) - mean) / scale
    X_scores = X_scaled @ x_rotations
    X_reconstructed_scaled = X_scores @ x_loadings.T
    return (X_reconstructed_scaled * scale + mean).astype(np.float32)


def _apply_band_selection_state(X: np.ndarray, step: dict[str, Any]) -> np.ndarray:
    method = str(step.get("method", "none")).strip().lower()
    if method == "none":
        return np.asarray(X, dtype=np.float32)

    indices = np.asarray(step["selected_indices"], dtype=int)
    expected = int(step.get("original_n_features", X.shape[1]))
    if X.shape[1] != expected:
        raise BaselineError(
            "Band selection state feature count does not match input spectra. "
            f"Expected {expected}, got {X.shape[1]}."
        )
    if indices.size < 1 or np.min(indices) < 0 or np.max(indices) >= X.shape[1]:
        raise BaselineError("Saved band selection indices are invalid for this input.")
    return np.asarray(X[:, indices], dtype=np.float32)


def _apply_saved_preprocessing(
    X: np.ndarray,
    preprocess_state: dict[str, Any],
    summary_config: dict[str, Any],
) -> tuple[np.ndarray, np.ndarray | None]:
    X_work = np.asarray(X, dtype=np.float32)
    selected_band_features: np.ndarray | None = None
    preprocess_cfg = summary_config.get("preprocess", {})

    if not bool(preprocess_state.get("enabled", preprocess_cfg.get("enabled", False))):
        return X_work, None

    for step_name in preprocess_state.get("applied_order", []):
        if step_name == "snv":
            step = _find_step(preprocess_state, "snv") or {}
            X_work = apply_snv(X_work, eps=float(step.get("eps", 1e-12))).astype(np.float32)
        elif step_name == "wavelet":
            step = _find_step(preprocess_state, "wavelet") or {}
            wavelet_cfg = WaveletDriftStepConfig(
                enabled=True,
                wavelet=str(step.get("wavelet", "db6")),
                level=int(step.get("level", 4)),
                mode=str(step.get("mode", "symmetric")),
                approximation_scale=float(step.get("approximation_scale", 0.0)),
            )
            X_work, _ = apply_wavelet_drift_removal(X_work, wavelet_cfg)
        elif step_name == "pls":
            step = _find_step(preprocess_state, "pls")
            if step is None:
                raise BaselineError("PLS was applied during training, but PLS state is missing.")
            X_work = _apply_pls_state(X_work, step)
        elif step_name == "band_selection":
            step = _find_step(preprocess_state, "band_selection")
            if step is None:
                raise BaselineError(
                    "Band selection was applied during training, but its state is missing."
                )
            fusion_mode = str(step.get("fusion_mode", "single")).strip().lower()
            selected = _apply_band_selection_state(X_work, step)
            if fusion_mode == "dual":
                selected_band_features = selected
            else:
                X_work = selected
                selected_band_features = None
        else:
            raise BaselineError(f"Unknown saved preprocessing step: {step_name}")

    return X_work, selected_band_features


def _build_head(checkpoint: dict[str, Any], device: torch.device) -> torch.nn.Module:
    head_type = str(checkpoint.get("head_type", "mlp"))
    input_dim = int(checkpoint["input_dim"])
    num_classes = int(checkpoint["num_classes"])

    if head_type == "linear":
        head = LinearHead(input_dim=input_dim, num_classes=num_classes)
    elif head_type == "prototype":
        head = PrototypeHead(
            input_dim=input_dim,
            hidden_dim=int(checkpoint["hidden_dim"]),
            prototype_dim=int(checkpoint["prototype_dim"]),
            num_classes=num_classes,
            dropout=float(checkpoint.get("dropout", 0.0)),
            temperature=float(checkpoint.get("prototype_temperature", 0.1)),
        )
    elif head_type in {"mlp", "small_mlp_2linear"}:
        head = SmallMLPHead(
            input_dim=input_dim,
            hidden_dim=int(checkpoint["hidden_dim"]),
            num_classes=num_classes,
            dropout=float(checkpoint.get("dropout", 0.0)),
        )
    else:
        raise BaselineError(f"Unsupported classifier head type: {head_type}")

    head.load_state_dict(checkpoint["state_dict"])
    head.to(device)
    head.eval()
    return head


def predict_single_model(
    model_dir: Path,
    input_root: Path,
    output_dir: Path,
    device_text: str,
    batch_size: int | None,
) -> Path:
    model_dir = model_dir.resolve()
    with open(model_dir / "run_summary.json", "r", encoding="utf-8") as f:
        summary = json.load(f)

    cfg = summary["config"]
    device_name = _resolve_device(device_text)
    device = torch.device(device_name)
    upstream = str(cfg["upstream"])
    s3prl_repo = Path(cfg["s3prl_repo"])
    batch = int(batch_size or cfg.get("batch_size", 8))

    output_dir.mkdir(parents=True, exist_ok=True)
    figures_dir = output_dir / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)

    print(f"Model folder: {model_dir}", flush=True)
    print(f"Input data: {input_root}", flush=True)
    print("Loading prediction dataset...", flush=True)
    loaded = load_dataset(input_root)

    wave_grid_path = model_dir / "wave_grid.csv"
    alignment_note = "aligned_to_saved_training_wave_grid"
    if wave_grid_path.exists():
        wave_grid = pd.read_csv(wave_grid_path)["wave"].to_numpy(dtype=float)
        X, sample_names, sample_paths, true_names = _align_to_wave_grid(loaded, wave_grid)
    else:
        aligned = align_dataset(loaded)
        X = aligned.X
        sample_names = aligned.sample_names
        sample_paths = aligned.sample_paths
        true_names = [aligned.class_names[int(v)] for v in aligned.y]
        alignment_note = "fallback_aligned_to_prediction_dataset_grid"
        if bool(cfg.get("preprocess", {}).get("enabled", False)):
            raise BaselineError(
                "This model folder does not contain wave_grid.csv, but preprocessing was enabled. "
                "Please retrain once with the updated training script."
            )

    preprocess_state_path = model_dir / "preprocess_state.pt"
    if preprocess_state_path.exists():
        preprocess_state = _torch_load(preprocess_state_path, map_location=torch.device("cpu"))
    else:
        preprocess_state = {"enabled": False, "applied_order": [], "steps": []}
        if bool(cfg.get("preprocess", {}).get("enabled", False)):
            raise BaselineError(
                "This model folder does not contain preprocess_state.pt, but preprocessing was enabled. "
                "Please retrain once with the updated training script."
            )

    print("Applying saved preprocessing...", flush=True)
    X_proc, selected_band_features = _apply_saved_preprocessing(X, preprocess_state, cfg)

    print(f"Extracting embeddings with upstream={upstream}...", flush=True)
    emb, upstream_stats = extract_embeddings(
        X=X_proc,
        upstream_name=upstream,
        s3prl_repo=s3prl_repo,
        device=device_name,
        batch_size=batch,
    )

    print("Loading classifier head and predicting...", flush=True)
    checkpoint = _torch_load(model_dir / "best_classifier_head.pt", map_location=device)
    head = _build_head(checkpoint, device)
    scaler_mean = np.asarray(checkpoint["scaler_mean"], dtype=np.float32)
    scaler_scale = np.asarray(checkpoint["scaler_scale"], dtype=np.float32)
    fusion = checkpoint.get("feature_fusion", {})
    selected_dim = int(fusion.get("selected_band_feature_dim", 0) or 0)
    if selected_dim > 0:
        if selected_band_features is None:
            raise BaselineError(
                "This checkpoint expects selected band features, but preprocessing did not produce them."
            )
        if selected_band_features.shape[1] != selected_dim:
            raise BaselineError(
                "Selected band feature dimension does not match checkpoint. "
                f"Expected {selected_dim}, got {selected_band_features.shape[1]}."
            )
        model_input = np.concatenate(
            [emb.astype(np.float32), selected_band_features.astype(np.float32)],
            axis=1,
        ).astype(np.float32)
    else:
        model_input = emb.astype(np.float32)

    emb_scaled = ((model_input - scaler_mean) / scaler_scale).astype(np.float32)

    with torch.no_grad():
        logits = head(torch.from_numpy(emb_scaled).to(device))
        probs = torch.softmax(logits, dim=1).cpu().numpy()
        pred = np.argmax(probs, axis=1)

    class_names = list(checkpoint["class_names"])
    class_to_id = {name: idx for idx, name in enumerate(class_names)}
    y_true = np.array([class_to_id.get(name, -1) for name in true_names], dtype=int)
    y_pred_names = [class_names[int(v)] for v in pred]

    rows = {
        "sample_name": sample_names,
        "sample_path": sample_paths,
        "y_true_name": true_names,
        "y_pred": pred,
        "y_pred_name": y_pred_names,
        "confidence": np.max(probs, axis=1),
    }
    for idx, class_name in enumerate(class_names):
        rows[f"prob_{class_name}"] = probs[:, idx]

    pd.DataFrame(rows).to_csv(output_dir / "predictions.csv", index=False)
    loaded.to_index_frame().to_csv(output_dir / "sample_index.csv", index=False)

    metrics_payload: dict[str, Any] = {}
    known_mask = y_true >= 0
    if np.any(known_mask):
        metrics, report = _compute_classification_metrics(
            y_true=y_true[known_mask],
            y_pred=pred[known_mask],
            class_names=class_names,
        )
        report.to_csv(output_dir / "classification_report.csv", index=False)
        cm = pd.crosstab(
            pd.Categorical([class_names[int(v)] for v in y_true[known_mask]], categories=class_names),
            pd.Categorical([class_names[int(v)] for v in pred[known_mask]], categories=class_names),
            rownames=["true"],
            colnames=["pred"],
            dropna=False,
        )
        cm.to_csv(output_dir / "confusion_matrix.csv")
        _save_confusion_matrix_plot(
            cm=cm.to_numpy(dtype=int),
            class_names=class_names,
            output_path=figures_dir / "confusion_matrix.png",
            title="Prediction Confusion Matrix",
        )
        metrics_payload = metrics

    run_summary = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "model_dir": str(model_dir),
        "input_root": str(input_root.resolve()),
        "output_dir": str(output_dir.resolve()),
        "alignment_note": alignment_note,
        "upstream": upstream,
        "device": device_name,
        "batch_size": batch,
        "n_samples": int(len(sample_names)),
        "class_names": class_names,
        "metrics": metrics_payload,
        "upstream_stats": upstream_stats,
        "feature_fusion": {
            "mode": fusion.get("mode", "s3prl_embedding_only"),
            "s3prl_embedding_dim": int(emb.shape[1]),
            "selected_band_feature_dim": int(selected_dim),
            "model_input_dim": int(model_input.shape[1]),
        },
        "artifacts": sorted(
            str(path.relative_to(output_dir)).replace("\\", "/")
            for path in output_dir.rglob("*")
            if path.is_file()
        ),
    }
    with open(output_dir / "run_summary.json", "w", encoding="utf-8") as f:
        json.dump(run_summary, f, indent=2, ensure_ascii=False)

    if metrics_payload:
        with open(output_dir / "metrics.json", "w", encoding="utf-8") as f:
            json.dump(metrics_payload, f, indent=2, ensure_ascii=False)
        print(
            f"Done. Accuracy={metrics_payload['accuracy']:.4f}, "
            f"F1_macro={metrics_payload['f1_macro']:.4f}",
            flush=True,
        )
    else:
        print("Done. Labels were not recognized, so metrics were not computed.", flush=True)
    print(f"Output: {output_dir}", flush=True)
    return output_dir


def _prediction_row(model_dir: Path, output_dir: Path, summary: dict[str, Any]) -> dict[str, Any]:
    metrics = summary.get("metrics", {})
    return {
        "status": "ok",
        "upstream": summary.get("upstream", model_dir.name),
        "n_samples": summary.get("n_samples", 0),
        "accuracy": metrics.get("accuracy", np.nan),
        "balanced_accuracy": metrics.get("balanced_accuracy", np.nan),
        "precision_macro": metrics.get("precision_macro", np.nan),
        "recall_macro": metrics.get("recall_macro", np.nan),
        "f1_macro": metrics.get("f1_macro", np.nan),
        "mcc": metrics.get("mcc", np.nan),
        "model_dir": str(model_dir),
        "output_dir": str(output_dir),
        "error": "",
    }


def _failed_prediction_row(model_dir: Path, output_dir: Path, error: Exception) -> dict[str, Any]:
    return {
        "status": "failed",
        "upstream": model_dir.name,
        "n_samples": np.nan,
        "accuracy": np.nan,
        "balanced_accuracy": np.nan,
        "precision_macro": np.nan,
        "recall_macro": np.nan,
        "f1_macro": np.nan,
        "mcc": np.nan,
        "model_dir": str(model_dir),
        "output_dir": str(output_dir),
        "error": str(error),
    }


def _plot_prediction_metric_bars(rows: list[dict[str, Any]], output_path: Path) -> None:
    ok_rows = [row for row in rows if row["status"] == "ok"]
    if not ok_rows:
        return

    plot_rows = sorted(ok_rows, key=lambda row: float(row["f1_macro"]), reverse=True)
    labels = [str(row["upstream"]) for row in plot_rows]
    metric_specs = [
        ("accuracy", "Accuracy"),
        ("balanced_accuracy", "Balanced Acc."),
        ("f1_macro", "Macro F1"),
        ("precision_macro", "Macro Precision"),
        ("recall_macro", "Macro Recall"),
    ]

    x = np.arange(len(labels))
    width = 0.15
    colors = ["#0072B2", "#009E73", "#D55E00", "#CC79A7", "#56B4E9"]

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
        fig, ax = plt.subplots(figsize=(10.5, 5.0))
        offsets = np.linspace(-2, 2, len(metric_specs)) * width
        for idx, ((key, label), offset) in enumerate(zip(metric_specs, offsets)):
            values = [float(row[key]) for row in plot_rows]
            bars = ax.bar(
                x + offset,
                values,
                width=width,
                label=label,
                color=colors[idx],
                edgecolor="black",
                linewidth=0.55,
            )
            for bar, value in zip(bars, values):
                ax.text(
                    bar.get_x() + bar.get_width() / 2,
                    value + 0.008,
                    f"{value:.3f}",
                    ha="center",
                    va="bottom",
                    fontsize=6.5,
                    rotation=90,
                )

        ax.set_ylabel("Score")
        ax.set_title("Prediction Performance Comparison")
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=25, ha="right")
        ax.set_ylim(0.0, 1.08)
        ax.grid(axis="y", linestyle="--", linewidth=0.5, alpha=0.45)
        ax.legend(frameon=False, ncols=3, loc="upper center", bbox_to_anchor=(0.5, 1.18))
        fig.tight_layout()
        fig.savefig(output_path)
        plt.close(fig)


def predict(
    model_dir: Path,
    input_root: Path,
    output_dir: Path | None,
    device_text: str,
    batch_size: int | None,
) -> Path:
    source_path = model_dir.resolve()
    model_dirs = _collect_model_dirs(source_path)
    is_multi = len(model_dirs) > 1

    if output_dir is None:
        run_dir = _create_prediction_run_dir(
            _default_prediction_root(source_path),
            source_path.name,
            input_root.name,
        )
    else:
        run_dir = output_dir.resolve()
        run_dir.mkdir(parents=True, exist_ok=False)

    if not is_multi:
        return predict_single_model(
            model_dir=model_dirs[0],
            input_root=input_root,
            output_dir=run_dir,
            device_text=device_text,
            batch_size=batch_size,
        )

    print(f"Multi-model prediction: {len(model_dirs)} models", flush=True)
    print(f"Prediction output folder: {run_dir}", flush=True)
    rows: list[dict[str, Any]] = []

    for index, single_model_dir in enumerate(model_dirs, start=1):
        with open(single_model_dir / "run_summary.json", "r", encoding="utf-8") as f:
            train_summary = json.load(f)
        upstream = str(train_summary.get("config", {}).get("upstream", single_model_dir.name))
        single_output = run_dir / _safe_name(upstream)
        print(f"[{index}/{len(model_dirs)} {upstream}] Predicting...", flush=True)
        try:
            result_dir = predict_single_model(
                model_dir=single_model_dir,
                input_root=input_root,
                output_dir=single_output,
                device_text=device_text,
                batch_size=batch_size,
            )
            with open(result_dir / "run_summary.json", "r", encoding="utf-8") as f:
                pred_summary = json.load(f)
            row = _prediction_row(single_model_dir, result_dir, pred_summary)
            print(
                f"[{index}/{len(model_dirs)} {upstream}] "
                f"Accuracy={row['accuracy']:.4f}, F1_macro={row['f1_macro']:.4f}",
                flush=True,
            )
        except Exception as exc:
            row = _failed_prediction_row(single_model_dir, single_output, exc)
            print(f"[{index}/{len(model_dirs)} {upstream}] Failed: {exc}", flush=True)
        rows.append(row)
        pd.DataFrame(rows).to_csv(run_dir / "prediction_comparison.csv", index=False)

    ok_rows = [row for row in rows if row["status"] == "ok"]
    figures_dir = run_dir / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)
    _plot_prediction_metric_bars(rows, figures_dir / "prediction_metrics_bar.png")

    if ok_rows:
        best = max(
            ok_rows,
            key=lambda row: (
                float(row["f1_macro"]),
                float(row["accuracy"]),
                float(row["balanced_accuracy"]),
            ),
        )
        best_payload = {
            "status": "ok",
            "selection_metric": "f1_macro",
            "tie_breakers": ["accuracy", "balanced_accuracy"],
            "best_model": best,
            "all_models": rows,
        }
        print(
            f"Best prediction model: {best['upstream']} "
            f"(F1_macro={best['f1_macro']:.4f}, Accuracy={best['accuracy']:.4f})",
            flush=True,
        )
    else:
        best_payload = {
            "status": "failed",
            "message": "No model prediction completed successfully.",
            "all_models": rows,
        }

    with open(run_dir / "best_prediction.json", "w", encoding="utf-8") as f:
        json.dump(best_payload, f, indent=2, ensure_ascii=False)

    with open(run_dir / "run_summary.json", "w", encoding="utf-8") as f:
        json.dump(
            {
                "created_at": datetime.now().isoformat(timespec="seconds"),
                "mode": "multi_model_prediction",
                "source_model_dir": str(source_path),
                "input_root": str(input_root.resolve()),
                "output_dir": str(run_dir.resolve()),
                "models": rows,
                "artifacts": [
                    "prediction_comparison.csv",
                    "best_prediction.json",
                    "figures/prediction_metrics_bar.png",
                    "run_summary.json",
                ],
            },
            f,
            indent=2,
            ensure_ascii=False,
        )

    if not ok_rows:
        raise BaselineError(f"No model prediction completed successfully. See {run_dir}")
    return run_dir


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Predict a labeled spectral folder with a trained HURTSPEC S3PRL output folder."
    )
    parser.add_argument(
        "--model-dir",
        required=True,
        help="Single model output folder, or a multi-upstream parent folder containing best_model.json.",
    )
    parser.add_argument("--input-root", default="data_test", help="Prediction data root.")
    parser.add_argument("--output-dir", default=None, help="Optional prediction output folder.")
    parser.add_argument("--device", default="auto", help="auto, cpu, or cuda.")
    parser.add_argument("--batch-size", type=int, default=None, help="Override embedding batch size.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    predict(
        model_dir=Path(args.model_dir),
        input_root=Path(args.input_root),
        output_dir=Path(args.output_dir) if args.output_dir else None,
        device_text=args.device,
        batch_size=args.batch_size,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
