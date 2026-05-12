from __future__ import annotations

import argparse
from dataclasses import asdict, replace
from datetime import datetime
import json
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from run_s3prl_baseline import (
    BaselineConfig,
    BandSelectionStepConfig,
    BaselineError,
    _safe_name,
    load_config,
    run_baseline,
)


SWEEP_METHODS = [
    "none",
    "manual",
    "pls_vip",
    "cars",
    "learnable_gate",
    "band_attention",
]
RATIO_METHODS = [
    "pls_vip",
    "cars",
    "learnable_gate",
    "band_attention",
]
SWEEP_RATIOS = [0.25, 0.5, 0.75]
DEFAULT_REPEATS = 1


def _create_sweep_dir(base_dir: Path) -> Path:
    base_dir.mkdir(parents=True, exist_ok=True)
    run_id = f"band_selection_sweep_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    out_dir = base_dir / run_id
    suffix = 1
    while out_dir.exists():
        out_dir = base_dir / f"{run_id}_{suffix:02d}"
        suffix += 1
    out_dir.mkdir(parents=True, exist_ok=False)
    (out_dir / "figures").mkdir(parents=True, exist_ok=True)
    return out_dir


def _setting_name(method: str, ratio: float | None) -> str:
    if method == "none":
        return "baseline_no_band_selection"
    if method == "manual":
        return "manual_band_selection"
    ratio_text = f"{float(ratio):.2f}".replace(".", "p")
    return f"{method}_top_ratio_{ratio_text}"


def _manual_branch_setting_name(fusion_mode: str) -> str:
    return f"manual_{fusion_mode}_branch"


def _repeat_setting_name(method: str, ratio: float | None, repeat: int) -> str:
    return f"repeat_{int(repeat):02d}_{_setting_name(method, ratio)}"


def _build_setting_config(
    base_config: BaselineConfig,
    method: str,
    ratio: float | None,
    upstream: str,
    output_root: Path,
    repeat: int,
) -> BaselineConfig:
    bs = base_config.preprocess.band_selection
    if method == "none":
        band_selection = replace(
            bs,
            enabled=False,
            method="none",
            top_k=0,
        )
    elif method == "manual":
        if not bs.manual_ranges:
            raise BaselineError(
                "Sweep method='manual' requires band_selection.manual_ranges in config.toml."
            )
        band_selection = replace(
            bs,
            enabled=True,
            method="manual",
            top_k=0,
        )
    else:
        if ratio is None:
            raise BaselineError(f"Missing top_ratio for method={method}")
        band_selection = replace(
            bs,
            enabled=True,
            method=method,
            top_k=0,
            top_ratio=float(ratio),
        )

    preprocess = replace(
        base_config.preprocess,
        enabled=True,
        order=["band_selection"],
        band_selection=band_selection,
    )
    return replace(
        base_config,
        upstream=upstream,
        upstreams=[upstream],
        output_root=str(output_root),
        random_state=int(base_config.random_state) + int(repeat) - 1,
        preprocess=preprocess,
    )


def _build_manual_branch_config(
    base_config: BaselineConfig,
    fusion_mode: str,
    upstream: str,
    output_root: Path,
    repeat: int,
) -> BaselineConfig:
    bs = base_config.preprocess.band_selection
    if not bs.manual_ranges:
        raise BaselineError(
            "Manual branch comparison requires band_selection.manual_ranges in config.toml."
        )
    band_selection = replace(
        bs,
        enabled=True,
        method="manual",
        fusion_mode=fusion_mode,
        top_k=0,
    )
    preprocess = replace(
        base_config.preprocess,
        enabled=True,
        order=["band_selection"],
        band_selection=band_selection,
    )
    return replace(
        base_config,
        upstream=upstream,
        upstreams=[upstream],
        output_root=str(output_root),
        random_state=int(base_config.random_state) + int(repeat) - 1,
        preprocess=preprocess,
    )


def _read_json(path: Path) -> dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _flatten_result_row(
    repeat: int,
    setting: str,
    method: str,
    ratio: float | None,
    upstream: str,
    model_dir: Path,
    summary: dict[str, Any],
) -> dict[str, Any]:
    test_metrics = summary.get("metrics", {}).get("test", {})
    all_metrics = summary.get("metrics", {}).get("all_data", {})
    dataset = summary.get("dataset", {})
    band_summary = summary.get("band_selection_summary", {})
    complexity = summary.get("complexity", {})
    training = complexity.get("training", {})

    return {
        "repeat": int(repeat),
        "setting": setting,
        "method": method,
        "top_ratio": np.nan if ratio is None else float(ratio),
        "upstream": upstream,
        "status": "ok",
        "test_accuracy": test_metrics.get("accuracy", np.nan),
        "test_balanced_accuracy": test_metrics.get("balanced_accuracy", np.nan),
        "test_f1_macro": test_metrics.get("f1_macro", np.nan),
        "test_precision_macro": test_metrics.get("precision_macro", np.nan),
        "test_recall_macro": test_metrics.get("recall_macro", np.nan),
        "test_mcc": test_metrics.get("mcc", np.nan),
        "all_data_accuracy": all_metrics.get("accuracy", np.nan),
        "all_data_f1_macro": all_metrics.get("f1_macro", np.nan),
        "original_band_count": band_summary.get("original_band_count", np.nan),
        "selected_band_count": band_summary.get("selected_band_count", np.nan),
        "removed_band_count": band_summary.get("removed_band_count", np.nan),
        "selected_ratio": band_summary.get("selected_ratio", np.nan),
        "best_epoch": training.get("best_epoch", np.nan),
        "n_train": dataset.get("n_train", np.nan),
        "n_test": dataset.get("n_test", np.nan),
        "output_dir": str(model_dir),
        "error": "",
    }


def _failed_row(
    repeat: int,
    setting: str,
    method: str,
    ratio: float | None,
    upstream: str,
    model_dir: Path,
    error: Exception,
) -> dict[str, Any]:
    return {
        "repeat": int(repeat),
        "setting": setting,
        "method": method,
        "top_ratio": np.nan if ratio is None else float(ratio),
        "upstream": upstream,
        "status": "failed",
        "test_accuracy": np.nan,
        "test_balanced_accuracy": np.nan,
        "test_f1_macro": np.nan,
        "test_precision_macro": np.nan,
        "test_recall_macro": np.nan,
        "test_mcc": np.nan,
        "all_data_accuracy": np.nan,
        "all_data_f1_macro": np.nan,
        "original_band_count": np.nan,
        "selected_band_count": np.nan,
        "removed_band_count": np.nan,
        "selected_ratio": np.nan,
        "best_epoch": np.nan,
        "n_train": np.nan,
        "n_test": np.nan,
        "output_dir": str(model_dir),
        "error": str(error),
    }


def _add_branch_column(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    enriched = []
    for row in rows:
        row_copy = dict(row)
        if "fusion_mode" not in row_copy:
            setting = str(row_copy.get("setting", ""))
            if "manual_single_branch" in setting:
                row_copy["fusion_mode"] = "single"
            elif "manual_dual_branch" in setting:
                row_copy["fusion_mode"] = "dual"
            else:
                row_copy["fusion_mode"] = ""
        enriched.append(row_copy)
    return enriched


def _aggregate_results(rows: list[dict[str, Any]]) -> pd.DataFrame:
    frame = pd.DataFrame(rows)
    ok = frame[frame["status"] == "ok"].copy()
    if ok.empty:
        return pd.DataFrame()

    grouped = ok.groupby(["method", "top_ratio"], dropna=False)
    summary = grouped.agg(
        n_runs=("upstream", "count"),
        n_repeats=("repeat", "nunique"),
        n_upstreams=("upstream", "nunique"),
        test_f1_macro_mean=("test_f1_macro", "mean"),
        test_f1_macro_std=("test_f1_macro", "std"),
        test_accuracy_mean=("test_accuracy", "mean"),
        test_accuracy_std=("test_accuracy", "std"),
        selected_band_count_mean=("selected_band_count", "mean"),
        selected_ratio_mean=("selected_ratio", "mean"),
    ).reset_index()

    summary["method_order"] = summary["method"].map({name: i for i, name in enumerate(SWEEP_METHODS)})
    summary = summary.sort_values(["method_order", "top_ratio"], na_position="first")
    return summary.drop(columns=["method_order"])


def _plot_metric_lines(rows: list[dict[str, Any]], output_path: Path) -> None:
    frame = pd.DataFrame(rows)
    ok = frame[frame["status"] == "ok"].copy()
    if ok.empty:
        return

    plot_data = (
        ok[ok["method"].isin(RATIO_METHODS)]
        .groupby(["method", "top_ratio"], dropna=False)["test_f1_macro"]
        .mean()
        .reset_index()
    )
    baseline = ok[ok["method"] == "none"]["test_f1_macro"].mean()

    colors = {
        "pls_vip": "#1F4E79",
        "cars": "#008B8B",
        "learnable_gate": "#B2182B",
        "band_attention": "#D55E00",
    }
    markers = {
        "pls_vip": "o",
        "cars": "s",
        "learnable_gate": "D",
        "band_attention": "P",
    }

    with plt.rc_context(
        {
            "font.family": "DejaVu Serif",
            "axes.linewidth": 0.9,
            "axes.edgecolor": "black",
            "axes.labelsize": 10.5,
            "axes.titlesize": 11.5,
            "xtick.labelsize": 9.5,
            "ytick.labelsize": 9.5,
            "legend.fontsize": 8.5,
            "figure.dpi": 160,
            "savefig.dpi": 300,
        }
    ):
        fig, ax = plt.subplots(figsize=(8.2, 4.8), constrained_layout=True)
        manual_f1 = ok[ok["method"] == "manual"]["test_f1_macro"].mean()

        for method in RATIO_METHODS:
            method_rows = plot_data[plot_data["method"] == method].sort_values("top_ratio")
            if method_rows.empty:
                continue
            ax.plot(
                method_rows["top_ratio"],
                method_rows["test_f1_macro"],
                marker=markers.get(method, "o"),
                linewidth=1.6,
                markersize=5,
                color=colors.get(method, "black"),
                label=method,
            )

        if not np.isnan(baseline):
            ax.axhline(
                float(baseline),
                color="#4D4D4D",
                linestyle="--",
                linewidth=1.1,
                label="No band selection",
            )
        if not np.isnan(manual_f1):
            ax.axhline(
                float(manual_f1),
                color="#6A3D9A",
                linestyle=":",
                linewidth=1.2,
                label="Manual selection",
            )

        ax.set_xlabel("Selected band ratio")
        ax.set_ylabel("Test Macro F1")
        ax.set_title("Band Selection Sweep")
        ax.set_xticks(SWEEP_RATIOS)
        ax.grid(axis="y", linestyle="--", linewidth=0.45, alpha=0.35)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.legend(frameon=False, ncols=2)
        fig.savefig(output_path)
        plt.close(fig)


def _plot_upstream_lines(rows: list[dict[str, Any]], output_path: Path) -> None:
    frame = pd.DataFrame(rows)
    ok = frame[(frame["status"] == "ok") & (frame["method"].isin(RATIO_METHODS))].copy()
    if ok.empty:
        return

    methods = list(RATIO_METHODS)
    with plt.rc_context(
        {
            "font.family": "DejaVu Serif",
            "axes.linewidth": 0.9,
            "axes.edgecolor": "black",
            "axes.labelsize": 9.5,
            "axes.titlesize": 10.5,
            "xtick.labelsize": 8.5,
            "ytick.labelsize": 8.5,
            "legend.fontsize": 7.5,
            "figure.dpi": 160,
            "savefig.dpi": 300,
        }
    ):
        upstreams = sorted(ok["upstream"].unique())
        n_cols = min(3, len(upstreams))
        n_rows = int(np.ceil(len(upstreams) / n_cols))
        fig, axes = plt.subplots(
            n_rows,
            n_cols,
            figsize=(4.2 * n_cols, 3.2 * n_rows),
            sharex=True,
            sharey=True,
            constrained_layout=True,
        )
        axes_arr = np.asarray(axes).reshape(-1)
        for ax, upstream in zip(axes_arr, upstreams):
            sub = ok[ok["upstream"] == upstream]
            baseline = frame[
                (frame["status"] == "ok")
                & (frame["method"] == "none")
                & (frame["upstream"] == upstream)
            ]["test_f1_macro"].mean()
            manual_f1 = frame[
                (frame["status"] == "ok")
                & (frame["method"] == "manual")
                & (frame["upstream"] == upstream)
            ]["test_f1_macro"].mean()
            for method in methods:
                method_rows = sub[sub["method"] == method].sort_values("top_ratio")
                if method_rows.empty:
                    continue
                ax.plot(
                    method_rows["top_ratio"],
                    method_rows["test_f1_macro"],
                    marker="o",
                    linewidth=1.2,
                    markersize=3.5,
                    label=method,
                )
            if not np.isnan(baseline):
                ax.axhline(
                    float(baseline),
                    color="#4D4D4D",
                    linestyle="--",
                    linewidth=1.0,
                    label="No band selection",
                )
            if not np.isnan(manual_f1):
                ax.axhline(
                    float(manual_f1),
                    color="#6A3D9A",
                    linestyle=":",
                    linewidth=1.0,
                    label="Manual selection",
                )
            ax.set_title(upstream)
            ax.grid(axis="y", linestyle="--", linewidth=0.4, alpha=0.35)
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)
            ax.set_xticks(SWEEP_RATIOS)

        for ax in axes_arr[len(upstreams):]:
            ax.axis("off")
        handles, labels = [], []
        for ax in axes_arr[: len(upstreams)]:
            ax_handles, ax_labels = ax.get_legend_handles_labels()
            for handle, label in zip(ax_handles, ax_labels):
                if label not in labels:
                    handles.append(handle)
                    labels.append(label)
        if handles:
            fig.legend(handles, labels, frameon=False, ncols=3, loc="upper center")
        fig.supxlabel("Selected band ratio")
        fig.supylabel("Test Macro F1")
        fig.savefig(output_path)
        plt.close(fig)


def _plot_manual_branch_comparison(rows: list[dict[str, Any]], output_path: Path) -> None:
    frame = pd.DataFrame(rows)
    ok = frame[frame["status"] == "ok"].copy()
    if ok.empty:
        return

    summary = ok.groupby("fusion_mode")["test_f1_macro"].agg(["mean", "std"]).reindex(["single", "dual"])
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
        }
    ):
        fig, ax = plt.subplots(figsize=(5.4, 4.3), constrained_layout=True)
        x = np.arange(len(summary.index))
        values = summary["mean"].to_numpy(dtype=float)
        errors = summary["std"].fillna(0.0).to_numpy(dtype=float)
        ax.bar(
            x,
            values,
            yerr=errors,
            capsize=4,
            color=["#4D4D4D", "#1F4E79"],
            edgecolor="black",
            linewidth=0.7,
        )
        for xi, value in zip(x, values):
            if np.isfinite(value):
                ax.text(xi, value + 0.01, f"{value:.3f}", ha="center", va="bottom", fontsize=9)
        ax.set_xticks(x)
        ax.set_xticklabels(["Single branch", "Dual branch"])
        ax.set_ylabel("Test Macro F1")
        ax.set_title("Manual Band Selection: Single vs Dual")
        ax.set_ylim(0.0, min(1.05, max(1.0, np.nanmax(values) + 0.12)))
        ax.grid(axis="y", linestyle="--", linewidth=0.45, alpha=0.35)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        fig.savefig(output_path)
        plt.close(fig)


def run_sweep(
    config_path: Path,
    output_root: Path | None,
    sweep_dir: Path | None,
    upstream_filter: list[str] | None,
    repeats: int,
    resume: bool,
) -> Path:
    base_config = load_config(config_path)
    if sweep_dir is None:
        parent_dir = _create_sweep_dir(output_root or Path(base_config.output_root))
    else:
        parent_dir = sweep_dir.resolve()
        parent_dir.mkdir(parents=True, exist_ok=True)
        (parent_dir / "figures").mkdir(parents=True, exist_ok=True)

    upstreams = base_config.upstreams
    if upstream_filter:
        requested = set(upstream_filter)
        upstreams = [name for name in upstreams if name in requested]
        missing = sorted(requested - set(upstreams))
        if missing:
            raise BaselineError(f"Requested upstream(s) not in config: {', '.join(missing)}")
    if not upstreams:
        raise BaselineError("No upstreams selected for sweep.")
    repeats = int(repeats)
    if repeats <= 0:
        raise BaselineError("--repeats must be > 0.")

    settings: list[tuple[str, float | None]] = [("none", None), ("manual", None)]
    for method in RATIO_METHODS:
        for ratio in SWEEP_RATIOS:
            settings.append((method, ratio))

    plan = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "config_path": str(config_path.resolve()),
        "upstreams": upstreams,
        "repeats": int(repeats),
        "random_states": [int(base_config.random_state) + i for i in range(repeats)],
        "settings": [
            {"method": method, "top_ratio": ratio, "name": _repeat_setting_name(method, ratio, repeat)}
            for repeat in range(1, repeats + 1)
            for method, ratio in settings
        ],
        "total_model_runs": int(repeats * len(upstreams) * len(settings)),
    }
    with open(parent_dir / "run_plan.json", "w", encoding="utf-8") as f:
        json.dump(plan, f, indent=2, ensure_ascii=False)

    rows: list[dict[str, Any]] = []
    results_csv = parent_dir / "band_selection_sweep_results.csv"
    total = repeats * len(upstreams) * len(settings)
    index = 0

    for repeat in range(1, repeats + 1):
        for method, ratio in settings:
            setting = _repeat_setting_name(method, ratio, repeat)
            setting_dir = parent_dir / setting
            setting_dir.mkdir(parents=True, exist_ok=True)
            for upstream in upstreams:
                index += 1
                model_dir = setting_dir / _safe_name(upstream)
                summary_path = model_dir / "run_summary.json"
                print(f"[{index}/{total}] repeat={repeat} | {setting} | {upstream}", flush=True)
                try:
                    if resume and summary_path.exists():
                        print("  Reusing existing run_summary.json", flush=True)
                    else:
                        cfg = _build_setting_config(
                            base_config=base_config,
                            method=method,
                            ratio=ratio,
                            upstream=upstream,
                            output_root=setting_dir,
                            repeat=repeat,
                        )
                        run_baseline(
                            cfg,
                            output_dir=model_dir,
                            progress_prefix=f"[{setting} {upstream}]",
                        )

                    summary = _read_json(summary_path)
                    row = _flatten_result_row(
                        repeat=repeat,
                        setting=setting,
                        method=method,
                        ratio=ratio,
                        upstream=upstream,
                        model_dir=model_dir,
                        summary=summary,
                    )
                except Exception as exc:
                    row = _failed_row(
                        repeat=repeat,
                        setting=setting,
                        method=method,
                        ratio=ratio,
                        upstream=upstream,
                        model_dir=model_dir,
                        error=exc,
                    )
                    print(f"  Failed: {exc}", flush=True)

                rows.append(row)
                pd.DataFrame(rows).to_csv(results_csv, index=False)

    summary = _aggregate_results(rows)
    if not summary.empty:
        summary.to_csv(parent_dir / "band_selection_sweep_summary.csv", index=False)

    figures_dir = parent_dir / "figures"
    _plot_metric_lines(rows, figures_dir / "band_selection_sweep_f1_lines.png")
    _plot_upstream_lines(rows, figures_dir / "band_selection_sweep_f1_by_upstream.png")

    with open(parent_dir / "run_summary.json", "w", encoding="utf-8") as f:
        json.dump(
            {
                **plan,
                "completed_at": datetime.now().isoformat(timespec="seconds"),
                "artifacts": [
                    "run_plan.json",
                    "band_selection_sweep_results.csv",
                    "band_selection_sweep_summary.csv",
                    "figures/band_selection_sweep_f1_lines.png",
                    "figures/band_selection_sweep_f1_by_upstream.png",
                    "run_summary.json",
                ],
            },
            f,
            indent=2,
            ensure_ascii=False,
        )
    print(f"Sweep finished: {parent_dir}", flush=True)
    return parent_dir


def run_manual_branch_compare(
    config_path: Path,
    output_root: Path | None,
    sweep_dir: Path | None,
    upstream_filter: list[str] | None,
    repeats: int,
    resume: bool,
) -> Path:
    base_config = load_config(config_path)
    if sweep_dir is None:
        parent_dir = _create_sweep_dir(output_root or Path(base_config.output_root))
    else:
        parent_dir = sweep_dir.resolve()
        parent_dir.mkdir(parents=True, exist_ok=True)
        (parent_dir / "figures").mkdir(parents=True, exist_ok=True)

    upstreams = base_config.upstreams
    if upstream_filter:
        requested = set(upstream_filter)
        upstreams = [name for name in upstreams if name in requested]
        missing = sorted(requested - set(upstreams))
        if missing:
            raise BaselineError(f"Requested upstream(s) not in config: {', '.join(missing)}")
    if not upstreams:
        raise BaselineError("No upstreams selected for manual branch comparison.")

    repeats = int(repeats)
    if repeats <= 0:
        raise BaselineError("--repeats must be > 0.")

    fusion_modes = ["single", "dual"]
    plan = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "mode": "manual_branch_compare",
        "config_path": str(config_path.resolve()),
        "upstreams": upstreams,
        "repeats": int(repeats),
        "fusion_modes": fusion_modes,
        "random_states": [int(base_config.random_state) + i for i in range(repeats)],
        "total_model_runs": int(repeats * len(upstreams) * len(fusion_modes)),
    }
    with open(parent_dir / "run_plan.json", "w", encoding="utf-8") as f:
        json.dump(plan, f, indent=2, ensure_ascii=False)

    rows: list[dict[str, Any]] = []
    results_csv = parent_dir / "manual_branch_comparison_results.csv"
    total = int(plan["total_model_runs"])
    index = 0

    for repeat in range(1, repeats + 1):
        for fusion_mode in fusion_modes:
            setting = f"repeat_{repeat:02d}_{_manual_branch_setting_name(fusion_mode)}"
            setting_dir = parent_dir / setting
            setting_dir.mkdir(parents=True, exist_ok=True)
            for upstream in upstreams:
                index += 1
                model_dir = setting_dir / _safe_name(upstream)
                summary_path = model_dir / "run_summary.json"
                print(f"[{index}/{total}] repeat={repeat} | {setting} | {upstream}", flush=True)
                try:
                    if resume and summary_path.exists():
                        print("  Reusing existing run_summary.json", flush=True)
                    else:
                        cfg = _build_manual_branch_config(
                            base_config=base_config,
                            fusion_mode=fusion_mode,
                            upstream=upstream,
                            output_root=setting_dir,
                            repeat=repeat,
                        )
                        run_baseline(
                            cfg,
                            output_dir=model_dir,
                            progress_prefix=f"[{setting} {upstream}]",
                        )

                    summary = _read_json(summary_path)
                    row = _flatten_result_row(
                        repeat=repeat,
                        setting=setting,
                        method="manual",
                        ratio=None,
                        upstream=upstream,
                        model_dir=model_dir,
                        summary=summary,
                    )
                    row["fusion_mode"] = fusion_mode
                except Exception as exc:
                    row = _failed_row(
                        repeat=repeat,
                        setting=setting,
                        method="manual",
                        ratio=None,
                        upstream=upstream,
                        model_dir=model_dir,
                        error=exc,
                    )
                    row["fusion_mode"] = fusion_mode
                    print(f"  Failed: {exc}", flush=True)

                rows.append(row)
                pd.DataFrame(rows).to_csv(results_csv, index=False)

    rows = _add_branch_column(rows)
    frame = pd.DataFrame(rows)
    ok = frame[frame["status"] == "ok"].copy()
    if not ok.empty:
        summary_frame = ok.groupby("fusion_mode").agg(
            n_runs=("upstream", "count"),
            n_repeats=("repeat", "nunique"),
            n_upstreams=("upstream", "nunique"),
            test_f1_macro_mean=("test_f1_macro", "mean"),
            test_f1_macro_std=("test_f1_macro", "std"),
            test_accuracy_mean=("test_accuracy", "mean"),
            test_accuracy_std=("test_accuracy", "std"),
            selected_band_count_mean=("selected_band_count", "mean"),
        ).reset_index()
        summary_frame.to_csv(parent_dir / "manual_branch_comparison_summary.csv", index=False)

        pivot = ok.pivot_table(
            index=["repeat", "upstream"],
            columns="fusion_mode",
            values="test_f1_macro",
            aggfunc="mean",
        ).reset_index()
        if {"single", "dual"}.issubset(pivot.columns):
            pivot["delta_dual_minus_single"] = pivot["dual"] - pivot["single"]
        pivot.to_csv(parent_dir / "manual_branch_comparison_paired_delta.csv", index=False)

    figures_dir = parent_dir / "figures"
    _plot_manual_branch_comparison(rows, figures_dir / "manual_branch_comparison_f1.png")

    with open(parent_dir / "run_summary.json", "w", encoding="utf-8") as f:
        json.dump(
            {
                **plan,
                "completed_at": datetime.now().isoformat(timespec="seconds"),
                "artifacts": [
                    "run_plan.json",
                    "manual_branch_comparison_results.csv",
                    "manual_branch_comparison_summary.csv",
                    "manual_branch_comparison_paired_delta.csv",
                    "figures/manual_branch_comparison_f1.png",
                    "run_summary.json",
                ],
            },
            f,
            indent=2,
            ensure_ascii=False,
        )
    print(f"Manual branch comparison finished: {parent_dir}", flush=True)
    return parent_dir


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run HURTSPEC band-selection sweep and plot line charts."
    )
    parser.add_argument("--config", default="config.toml", help="Base TOML config.")
    parser.add_argument("--output-root", default=None, help="Optional sweep output root.")
    parser.add_argument(
        "--sweep-dir",
        default=None,
        help="Existing or explicit sweep directory. Use this to resume an interrupted sweep.",
    )
    parser.add_argument(
        "--upstream",
        action="append",
        default=None,
        help="Limit to one upstream. Can be repeated. Default: all upstreams in config.",
    )
    parser.add_argument(
        "--repeats",
        type=int,
        default=DEFAULT_REPEATS,
        help=f"Number of repeated runs with different random_state values. Default: {DEFAULT_REPEATS}.",
    )
    parser.add_argument(
        "--manual-branch-compare",
        action="store_true",
        help="Only compare manual band selection in single-branch and dual-branch modes.",
    )
    parser.add_argument(
        "--no-resume",
        action="store_true",
        help="Do not reuse existing run_summary.json files inside the sweep folder.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    common_kwargs = {
        "config_path": Path(args.config),
        "output_root": Path(args.output_root) if args.output_root else None,
        "sweep_dir": Path(args.sweep_dir) if args.sweep_dir else None,
        "upstream_filter": args.upstream,
        "repeats": int(args.repeats),
        "resume": not bool(args.no_resume),
    }
    if args.manual_branch_compare:
        run_manual_branch_compare(**common_kwargs)
    else:
        run_sweep(**common_kwargs)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
