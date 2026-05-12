# HURTSPEC S3PRL Baseline

Standalone spectral classification baseline using local `s3prl-main`.

## Run

```powershell
E:\VScode_Python\.venv\Scripts\python.exe E:\VScode_Python\HURTSPEC\run_s3prl_baseline.py --config E:\VScode_Python\HURTSPEC\config.toml
```

All parameters are read from `config.toml`. The required S3PRL model selector is:

```toml
[s3prl]
upstream = "distilhubert"
```

For multiple models:

```toml
[s3prl]
upstream = ["hubert", "distilhubert"]
```

Single-model runs write to:

```text
output_s3prl/{upstream}_YYYYMMDD_HHMMSS
```

Multi-model runs write all results under one folder:

```text
output_s3prl/multi_upstream_YYYYMMDD_HHMMSS
```

Each model has its own subfolder inside the run folder. The parent folder also
contains `model_comparison.csv`, `best_model.json`, and `run_summary.json`.
Multi-model runs also save SCI-style comparison figures:

```text
figures/model_metrics_bar.png
figures/model_complexity_bar.png
```

Single-model training figures include PCA views of the pipeline:

```text
figures/pca_input_spectra.png
figures/pca_preprocessed_spectra.png       # only when preprocessing actually runs
figures/pca_true_labels.png                # S3PRL upstream embedding PCA
figures/pca_test_predictions.png           # upstream embedding PCA, test errors marked
figures/pca_classifier_logits.png          # classifier-head output PCA
```

PNG figures are saved under the run folder's `figures` subfolder. Run parameters,
metrics, preprocessing metadata, model complexity, and artifact names are stored
in one `run_summary.json`.

## Config Notes

Preprocessing is intentionally limited to:

```toml
[preprocess]
order = ["snv", "wavelet", "band_selection", "pls"]
```

Band selection is optional and is off by default, so the baseline remains
"no band selection". Enable it as a preprocessing step:

```toml
[preprocess]
enabled = true
order = ["band_selection"]

[band_selection]
enabled = true
method = "pls_vip"         # none, manual, pls_vip, lasso, cars, ga, learnable_gate, band_attention
manual_ranges = []         # e.g. [[420, 520], [760, 820]] when method = "manual"
top_ratio = 0.25           # or set top_k > 0
min_bands = 16
```

Manual selection (`manual`) keeps all wavelength points inside the configured,
possibly discontinuous ranges. Traditional selectors fit only on the train split:
`pls_vip`, `lasso`, `cars`, and the GA selector (`ga`), which evolves fixed-size
band subsets using internal PLS-DA validation accuracy as fitness. Learnable selectors (`learnable_gate`,
`band_attention`) train a small band-scoring network on the train split.

When band selection is enabled, HURTSPEC uses a two-branch fusion path: the full
continuous spectrum is still passed to S3PRL for the global sequence embedding,
while the selected discrete band intensities bypass S3PRL and are concatenated
with the S3PRL embedding before the classifier head. This avoids stitching
discontinuous wavelengths into a fake continuous sequence. Runs save
`selected_bands.csv`, `band_selection_scores.csv`,
`band_selection_summary.csv/json`, the SCI-style selected-band curve
`figures/band_selection_heatmap.png`, and the replayable state in
`preprocess_state.pt`.

To run the full band-selection sweep:

```powershell
E:\VScode_Python\.venv\Scripts\python.exe E:\VScode_Python\HURTSPEC\run_band_selection_sweep.py --config E:\VScode_Python\HURTSPEC\config.toml --repeats 3
```

The sweep evaluates no band selection, manual selection once using
`band_selection.manual_ranges`, plus `pls_vip`, `cars`, `learnable_gate`, and
`band_attention` at `top_ratio = 0.25, 0.5, 0.75`. By default it repeats the
whole plan 3 times with different `random_state` values.
It writes `band_selection_sweep_results.csv`,
`band_selection_sweep_summary.csv`, and line figures under the sweep output
folder. Use `--sweep-dir {existing_folder}` to resume into a previous sweep
directory.

To compare manual band selection in single-branch vs dual-branch mode:

```powershell
E:\VScode_Python\.venv\Scripts\python.exe E:\VScode_Python\HURTSPEC\run_band_selection_sweep.py --config E:\VScode_Python\HURTSPEC\config.toml --manual-branch-compare --repeats 3
```

The classifier head is selected in `[classifier]`:

```toml
head_type = "mlp"     # linear, mlp, or prototype
```

`prototype` uses a projection encoder plus learnable class prototypes. It can
also add Supervised Contrastive Loss to the weighted cross-entropy objective:

```toml
[classifier]
head_type = "prototype"
hidden_dim = 256
dropout = 0.3
prototype_dim = 128
prototype_temperature = 0.1
supcon_enabled = true
supcon_weight = 0.1
supcon_temperature = 0.1
```

Label smoothing is optional:

```toml
[classifier]
label_smoothing_enabled = false
label_smoothing = 0.1
```

## Predict New Data

Use a trained single-model folder, or a multi-model parent folder. Prediction
results are written to a new folder named `output_s3prl_predict`, at the same
level as `output_s3prl`.

When a multi-model parent folder is passed, every model subfolder is predicted
on the same input data, then compared in `prediction_comparison.csv` and
`best_prediction.json`.

Basic form:

```powershell
python predict_s3prl.py --model-dir {path}
```

With a custom prediction dataset:

```powershell
python predict_s3prl.py --model-dir {path} --input-root {data_test_path}
```

Example with this project:

```powershell
E:\VScode_Python\.venv\Scripts\python.exe E:\VScode_Python\HURTSPEC\predict_s3prl.py --model-dir E:\VScode_Python\HURTSPEC\output_s3prl\multi_upstream_baseline --input-root E:\VScode_Python\HURTSPEC\data_test
```

Prediction outputs include `predictions.csv`, `metrics.json`,
`classification_report.csv`, `confusion_matrix.csv`, `figures/confusion_matrix.png`,
and `run_summary.json`.

For multi-model prediction, the parent prediction folder also includes:

```text
prediction_comparison.csv
best_prediction.json
figures/prediction_metrics_bar.png
run_summary.json
```
