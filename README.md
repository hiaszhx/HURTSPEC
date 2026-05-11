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
order = ["snv", "wavelet", "pls"]
```

The classifier head is selected in `[classifier]`:

```toml
head_type = "mlp"     # or "linear"
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
