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
