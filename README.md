# Long-Term Spectrum Monitoring: Anomaly Detection Benchmark

The Python script evaluates the detectors on one feature CSV from the TCCN case study per run.
Keep `data_baseline_compare_no_classic_log1p_Sz_pot_mask_new_compare.py` and this README in the
repository root, next to the `abnormal_timeseries_csv/` directory.

## Setup

Install the Python dependencies in your environment:

```sh
python -m pip install numpy pandas scikit-learn torch matplotlib scipy
```

The script uses CUDA when it is available to PyTorch and otherwise uses the CPU. To run the
benchmark from the feature CSVs, you do not need raw I/Q data or MATLAB.

## Run or switch scenarios

Run the script once for each anomaly pattern and severity you want to evaluate.
For example, from the repository root, run strong Ab1 with:

```sh
python data_baseline_compare_no_classic_log1p_Sz_pot_mask_new_compare.py --csv "abnormal_timeseries_csv/ab1/feature_levelS_FM_result_set_retime_new.csv"
```

This is also the default when `--csv` is omitted. To switch to medium Ab2, change only `--csv`:

```sh
python data_baseline_compare_no_classic_log1p_Sz_pot_mask_new_compare.py --csv "abnormal_timeseries_csv/ab2/feature_LevelM1_interval_abnormal_result_set_retime.csv"
```

Use a path relative to the repository root, including the `abnormal_timeseries_csv/` prefix.
The script resolves this path from its own location, not from the terminal's current directory.
Use the exact filename, including capitalization, in your checkout. The table lists the paths used
by the experiment files; combine `abnormal_timeseries_csv/`, the scenario folder, and the selected
filename.

| Scenario folder | Strong | Medium | Weak |
| --- | --- | --- | --- |
| `ab1/` | `feature_levelS_FM_result_set_retime_new.csv` | `feature_levelM_FM_result_set_retime_new.csv` | `feature_levelW_FM_result_set_retime_new.csv` |
| `ab2/` | `feature_LevelS1_interval_abnormal_result_set_retime.csv` | `feature_LevelM1_interval_abnormal_result_set_retime.csv` | `feature_LevelW1_interval_abnormal_result_set_retime.csv` |
| `ab3/` | `feature_levelS_denoise_signal_result_set_retime.csv` | `feature_levelM_denoise_signal_result_set_retime.csv` | `feature_levelW_denoise_signal_result_set_retime.csv` |
| `ab4/` | `feature_levelS_removepeak_result_set_retime_new.csv` | `feature_levelM_removepeak_result_set_retime_new.csv` | `feature_levelW_removepeak_result_set_retime_new.csv` |

Keep the CSV's original row and column order. The first column is the time index and is excluded
from the inputs. All intermediate columns are used as provided, including the relative-time
coordinate if present. The last column is the binary label, with `0` for nominal observations and
`1` for anomalous observations.

## Results

All files from a run are saved under `results/<CSV relative path without .csv>/run_NNN/`.
For example, the first strong Ab1 run writes to:

```text
results/abnormal_timeseries_csv/ab1/feature_levelS_FM_result_set_retime_new/run_001/
```

Different CSVs have distinct output folders, even if their filenames match in different scenario
folders. Repeating the same CSV selects the next available run folder, such as `run_002`, without
overwriting existing files. The script prints the selected input and output paths.

| File | Contents |
| --- | --- |
| `baseline_results_with_ours_transformer_feat.csv` | Per-method thresholds and point-level and event-level metrics |
| `baseline_ours_f1_bar.png`, `baseline_ours_auc_bar.png` | Summary plots for F1 and AUC |
| `roc_pr_*.png`, `anomaly_segments_*.png`, `scores_*.png` | Per-method curves, event timelines, and anomaly scores |
