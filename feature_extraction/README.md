# I/Q Feature Extraction for the TCCN Case Study

This directory contains the MATLAB code that extracts statistical features from two-channel I/Q WAV files. Keep `iq_feature_extraction.m` and `spectral_peak_similarity.m` in the same directory. No raw recordings are included. The benchmark feature CSVs are separate inputs to the detection code; this extractor does not reproduce their label assignment, feature replacement, or relative-time preprocessing.

## Requirements

- MATLAB. GPU mode is the default and uses `gpuArray` for the FFT and feature calculations. It requires Parallel Computing Toolbox, a supported NVIDIA GPU and driver, and enough GPU memory for the full segments. Check `gpuDevice` in MATLAB before running and see the [MathWorks GPU computing requirements](https://www.mathworks.com/help/parallel-computing/gpu-computing-requirements.html) for your release.
- Signal Processing Toolbox for `findpeaks` when computing the optional peak similarity. That step runs on the CPU even in GPU mode. It can also require substantial host memory.
- For a short CPU check without either toolbox, set both `UseGPU` and `ComputePeakSimilarity` to `false`. A full-length segment contains about 112 million complex samples in the case-study setting, so a CPU run is not a low-memory substitute.

## Input files and timestamps

Place the WAV files for one monitoring sequence in one directory. Each WAV must have exactly two channels: column 1 is the in-phase component I, and column 2 is the quadrature component Q. The complex samples are formed as `I + 1i*Q`. The sample rate is read from the WAV header and must be the same across the files in one run. There is no resampling.

By default, characters 25 through 32 of the filename stem, counted from 1, must contain the eight digits `MMDDHHMM`. For example, a stem consisting of a 24-character prefix followed by `08140930` represents August 14 at 09:30. The `time` field stores the token as a number, so this example is stored as `8140930`. The token has no year or time zone. If the same eight digits start elsewhere, set `TimestampRange` to their positions. For example, `TimestampRange`, `[1 8]` works with `08140930_example.wav`.

The default processing order is the order returned by MATLAB's `dir`, as in the original script. Use one sequence per run and check that this order is chronological. `SortByTimestamp` optionally sorts by the numeric timestamp, which can change the pairs used for the cross-observation features. It is not suitable for sorting a sequence across a year boundary. Missing monitoring epochs are not inserted; consecutive retained WAVs are compared even if an acquisition gap lies between them.

## Run the extractor

From the repository root in MATLAB, update the two example paths for your input directory and output MAT file:

```matlab
addpath('feature_extraction');
feature = iq_feature_extraction('D:\example\iq_wav', 'D:\example\results\features.mat');
```

To run on a short CPU sample with a different timestamp position and no optional peak similarity:

```matlab
feature = iq_feature_extraction('D:\example\short_wav', 'D:\example\results\short_features.mat', ...
    'UseGPU', false, 'ComputePeakSimilarity', false, 'TimestampRange', [1 8]);
```

The output directory is created if necessary. An existing MAT file is not overwritten unless `Overwrite` is explicitly set to `true`.

| Option | Default | Meaning |
| --- | --- | --- |
| `TimestampRange` | `[25 32]` | Inclusive positions of the eight timestamp digits in the filename stem |
| `SortByTimestamp` | `false` | Sort observations by the numeric timestamp before computing features |
| `UseGPU` | `true` | Run FFT and statistical calculations on the GPU |
| `ComputePeakSimilarity` | `true` | Also compute the optional peak similarity for pairs of WAV files |
| `PeakProminence` | `10000` | `MinPeakProminence` for the optional peak similarity |
| `PeakSigmaHz` | `5` | Gaussian frequency-matching width in Hz for the optional peak similarity |
| `ThresholdFactor` | `3` | Factor multiplying the mean spectrum in the `exceeding_count` feature |
| `InitialReferenceLength` | `112000000` | Length of the original all-ones reference for the first observation |
| `Overwrite` | `false` | Allow replacement of the named output MAT file |

## Output and compatibility

The MAT file contains the struct array `feature`, with one entry per WAV. The original field names are retained:

- Timestamp: `time`.
- Nine statistical features: `mean_spectrum`, `var_spectrum`, `enp_spectrum`, `mean_IQ`, `var_IQ`, `KL_spectrum`, `RMSE_spectrum`, `CS_spectrum`, `exceeding_count`.
- Optional diagnostic: `peakSimilar_spectrum`. This is not one of the nine statistical inputs in the benchmark CSV. It is empty for the first observation, and `NaN` for a pair in which either spectrum has no qualifying peaks. The field is omitted when this calculation is disabled or when there is only one WAV.

The code preserves the original full, unwindowed FFT magnitude and MATLAB double-precision `eps`. For different FFT lengths, it compares leading bins up to the shorter length; spectral mean, variance, and entropy are still computed from the full current spectrum. `exceeding_count` is the percentage of compared bins above the threshold based on that full-spectrum mean. The historical `RMSE_spectrum` field keeps the source expression `sqrt(mean(currentSpectrum - previousSpectrum).^2)` on the compared bins. This is the absolute mean difference, not the conventional per-bin RMSE. It is left unchanged for compatibility with the original extraction code.

## Smoke test

From the repository root, run `addpath('feature_extraction'); smoke_test_iq_feature_extraction` in MATLAB. The test uses short synthetic two-channel WAVs, checks timestamp parsing and the saved features against the legacy formulas, and needs no raw monitoring data. When Signal Processing Toolbox is installed, it also checks the optional peak-matching helper.
