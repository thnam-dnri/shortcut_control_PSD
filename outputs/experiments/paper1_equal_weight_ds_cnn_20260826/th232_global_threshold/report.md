# Th-232 usable-peak global-threshold re-optimization

## Decision

- The previous `0.4370` threshold is superseded because its objective used the unreliable 338.3-keV region and unsupported 2614.5-keV P/B region.
- The new objective uses ten audited P/B anchors from 209.3 through 1460.8 keV.
- The 338.3- and 969-keV regions, the 2614-keV single- and double-escape regions, and the 2614.5-keV full-energy line are excluded from selection.
- The 583.2- and 911.2-keV anchors use only their higher-energy sideband.
- Re-optimized sweet-spot global threshold: `0.4590`.
- No fixed retention floor is used. The threshold is the normalized Pareto knee between geometric-mean P/B gain and worst-peak retention loss.
- Geometric-mean P/B improvement over the ten usable peaks: `1.2229x`.
- Minimum/mean usable-peak net retention: `77.08%` / `78.72%`.
- Total admitted-event retention: `65.72%`.

## FPGA advisory presets

The deployed comparator threshold remains user-adjustable; these are recommended starting presets, not hard-coded limits.

| Preset | Threshold | Geometric-mean P/B | Pooled peak retention | Worst-peak retention | Total event retention |
|:---|---:|---:|---:|---:|---:|
| Sweet spot | 0.4590 | 1.2229x | 79.29% | 77.08% | 65.72% |
| No-brainer conservative | 0.2680 | 1.0446x | 99.01% | 97.74% | 96.33% |

The no-brainer preset is the highest-P/B threshold with pooled net counts across the ten usable peaks retained at >=99%. This is a pooled criterion, not a claim that every individual peak exceeds 99% retention.
Machine-readable preset metadata is in `fpga_recommended_threshold_presets.csv`.

## Stability against the weak 209-keV anchor

Excluding 209.3 keV and reselecting the Pareto knee gives `0.4400`. Its nine-peak geometric-mean P/B improvement is `1.2171x` with minimum retention `80.46%`.

## Comparison with the superseded threshold

| Operating point | Threshold | Geometric-mean P/B | Minimum retention | Mean retention | Total event retention |
|:---|---:|---:|---:|---:|---:|
| Superseded global | 0.4370 | 1.1949x | 80.42% | 82.94% | 71.66% |
| Re-optimized global | 0.4590 | 1.2229x | 77.08% | 78.72% | 65.72% |
| Maximum reliable P/B | 0.5035 | 1.2761x | 55.96% | 67.46% | 52.54% |

## Re-optimized per-peak result

| Energy (keV) | Anchor | Sideband | P/B improvement | Net retention |
|---:|:---|:---|---:|---:|
| 209.253 | 209.3 keV Ac-228 | both | 1.0638x | 77.21% |
| 238.632 | 238.6 keV Pb-212 | both | 1.0527x | 78.77% |
| 300.087 | 300.1 keV Pb-212 | both | 1.1264x | 77.61% |
| 409.462 | 409.5 keV Ac-228 | both | 1.2467x | 79.08% |
| 510.770 | 510.8 keV Tl-208/pair | both | 1.2569x | 77.08% |
| 583.191 | 583.2 keV Tl-208 | higher_only | 1.2900x | 78.47% |
| 727.330 | 727.3 keV Bi-212 | both | 1.3086x | 81.06% |
| 911.204 | 911.2 keV Ac-228 | higher_only | 1.3154x | 82.22% |
| 1247.080 | 1247.1 keV Ac-228 | both | 1.3244x | 77.23% |
| 1460.830 | 1460.8 keV K-40/background | both | 1.2868x | 78.51% |

## Full-spectrum artifacts

- Linear/log comparison: `th232_spectrum_threshold_0.4590` with the decimal point encoded as `p` in the filename.
- Reproducible 1-keV no-cut and selected counts use the same threshold-coded filename for `0.4590`.
- Equal-net-peak-area P/B comparison: `usable_peak_zooms_normalized.png`.

## Claim boundary

The checkpoint recorded in the input score cache is unchanged during threshold selection. Historical Th-232 events are used directly to select this deployment threshold, so this is in-sample threshold optimization, not external validation. Locked test and Eu-152 remain unopened and unused.
