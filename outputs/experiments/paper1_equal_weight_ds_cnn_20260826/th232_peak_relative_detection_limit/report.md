# Th-232 peak-specific relative detection-limit optimization

## Decision

Each audited spectral line is optimized independently by minimizing a finite-count relative detection limit. Lower values are better; 1.0 is the unfiltered spectrum. The objective includes both sideband-estimation variance and net-photopeak retention.

| Energy (keV) | Anchor | Threshold | Relative DL | Change | Peak retained | P/B gain | Bootstrap supports improvement | Bootstrap threshold 95% interval |
|---:|:---|---:|---:|---:|---:|---:|:---:|:---|
| 209.253 | 209.3 keV Ac-228 | 0.2805 | 0.9920 | -0.80% | 100.00% | 1.0163x | no | [0.0875, 0.4120] |
| 238.632 | 238.6 keV Pb-212 | 0.2470 | 0.9944 | -0.56% | 99.97% | 1.0115x | yes | [0.2250, 0.3235] |
| 300.087 | 300.1 keV Pb-212 | 0.2940 | 0.9875 | -1.25% | 99.99% | 1.0257x | yes | [0.2805, 0.3633] |
| 409.462 | 409.5 keV Ac-228 | 0.3435 | 0.9711 | -2.89% | 97.03% | 1.0937x | yes | [0.3185, 0.4655] |
| 510.770 | 510.8 keV Tl-208/pair | 0.3230 | 0.9859 | -1.41% | 95.30% | 1.0804x | yes | [0.2905, 0.3820] |
| 583.191 | 583.2 keV Tl-208 | 0.3165 | 0.9739 | -2.61% | 95.79% | 1.1015x | yes | [0.3005, 0.4100] |
| 727.330 | 727.3 keV Bi-212 | 0.4025 | 0.9637 | -3.63% | 88.44% | 1.2202x | yes | [0.3250, 0.4525] |
| 911.204 | 911.2 keV Ac-228 | 0.4180 | 0.9467 | -5.33% | 87.33% | 1.2828x | yes | [0.3975, 0.4840] |
| 1247.080 | 1247.1 keV Ac-228 | 0.2810 | 0.9566 | -4.34% | 97.66% | 1.1210x | yes | [0.2467, 0.4950] |
| 1460.830 | 1460.8 keV K-40/background | 0.2440 | 0.9726 | -2.74% | 99.44% | 1.0643x | yes | [0.2345, 0.4703] |

## Statistical definition

For net counts `N_ROI - B_hat`, the null variance is approximated as `B_hat + Var(B_hat)`, where `Var(B_hat)` is propagated from Poisson sideband counts and their ROI scaling/interpolation coefficients. With one-sided alpha=beta=0.05, the detection limit solves `Ld = Lc + k_beta sqrt(Ld + B_hat + Var(B_hat))`. The reported objective is `(Ld_cut / peak_retention) / Ld_no_cut`. Because a cut cannot physically retain more than all signal events, net-area retention estimates slightly above 1.0 are conservatively capped at 1.0 only in this denominator; the uncapped estimate remains in the CSV and JSON.

The uncertainty intervals use a 30-file cluster bootstrap. They describe acquisition-segment stability and are not an external-validation interval.

## Scope and claim boundary

This is a relative, spectrum-conditioned detection limit, not an absolute minimum detectable activity in Bq. Absolute MDA additionally requires calibrated full-energy efficiency, emission probability, live time/dead-time correction, source geometry, and attenuation corrections. The 1460.8-keV anchor is K-40/background and is not a Th-232-chain MDA claim.

Historical Th-232 events directly select these thresholds. Locked test and Eu-152 remain unopened and unused.
