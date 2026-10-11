# Motion-compensated Bernoulli model vs bit2bit-style baseline (simulated 1-bit quanta video)

Question: can a motion-compensated, Bernoulli-photon-statistics model reconstruct 1-bit quanta video better than a bit2bit-style self-supervised photon-prediction baseline in very dark, fast-motion conditions?

Data: Olivetti faces turned into constant-velocity translation videos, then sampled as Bernoulli 1-bit frames (seed 42). Train and test are split by subject (30 train / 10 test subjects).

Baseline: sklearn MLP on a 5-frame x 5x5 bit patch with the centre pixel masked.

Improved: phase-correlation motion estimate from the bits only, frame alignment, then LightGBM on multi-scale aligned and unaligned photon-count features (centre bit excluded, using exact float64 neighbour counts), pixel position and estimated velocity. It uses the same self-supervised bit target, followed by Bernoulli ML flux inversion. Hyper-parameters are chosen by subject-grouped cross-validation on training data only.

Run: `pip install -r requirements.txt && python analysis.py`. Outputs: `results.json`, `figures/psnr.png`. CPU only, about a few minutes.

---
_The code in this repository was drafted with AI assistance and runs automatically on GitHub Actions; results come from the latest run._

## Results (latest run)

On simulated 1-bit Olivetti-face videos, PSNR in the darkest, fastest setting was 9.78 dB for my re-implementation of your bit2bit-style baseline and 15.08 dB for the motion-compensated Bernoulli model, on the same held-out test subjects from a single seed.

| Metric | Value |
|---|---|
| baseline_score | 9.78 |
| improved_score | 15.08 |
| baseline_ssim | 0.02416 |
| improved_ssim | 0.1249 |
| baseline_psnr_all | 12.34 |
| improved_psnr_all | 16.94 |
| motion_error_px_per_frame_main | 1.941 |
| improvement | 5.302 |
| baseline_psnr_flux0.05_speed3.0 | 9.78 |
| improved_psnr_flux0.05_speed3.0 | 15.08 |
| baseline_psnr_flux0.05_speed0.5 | 9.993 |
| improved_psnr_flux0.05_speed0.5 | 15.63 |
| baseline_psnr_flux0.2_speed0.5 | 14.61 |
| improved_psnr_flux0.2_speed0.5 | 18.92 |
| baseline_psnr_flux0.2_speed3.0 | 14.97 |
| improved_psnr_flux0.2_speed3.0 | 18.13 |

Run on 2026-10-11. Full output: [results.json](results.json) · [run.log](run.log)

Notes: Small-scale simulation, not the paper's data or network (bit2bit uses a deep 3D model). Separate models per setting; hyper-parameters chosen by subject-grouped CV (held-out bit Brier score) on the main setting's training data. Both models' outputs go through the same -log(1-p) inversion and are normalised by the peak flux, which is assumed known. Global translation only; 'nearest' edge fill. A previous run had a leak: float32 rounding residue in the same-frame neighbour features encoded the centre (target) bit; neighbour counts are now computed in float64 with exact integer rounding. Single seed, one split, so differences may be within noise.
