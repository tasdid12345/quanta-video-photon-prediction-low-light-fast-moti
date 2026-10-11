# Quanta video photon prediction (low light, fast motion)

Question: can a motion-compensated, Bernoulli-photon-statistics model reconstruct simulated 1-bit quanta video better than a bit2bit-style self-supervised photon-prediction baseline in very dark, fast-motion conditions?

- Data: Olivetti faces, turned into constant-velocity translation videos and sampled as 1-bit Bernoulli frames (flux 0.05/0.2, speed 0.5/3.0 px/frame). Train/test split by subject (30 / 10 subjects), seed 42.
- Baseline: sklearn MLPRegressor on a 5-frame x 5x5 bit patch with the centre pixel masked.
- Improved: phase-correlation motion estimate from bits, frame alignment, LightGBM on multi-scale aligned photon-count features, then Bernoulli ML flux inversion.
- Hyper-parameters are chosen only by subject-grouped cross-validation on the training set (held-out bit Brier score).
- Metric: PSNR (dB) on the darkest, fastest test setting. Run `python analysis.py` to produce `results.json` and `figures/psnr.png`.

Fix note: `MLPRegressor` hyper-parameters are now passed by keyword (newer scikit-learn puts `loss` first positionally), and the bit labels are cast to int.

---
_The code in this repository was drafted with AI assistance and runs automatically on GitHub Actions; results come from the latest run._
