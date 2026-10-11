# Motion-compensated Bernoulli model vs bit2bit-style baseline (simulated 1-bit quanta video)

Question: can a motion-compensated, Bernoulli-photon-statistics model reconstruct 1-bit quanta video better than a bit2bit-style self-supervised photon-prediction baseline in very dark, fast-motion conditions?

Data: Olivetti faces turned into constant-velocity translation videos, then sampled as Bernoulli 1-bit frames (seed 42). Train and test are split by subject (30 train / 10 test subjects).

Baseline: sklearn MLP on a 5-frame x 5x5 bit patch with the centre pixel masked.

Improved: phase-correlation motion estimate from the bits only, frame alignment, then LightGBM on multi-scale aligned and unaligned photon-count features (centre bit excluded, using exact float64 neighbour counts), pixel position and estimated velocity. It uses the same self-supervised bit target, followed by Bernoulli ML flux inversion. Hyper-parameters are chosen by subject-grouped cross-validation on training data only.

Run: `pip install -r requirements.txt && python analysis.py`. Outputs: `results.json`, `figures/psnr.png`. CPU only, about a few minutes.

---
_The code in this repository was drafted with AI assistance and runs automatically on GitHub Actions; results come from the latest run._
