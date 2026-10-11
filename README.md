# Motion-compensated Bernoulli reconstruction of 1-bit quanta video

Single-photon (SPAD / quanta) sensors output 1-bit frames, and in very dark, fast-moving scenes the photon counts are sparse and motion blurs any temporal averaging. The paper "bit2bit: 1-bit quanta video reconstruction via self-supervised photon prediction" (NeurIPS 2024) reconstructs such video by predicting held-out photon detections. This experiment tests whether explicit motion compensation plus Bernoulli photon statistics improves on a bit2bit-style baseline.

## Question
Can a motion-compensated, Bernoulli-photon-statistics model reconstruct 1-bit quanta video better than a bit2bit-style self-supervised photon-prediction baseline in very dark, fast-motion conditions?

## Dataset
Olivetti faces (64x64, via `sklearn.datasets.fetch_olivetti_faces`). This is NOT the paper's dataset. Each image becomes a short constant-velocity translation video, then 1-bit frames with p = 1 - exp(-flux) at several dark light levels and speeds. Test = videos of 10 held-out subjects; training uses only the binary frames of the other subjects (never the clean images).

## Models
- **Baseline:** bit2bit-style sklearn MLP that sees a spatio-temporal patch of neighbouring bit frames with the centre pixel masked and predicts that bit; the predicted photon probability is the reconstruction.
- **Improved:** phase-correlation motion estimation on short bit-count sums, alignment of neighbouring frames, LightGBM on multi-scale aligned photon-count features with the same self-supervised target, then Bernoulli maximum-likelihood flux inversion.

Main metric: PSNR on the darkest, fastest test setting; secondary: SSIM and PSNR averaged over all settings. Hyper-parameters are chosen by subject-grouped cross-validation on training data only.

## How to run
```
pip install -r requirements.txt
python analysis.py
```

## Results
Results are written to `results.json` (and a chart to `figures/`).

---
_The code in this repository was drafted with AI assistance and runs automatically on GitHub Actions; results come from the latest run._
