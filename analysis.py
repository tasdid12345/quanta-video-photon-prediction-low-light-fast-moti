"""Motion-compensated Bernoulli model vs bit2bit-style baseline on simulated 1-bit quanta video (Olivetti faces)."""
import json, os, sys, time, warnings
import numpy as np
from numpy.lib.stride_tricks import sliding_window_view as swv
from scipy import ndimage as ndi
from sklearn.datasets import fetch_olivetti_faces
from sklearn.model_selection import GroupKFold
from sklearn.neural_network import MLPRegressor
from lightgbm import LGBMClassifier
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore")
SEED, T, T0, H, N_PER = 42, 9, 4, 64, 600      # frames per video, reference frame, image size, samples/video
PEAKS, SPEEDS = [0.05, 0.2], [0.5, 3.0]        # peak photons/pixel/frame, px/frame
MAIN = (min(PEAKS), max(SPEEDS))               # darkest + fastest
WIN = np.outer(np.hanning(H), np.hanning(H))
t_start = time.time()

# ---------- data ----------
for k in range(4):                              # retry download with backoff
    try:
        faces = fetch_olivetti_faces(random_state=SEED); break
    except Exception as e:
        print("download failed:", e); time.sleep(2 ** k)
else:
    sys.exit("could not fetch Olivetti faces")
imgs, subj = faces.images.astype(np.float32), faces.target
test_subj = np.random.default_rng(SEED).permutation(np.unique(subj))[:10]   # single split by subject (25%)
is_te = np.isin(subj, test_subj); tr, te = np.where(~is_te)[0], np.where(is_te)[0]
print(f"{len(tr)} train / {len(te)} test videos per setting")

def make_videos(peak, speed, rng):
    """Constant-velocity translation (random direction), then Bernoulli bits p=1-exp(-flux)."""
    ang = rng.uniform(0, 2 * np.pi, len(imgs)); vel = speed * np.stack([np.sin(ang), np.cos(ang)], 1)
    clean = np.stack([np.stack([ndi.shift(im, (t - T0) * v, order=1, mode="nearest") for t in range(T)])
                      for im, v in zip(imgs, vel)])
    return (rng.random(clean.shape) < 1 - np.exp(-peak * clean)).astype(np.float32), vel

# ---------- motion estimation (bits only) ----------
def phasecorr(a, b):
    """Sub-pixel shift d with b(x)=a(x-d), via regularised phase correlation."""
    F = lambda z: np.fft.fft2((ndi.gaussian_filter(z, 2) - z.mean()) * WIN)
    P = F(b) * np.conj(F(a)); r = np.fft.ifft2(P / (np.abs(P) + np.abs(P).mean())).real
    iy, ix = np.unravel_index(r.argmax(), r.shape); d = []
    for ax, i in ((0, iy), (1, ix)):
        l, c, u = (r[(i + k) % H, ix] if ax == 0 else r[iy, (i + k) % H] for k in (-1, 0, 1))
        o = np.clip(0.5 * (l - u) / (l - 2 * c + u - 1e-12), -0.5, 0.5); i = i + o
        d.append(i - H if i > H / 2 else i)
    return np.array(d)

def est_vel(b):
    """Least-squares velocity from shifts between 3-frame count sums (centres 1, 4, 7)."""
    G = [b[0:3].sum(0), b[3:6].sum(0), b[6:9].sum(0)]
    return sum(dt * phasecorr(G[i], G[j]) for i, j, dt in ((0, 1, 3), (1, 2, 3), (0, 2, 6))) / 54.0

# ---------- features ----------
def patches(b):  # baseline: 5 frames x 5x5 patch around each pixel
    return swv(np.pad(b, ((0, 0), (2, 2), (2, 2)), mode="reflect"), (5, 5, 5), axis=(0, 1, 2))

def feats(b, v, t):
    """Improved: align all other frames to frame t, multi-scale photon-count features; centre bit never used."""
    us = [u for u in range(T) if u != t]
    al = [ndi.shift(b[u], -(u - t) * v, order=1, mode="nearest") for u in us]
    allm, near = np.mean(al, 0), np.mean([a for a, u in zip(al, us) if abs(u - t) <= 2], 0)
    box = lambda z, k: ndi.uniform_filter(z, k, mode="mirror")
    f = [allm] + [box(allm, k) for k in (3, 5, 9, 17)] + [near, box(near, 3), box(near, 5)]
    f += [(box(b[t], k) * k * k - b[t]) / (k * k - 1) for k in (3, 5, 9)]   # same-frame neighbours, centre excluded
    return np.stack(f, -1).astype(np.float32)

# ---------- metrics ----------
def to_flux(p): return -np.log1p(-np.clip(p, 0, 0.999))   # Bernoulli ML inversion: lambda = -log(1-p)
def psnr(e, c): return 10 * np.log10(1 / np.maximum(((e - c) ** 2).mean((1, 2)), 1e-10))
def ssim(e, c, k=7):
    u = lambda z: ndi.uniform_filter(z, (1, k, k)); ma, mb = u(e), u(c)
    va, vb, cv = u(e * e) - ma ** 2, u(c * c) - mb ** 2, u(e * c) - ma * mb
    s = ((2 * ma * mb + 1e-4) * (2 * cv + 9e-4)) / ((ma ** 2 + mb ** 2 + 1e-4) * (va + vb + 9e-4))
    return s.mean((1, 2))

# ---------- label-free CV (held-out bit Brier score, grouped by subject) ----------
def make_mlp(p): return MLPRegressor(p[0], alpha=p[1], batch_size=512, max_iter=25, early_stopping=True,
                                     n_iter_no_change=3, random_state=SEED)
def make_gbm(p): return LGBMClassifier(n_estimators=250, learning_rate=0.05, num_leaves=p, min_child_samples=100,
                                       subsample=0.8, subsample_freq=1, random_state=SEED, n_jobs=4, verbose=-1)
def predict(m, X): return m.predict_proba(X)[:, 1] if hasattr(m, "predict_proba") else np.clip(m.predict(X), 0, 1)
def cv_select(make, grid, X, y, g):
    sc = [np.mean([np.mean((y[va] - predict(make(p).fit(X[a], y[a]), X[va])) ** 2)
                   for a, va in GroupKFold(3).split(X, y, g)]) for p in grid]
    print("  CV Brier:", dict(zip(map(str, grid), np.round(sc, 6)))); return grid[int(np.argmin(sc))]

hp, res = None, {}
settings = sorted([(p, s) for p in PEAKS for s in SPEEDS], key=lambda ps: ps != MAIN)   # main setting first
for peak, speed in settings:
    print(f"setting flux={peak} speed={speed} ({time.time() - t_start:.0f}s)")
    rng = np.random.default_rng(SEED + int(1000 * peak) + int(10 * speed))
    bits, vel = make_videos(peak, speed, rng)
    vh = np.array([est_vel(b) for b in bits])                  # motion from binary frames only
    Xb, Xi, y, g = [], [], [], []
    for i in tr:                                                # self-supervised samples: (frame, pixel) -> its own bit
        t = np.sort(rng.integers(2, 7, N_PER)); ys, xs = rng.integers(0, H, N_PER), rng.integers(0, H, N_PER)
        Xb.append(patches(bits[i])[t - 2, ys, xs].reshape(N_PER, 125))
        Xi.append(np.concatenate([feats(bits[i], vh[i], tt)[ys[t == tt], xs[t == tt]] for tt in np.unique(t)]))
        y.append(bits[i][t, ys, xs]); g.append(np.full(N_PER, subj[i]))
    Xb, Xi, y, g = np.concatenate(Xb), np.concatenate(Xi), np.concatenate(y), np.concatenate(g)
    Xb[:, 62] = 0                                               # mask the centre pixel (the target bit)
    if hp is None:                                              # CV on training data of the main setting only
        sub = rng.choice(len(y), 50000, replace=False)
        hp = (cv_select(make_mlp, [((32,), 1e-4), ((64, 32), 1e-3)], Xb[sub], y[sub], g[sub]),
              cv_select(make_gbm, [15, 63], Xi[sub], y[sub], g[sub]))
        print("  selected:", hp)
    mlp, gbm = make_mlp(hp[0]).fit(Xb, y), make_gbm(hp[1]).fit(Xi, y)
    XbT = np.concatenate([np.array(patches(bits[i])[T0 - 2]).reshape(-1, 125) for i in te]); XbT[:, 62] = 0
    XiT = np.concatenate([feats(bits[i], vh[i], T0).reshape(-1, Xi.shape[1]) for i in te])
    clean = imgs[te]
    out = {}
    for name, p in (("baseline", predict(mlp, XbT)), ("improved", predict(gbm, XiT))):
        est = np.clip(to_flux(p).reshape(-1, H, H) / peak, 0, 1)   # flux normalised by peak flux -> [0,1]
        out[name] = (psnr(est, clean).mean(), ssim(est, clean).mean())
    out["motion_err"] = np.abs(vh[te] - vel[te]).mean()
    res[(peak, speed)] = out
    print("  ", {k: np.round(v, 3).tolist() for k, v in out.items()})

# ---------- results ----------
m = {"baseline_score": res[MAIN]["baseline"][0], "improved_score": res[MAIN]["improved"][0],
     "baseline_ssim": res[MAIN]["baseline"][1], "improved_ssim": res[MAIN]["improved"][1],
     "baseline_psnr_all": np.mean([r["baseline"][0] for r in res.values()]),
     "improved_psnr_all": np.mean([r["improved"][0] for r in res.values()]),
     "motion_error_px_per_frame_main": res[MAIN]["motion_err"]}
m["improvement"] = m["improved_score"] - m["baseline_score"]
for (p, s), r in res.items():
    m[f"baseline_psnr_flux{p}_speed{s}"], m[f"improved_psnr_flux{p}_speed{s}"] = r["baseline"][0], r["improved"][0]
m = {k: float(v) for k, v in m.items()}
if not all(np.isfinite(list(m.values()))): sys.exit("non-finite metrics")

os.makedirs("figures", exist_ok=True)
lab = [f"flux {p}\nspeed {s}" for p, s in res]; x = np.arange(len(lab))
plt.bar(x - .2, [r["baseline"][0] for r in res.values()], .4, label="bit2bit-style MLP")
plt.bar(x + .2, [r["improved"][0] for r in res.values()], .4, label="aligned LightGBM + Bernoulli MLE")
plt.xticks(x, lab); plt.ylabel("test PSNR (dB)"); plt.legend(); plt.tight_layout(); plt.savefig("figures/psnr.png", dpi=120)

json.dump({
    "question": "Can a motion-compensated, Bernoulli-photon-statistics model reconstruct 1-bit quanta video better than a bit2bit-style self-supervised photon-prediction baseline in very dark, fast-motion conditions?",
    "method": "Olivetti faces -> constant-velocity translation videos -> Bernoulli 1-bit frames at several flux levels/speeds; train on binary frames of training subjects only (self-supervised bit prediction), test on held-out subjects; PSNR vs clean frame.",
    "dataset": {"name": "Olivetti faces (simulated 1-bit videos)", "source_url": "https://scikit-learn.org/stable/modules/generated/sklearn.datasets.fetch_olivetti_faces.html",
                "n_rows": int(len(imgs)), "n_train": int(len(tr)), "n_test": int(len(te)), "same_as_paper": False},
    "metric_name": "PSNR (dB) on darkest, fastest test setting", "higher_is_better": True,
    "baseline": "bit2bit-style sklearn MLPRegressor on a 5-frame x 5x5 bit patch with the centre pixel masked; predicted photon probability is the reconstruction.",
    "improved": "Phase-correlation motion estimate on short bit-count sums, frame alignment, LightGBM on multi-scale aligned photon-count features with the same self-supervised bit target, then Bernoulli ML flux inversion.",
    "metrics": m,
    "notes": "Small-scale simulation, not the paper's data or network (bit2bit uses a deep 3D model). Separate models per setting; hyper-parameters chosen by subject-grouped CV (held-out bit Brier score) on the main setting's training data. Both models' outputs go through the same -log(1-p) inversion and are normalised by the peak flux, which is assumed known. Global translation only; 'nearest' edge fill. Single seed, one split, so differences may be within noise."},
    open("results.json", "w", encoding="utf-8"), indent=2)
print(f"done in {time.time() - t_start:.0f}s")
