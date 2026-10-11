import warnings

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

SEED, N_BOOT, NB = 42, 2000, 10
FRACS = (1.0, 0.8, 0.6, 0.4, 0.2)
R = "outputs/reports/"
folds = pd.read_csv("manifests/folds_v2.csv", dtype={"case_id": str, "group": str})
classes = sorted(folds.label.unique())
y = folds.label.values
yi = np.array([classes.index(c) for c in y])
n = len(y)

MODELS = {
    "Fine-tuned MobileNetV3-S (weighted)": "oof_v2_finetune_mobilenet.csv",
    "Frozen MobileNetV3-S": "oof_v2_frozen_mobilenet_v3_small.csv",
    "Frozen ResNet-18": "oof_v2_frozen_resnet18.csv",
    "Frozen EfficientNet-B0": "oof_v2_frozen_efficientnet_b0.csv",
    "Frozen MobileNetV3-S (weighted)": "oof_v2_frozenbal_mobilenet_v3_small.csv",
    "Frozen ResNet-18 (weighted)": "oof_v2_frozenbal_resnet18.csv",
    "Frozen EfficientNet-B0 (weighted)": "oof_v2_frozenbal_efficientnet_b0.csv",
}


def load(name):
    o = pd.read_csv(R + name, dtype={"case_id": str}).set_index("case_id").loc[folds.case_id]
    assert (o["true"].values == y).all()
    P = o[classes].values
    assert np.allclose(P.sum(1), 1, atol=1e-4)
    return P


def ece(conf, ok):
    b = np.minimum((conf * NB).astype(int), NB - 1)
    return sum(abs(ok[b == k].mean() - conf[b == k].mean()) * (b == k).sum()
               for k in range(NB) if (b == k).any()) / len(conf)


def metrics(P, idx):
    p, t = P[idx], yi[idx]
    conf, ok = p.max(1), (p.argmax(1) == t).astype(float)
    brier = ((p - np.eye(len(classes))[t]) ** 2).sum(1).mean()
    nll = -np.log(np.clip(p[np.arange(len(t)), t], 1e-12, None)).mean()
    return ece(conf, ok), brier, nll


def selective(P, idx):
    p, t = P[idx], yi[idx]
    order = np.argsort(-p.max(1), kind="stable")
    out = []
    for f in FRACS:
        k = max(1, int(round(f * len(idx))))
        keep = order[:k]
        acc = (p[keep].argmax(1) == t[keep]).mean()
        share = np.bincount(t[keep], minlength=len(classes)).max() / k
        out.append((acc, share))
    return out


rng = np.random.default_rng(SEED)
grp = {g: np.where(folds.group.values == g)[0] for g in folds.group.unique()}
gl = list(grp)
boots = [np.concatenate([grp[g] for g in rng.choice(gl, len(gl))]) for _ in range(N_BOOT)]
allidx = np.arange(n)

rows, fig, axes = [], *plt.subplots(2, 4, figsize=(15, 7))
axes = axes.ravel()
for ax_i, (name, f) in enumerate(MODELS.items()):
    P = load(f)
    pt = metrics(P, allidx)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        Bm = np.array([metrics(P, b) for b in boots])
        Bs = np.array([[a for a, _ in selective(P, b)] for b in boots])
    lo, hi = np.percentile(Bm, [2.5, 97.5], axis=0)
    slo, shi = np.percentile(Bs, [2.5, 97.5], axis=0)
    sel = selective(P, allidx)
    conf = P.max(1)
    print(f"\n{name}: mean confidence {conf.mean():.3f} | top-1 accuracy {(P.argmax(1) == yi).mean():.3f}")
    print(f"  ECE {pt[0]:.3f} [{lo[0]:.3f}, {hi[0]:.3f}] | Brier {pt[1]:.3f} [{lo[1]:.3f}, {hi[1]:.3f}]"
          f" | NLL {pt[2]:.3f} [{lo[2]:.3f}, {hi[2]:.3f}]")
    for k, fr in enumerate(FRACS):
        print(f"  keep top {int(fr * 100):3d}%: accuracy {sel[k][0]:.3f} [{slo[k]:.3f}, {shi[k]:.3f}]"
              f" | largest-class share of kept set {sel[k][1]:.3f}")
        rows.append({"model": name, "keep_pct": int(fr * 100), "accuracy": round(sel[k][0], 3),
                     "acc_lo": round(slo[k], 3), "acc_hi": round(shi[k], 3),
                     "largest_class_share": round(sel[k][1], 3),
                     "ECE": round(pt[0], 3), "Brier": round(pt[1], 3), "NLL": round(pt[2], 3)})
    ok = (P.argmax(1) == yi).astype(float)
    b = np.minimum((conf * NB).astype(int), NB - 1)
    xs = [conf[b == k].mean() for k in range(NB) if (b == k).any()]
    ys = [ok[b == k].mean() for k in range(NB) if (b == k).any()]
    ns = [(b == k).sum() for k in range(NB) if (b == k).any()]
    ax = axes[ax_i]
    ax.plot([0, 1], [0, 1], "--", color="gray")
    ax.scatter(xs, ys, s=[10 + v / 4 for v in ns])
    ax.set_title(name, fontsize=8)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_xlabel("mean top-label probability", fontsize=7)
    ax.set_ylabel("accuracy in bin", fontsize=7)
axes[-1].axis("off")
plt.tight_layout()
plt.savefig("outputs/figures/reliability_v2.png", dpi=150)
pd.DataFrame(rows).to_csv(R + "calibration_selective_v2.csv", index=False)
print("\nsaved outputs/figures/reliability_v2.png and outputs/reports/calibration_selective_v2.csv")