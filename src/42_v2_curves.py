import warnings

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import (average_precision_score, precision_recall_curve,
                             roc_auc_score, roc_curve)

SEED, N_BOOT = 42, 2000
R = "outputs/reports/"
folds = pd.read_csv("manifests/folds_v2.csv", dtype={"case_id": str, "group": str})
classes = sorted(folds.label.unique())
K = len(classes)
y = folds.label.values
Y = np.column_stack([(y == c).astype(int) for c in classes])
prev = Y.mean(0)

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


def per_class(P, idx):
    auc, ap = np.full(K, np.nan), np.full(K, np.nan)
    for k in range(K):
        t = Y[idx, k]
        if 0 < t.sum() < len(t):
            auc[k] = roc_auc_score(t, P[idx, k])
            ap[k] = average_precision_score(t, P[idx, k])
    return auc, ap


rng = np.random.default_rng(SEED)
grp = {g: np.where(folds.group.values == g)[0] for g in folds.group.unique()}
gl = list(grp)
boots = [np.concatenate([grp[g] for g in rng.choice(gl, len(gl))]) for _ in range(N_BOOT)]
allidx = np.arange(len(y))

res, macro_boot = {}, {}
with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    for name, f in MODELS.items():
        P = load(f)
        a0, p0 = per_class(P, allidx)
        BA = np.array([per_class(P, b) for b in boots])  # (N_BOOT, 2, K)
        auc_b, ap_b = BA[:, 0, :], BA[:, 1, :]
        mb = np.nanmean(auc_b, axis=1)
        macro_boot[name] = mb
        res[name] = (P, a0, p0, auc_b, ap_b, mb)
        print(f"\n{name}")
        print(f"  macro ROC-AUC {np.nanmean(a0):.3f} [{np.percentile(mb, 2.5):.3f}, {np.percentile(mb, 97.5):.3f}]"
              f" | macro AP {np.nanmean(p0):.3f} (chance reference about {prev.mean():.3f})")
        for k, c in enumerate(classes):
            la, ha = np.nanpercentile(auc_b[:, k], [2.5, 97.5])
            lp, hp = np.nanpercentile(ap_b[:, k], [2.5, 97.5])
            flag = "AUC CI excludes 0.5" if la > 0.5 else "AUC CI includes 0.5"
            print(f"  {c:30s} AUC {a0[k]:.3f} [{la:.3f}, {ha:.3f}] | AP {p0[k]:.3f} [{lp:.3f}, {hp:.3f}] "
                  f"vs prevalence {prev[k]:.3f} | {flag}")

ft = "Fine-tuned MobileNetV3-S (weighted)"
print("\npaired macro-AUC difference, fine-tuned minus each frozen model (same resampled groups):")
for name in MODELS:
    if name == ft:
        continue
    d = macro_boot[ft] - macro_boot[name]
    pt = np.nanmean(res[ft][1]) - np.nanmean(res[name][1])
    lo, hi = np.percentile(d, [2.5, 97.5])
    print(f"  vs {name:36s} {pt:+.3f} [{lo:+.3f}, {hi:+.3f}] {'CI excludes 0' if lo > 0 or hi < 0 else 'CI includes 0'}")

rows = []
for name, (P, a0, p0, auc_b, ap_b, mb) in res.items():
    rows.append({"model": name, "class": "MACRO", "auc": np.nanmean(a0),
                 "auc_lo": np.percentile(mb, 2.5), "auc_hi": np.percentile(mb, 97.5), "ap": np.nanmean(p0)})
    for k, c in enumerate(classes):
        rows.append({"model": name, "class": c, "auc": a0[k],
                     "auc_lo": np.nanpercentile(auc_b[:, k], 2.5), "auc_hi": np.nanpercentile(auc_b[:, k], 97.5),
                     "ap": p0[k], "ap_lo": np.nanpercentile(ap_b[:, k], 2.5),
                     "ap_hi": np.nanpercentile(ap_b[:, k], 97.5), "prevalence": prev[k]})
pd.DataFrame(rows).round(3).to_csv(R + "curves_v2.csv", index=False)

P = res[ft][0]
fig, ax = plt.subplots(figsize=(6.5, 6))
for k, c in enumerate(classes):
    fpr, tpr, _ = roc_curve(Y[:, k], P[:, k])
    ax.plot(fpr, tpr, label=f"{c} (AUC {res[ft][1][k]:.2f})")
ax.plot([0, 1], [0, 1], "--", color="gray")
ax.set_xlabel("false positive rate")
ax.set_ylabel("true positive rate")
ax.set_title("One-vs-rest ROC, fine-tuned MobileNetV3-S (out-of-fold)")
ax.legend(fontsize=7)
plt.tight_layout()
plt.savefig("outputs/figures/roc_v2_finetuned.png", dpi=150)

fig, axes = plt.subplots(2, 4, figsize=(15, 7))
for k, (c, ax) in enumerate(zip(classes, axes.ravel())):
    pr, rc, _ = precision_recall_curve(Y[:, k], P[:, k])
    ax.plot(rc, pr)
    ax.axhline(prev[k], ls="--", color="gray")
    ax.set_title(f"{c}\nAP {res[ft][2][k]:.2f} vs prevalence {prev[k]:.2f}", fontsize=8)
    ax.set_xlabel("recall", fontsize=7)
    ax.set_ylabel("precision", fontsize=7)
    ax.set_ylim(0, 1)
plt.tight_layout()
plt.savefig("outputs/figures/pr_v2_finetuned.png", dpi=150)

pred = np.array(classes)[P.argmax(1)]
cm = pd.crosstab(pd.Series(y, name="true"), pd.Series(pred, name="pred")).reindex(
    index=classes, columns=classes, fill_value=0)
rn = cm.div(cm.sum(1), axis=0).values
fig, ax = plt.subplots(figsize=(8, 6.5))
im = ax.imshow(rn, vmin=0, vmax=0.6)
ax.set_xticks(range(K))
ax.set_yticks(range(K))
ax.set_xticklabels(classes, rotation=45, ha="right", fontsize=7)
ax.set_yticklabels(classes, fontsize=7)
for i in range(K):
    for j in range(K):
        ax.text(j, i, f"{rn[i, j]:.2f}\n({cm.values[i, j]})", ha="center", va="center", fontsize=6,
                color="white" if rn[i, j] < 0.3 else "black")
ax.set_xlabel("predicted")
ax.set_ylabel("true")
ax.set_title("Row-normalised confusion matrix (count), fine-tuned MobileNetV3-S")
plt.colorbar(im, fraction=0.04)
plt.tight_layout()
plt.savefig("outputs/figures/confusion_v2_finetuned.png", dpi=150)

names = list(MODELS)
m0 = [np.nanmean(res[n][1]) for n in names]
lo = [np.percentile(res[n][5], 2.5) for n in names]
hi = [np.percentile(res[n][5], 97.5) for n in names]
fig, ax = plt.subplots(figsize=(8, 4.2))
yy = np.arange(len(names))[::-1]
ax.errorbar(m0, yy, xerr=[np.array(m0) - lo, np.array(hi) - m0], fmt="o", capsize=3)
ax.axvline(0.5, ls="--", color="gray")
ax.set_yticks(yy)
ax.set_yticklabels(names, fontsize=8)
ax.set_xlabel("macro one-vs-rest ROC-AUC (95% group-bootstrap CI)")
ax.set_title("SCIN v2: ranking performance, out-of-fold")
plt.tight_layout()
plt.savefig("outputs/figures/macro_auc_v2.png", dpi=150)
print("\nsaved outputs/reports/curves_v2.csv and 4 figures in outputs/figures/")