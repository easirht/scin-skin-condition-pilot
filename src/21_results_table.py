import warnings

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score

SEED, N_BOOT = 42, 2000
folds = pd.read_csv("manifests/folds_headneck_v1.csv", dtype={"case_id": str, "group": str})
classes = sorted(folds.label.unique())
y = folds.label.values


def load(name):
    o = pd.read_csv(f"outputs/reports/oof_{name}.csv", dtype={"case_id": str}).set_index("case_id")
    o = o.loc[folds.case_id]
    assert (o["true"].values == y).all()
    return o[classes]


def majority():
    pred = {}
    for f in sorted(folds.fold.unique()):
        maj = folds[folds.fold != f].label.value_counts().idxmax()
        for c in folds[folds.fold == f].case_id:
            pred[c] = maj
    return pd.Series(pred).loc[folds.case_id].values


frozen = {"mobilenet_v3_small": "Frozen MobileNetV3-S + LR",
          "resnet18": "Frozen ResNet-18 + LR",
          "efficientnet_b0": "Frozen EfficientNet-B0 + LR"}
P = {k: load(k) for k in frozen}
ens = sum(P.values()) / len(P)

rows = [("Majority class (training folds)", "primary", majority())]
rows += [(frozen[k], "primary", P[k].idxmax(axis=1).values) for k in frozen]
rows += [("Fine-tuned MobileNetV3-S", "primary", load("finetune_mobilenet_v3_small").idxmax(axis=1).values),
         ("From-scratch CNN", "primary", load("scratch_cnn").idxmax(axis=1).values),
         ("Frozen ensemble (post-hoc)", "exploratory", ens.idxmax(axis=1).values)]


def stat(yy, pp):
    return [accuracy_score(yy, pp), balanced_accuracy_score(yy, pp),
            f1_score(yy, pp, labels=classes, average="macro", zero_division=0)]


rng = np.random.default_rng(SEED)
grp = {g: np.where(folds.group.values == g)[0] for g in folds.group.unique()}
gl = list(grp)
boots = [np.concatenate([grp[g] for g in rng.choice(gl, len(gl))]) for _ in range(N_BOOT)]

out = []
for name, kind, pred in rows:
    pred = np.asarray(pred)
    pt = stat(y, pred)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        B = np.array([stat(y[b], pred[b]) for b in boots])
    lo, hi = np.percentile(B, [2.5, 97.5], axis=0)
    out.append({"model": name, "status": kind,
                "accuracy": round(pt[0], 3), "acc_lo": round(lo[0], 3), "acc_hi": round(hi[0], 3),
                "balanced_acc": round(pt[1], 3), "bacc_lo": round(lo[1], 3), "bacc_hi": round(hi[1], 3),
                "macro_F1": round(pt[2], 3), "f1_lo": round(lo[2], 3), "f1_hi": round(hi[2], 3)})
tab = pd.DataFrame(out)
tab.to_csv("outputs/reports/results_table.csv", index=False)
print(tab[["model", "status", "accuracy", "balanced_acc", "bacc_lo", "bacc_hi"]].to_string(index=False))

fig, ax = plt.subplots(figsize=(7.5, 3.8))
yy = np.arange(len(tab))[::-1]
ax.errorbar(tab.balanced_acc, yy, xerr=[tab.balanced_acc - tab.bacc_lo, tab.bacc_hi - tab.balanced_acc],
            fmt="o", capsize=3)
ax.axvline(0.25, ls="--", color="gray")
ax.text(0.252, yy.max() + 0.45, "chance (0.25)", color="gray", fontsize=8)
ax.set_yticks(yy)
ax.set_yticklabels(tab.model)
ax.set_xlabel("Balanced accuracy, pooled out-of-fold (95% group-bootstrap CI)")
ax.set_title("SCIN head/neck, 4 classes, 114 cases")
plt.tight_layout()
plt.savefig("outputs/figures/results_balanced_accuracy.png", dpi=150)
print("saved outputs/figures/results_balanced_accuracy.png")