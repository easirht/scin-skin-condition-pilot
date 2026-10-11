import warnings

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score

SEED, N_BOOT = 42, 2000
R = "outputs/reports/"
folds = pd.read_csv("manifests/folds_v2.csv", dtype={"case_id": str, "group": str})
classes = sorted(folds.label.unique())
y = folds.label.values


def load(name, probs):
    o = pd.read_csv(R + name, dtype={"case_id": str}).set_index("case_id").loc[folds.case_id]
    assert (o["true"].values == y).all()
    return o[classes].idxmax(axis=1).values if probs else o["pred"].values


L, X = "locked protocol", "exploratory control (class-weighted)"
rows = [
    ("Majority class", L, load("oof_v2_majority.csv", False)),
    ("Body parts only + LR", L, load("oof_v2_meta_body.csv", False)),
    ("Richer metadata + LR", L, load("oof_v2_meta_rich.csv", False)),
    ("Frozen MobileNetV3-S + LR", L, load("oof_v2_frozen_mobilenet_v3_small.csv", True)),
    ("Frozen ResNet-18 + LR", L, load("oof_v2_frozen_resnet18.csv", True)),
    ("Frozen EfficientNet-B0 + LR", L, load("oof_v2_frozen_efficientnet_b0.csv", True)),
    ("Fine-tuned MobileNetV3-S (class-weighted loss)", L, load("oof_v2_finetune_mobilenet.csv", True)),
    ("Body parts only + LR, class-weighted", X, load("oof_v2bal_meta_body.csv", False)),
    ("Richer metadata + LR, class-weighted", X, load("oof_v2bal_meta_rich.csv", False)),
    ("Frozen MobileNetV3-S + LR, class-weighted", X, load("oof_v2_frozenbal_mobilenet_v3_small.csv", True)),
    ("Frozen ResNet-18 + LR, class-weighted", X, load("oof_v2_frozenbal_resnet18.csv", True)),
    ("Frozen EfficientNet-B0 + LR, class-weighted", X, load("oof_v2_frozenbal_efficientnet_b0.csv", True)),
]


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
tab.to_csv(R + "results_table_v2.csv", index=False)
for r in tab.itertuples():
    print(f"{r.model:48s} acc {r.accuracy:.3f} [{r.acc_lo:.3f}, {r.acc_hi:.3f}] | "
          f"bal {r.balanced_acc:.3f} [{r.bacc_lo:.3f}, {r.bacc_hi:.3f}] | F1 {r.macro_F1:.3f}")

fig, ax = plt.subplots(figsize=(8.5, 5))
yy = np.arange(len(tab))[::-1]
for kind, marker in ((L, "o"), (X, "s")):
    m = (tab.status == kind).values
    ax.errorbar(tab.balanced_acc[m], yy[m],
                xerr=[tab.balanced_acc[m] - tab.bacc_lo[m], tab.bacc_hi[m] - tab.balanced_acc[m]],
                fmt=marker, capsize=3, label=kind)
ax.axvline(0.125, ls="--", color="gray")
ax.text(0.127, yy.max() + 0.5, "chance (0.125)", color="gray", fontsize=8)
ax.set_yticks(yy)
ax.set_yticklabels(tab.model, fontsize=8)
ax.set_xlabel("Balanced accuracy, pooled out-of-fold (95% group-bootstrap CI)")
ax.set_title("SCIN v2: 8 classes, 1,172 cases, all body parts")
ax.legend(fontsize=8, loc="lower right")
plt.tight_layout()
plt.savefig("outputs/figures/results_v2_balanced_accuracy.png", dpi=150)
print("saved outputs/figures/results_v2_balanced_accuracy.png and outputs/reports/results_table_v2.csv")