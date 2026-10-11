import warnings

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score

SEED, N_BOOT = 42, 2000
MODELS = ["mobilenet_v3_small", "resnet18", "efficientnet_b0"]

folds = pd.read_csv("manifests/folds_headneck_v1.csv", dtype={"case_id": str, "group": str})
classes = sorted(folds.label.unique())
y = folds.label.values


def load(m):
    o = pd.read_csv(f"outputs/reports/oof_{m}.csv", dtype={"case_id": str}).set_index("case_id")
    assert (o.loc[folds.case_id, "true"].values == y).all()
    return o.loc[folds.case_id, classes]


def stat(yy, pp):
    return [accuracy_score(yy, pp), balanced_accuracy_score(yy, pp),
            f1_score(yy, pp, labels=classes, average="macro", zero_division=0)]


rng = np.random.default_rng(SEED)
grp = {g: np.where(folds.group.values == g)[0] for g in folds.group.unique()}
gl = list(grp)
boots = [np.concatenate([grp[g] for g in rng.choice(gl, len(gl))]) for _ in range(N_BOOT)]

P = sum(load(m) for m in MODELS) / len(MODELS)
pred = P.idxmax(axis=1).values
with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    B = np.array([stat(y[b], pred[b]) for b in boots])
lo, hi = np.percentile(B, [2.5, 97.5], axis=0)
print(f"=== equal-weight ensemble of {MODELS} (pooled out-of-fold, n={len(y)}) ===")
for n, p, a, b in zip(["accuracy", "balanced_acc", "macro_F1"], stat(y, pred), lo, hi):
    print(f"  {n:14s} {p:.3f}  [{a:.3f}, {b:.3f}]")
print(pd.crosstab(pd.Series(y, name="true"), pd.Series(pred, name="pred")).to_string())