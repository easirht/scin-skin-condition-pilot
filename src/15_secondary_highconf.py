import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score

SEED, N_BOOT = 42, 2000
MODELS = ["mobilenet_v3_small", "resnet18", "efficientnet_b0"]

folds = pd.read_csv("manifests/folds_headneck_v1.csv", dtype={"case_id": str, "group": str})
raw = pd.read_csv("manifests/scin_merged_raw.csv", dtype={"case_id": str})[["case_id", "top1_w"]]
f = folds.merge(raw, on="case_id", validate="one_to_one")
sub = f[f.top1_w >= 0.5].reset_index(drop=True)
print("subset (top-1 weight >= 0.5):", len(sub), "of", len(f), "cases")
print(sub.label.value_counts().to_string())
print("largest-class share (chance reference): %.3f" % (sub.label.value_counts().max() / len(sub)))

y = sub.label.values
rng = np.random.default_rng(SEED)
grp = {g: np.where(sub.group.values == g)[0] for g in sub.group.unique()}
gl = list(grp)
boots = [np.concatenate([grp[g] for g in rng.choice(gl, len(gl))]) for _ in range(N_BOOT)]
classes = sorted(f.label.unique())


def stat(yy, pp):
    return [accuracy_score(yy, pp), balanced_accuracy_score(yy, pp),
            f1_score(yy, pp, labels=classes, average="macro", zero_division=0)]


for m in MODELS:
    o = pd.read_csv(f"outputs/reports/oof_{m}.csv", dtype={"case_id": str}).set_index("case_id")
    assert (o.loc[sub.case_id, "true"].values == y).all()
    pred = o.loc[sub.case_id].drop(columns="true").idxmax(axis=1).values
    pt = stat(y, pred)
    B = np.array([stat(y[b], pred[b]) for b in boots])
    lo, hi = np.percentile(B, [2.5, 97.5], axis=0)
    print(f"\n{m} (secondary, n={len(sub)})")
    for n, p, a, b in zip(["accuracy", "balanced_acc", "macro_F1"], pt, lo, hi):
        print(f"  {n:14s} {p:.3f} [{a:.3f}, {b:.3f}]")
        