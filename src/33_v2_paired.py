import numpy as np
import pandas as pd
from sklearn.metrics import balanced_accuracy_score, recall_score

SEED, N_BOOT = 42, 2000
R = "outputs/reports/"
folds = pd.read_csv("manifests/folds_v2.csv", dtype={"case_id": str, "group": str})
classes = sorted(folds.label.unique())
y = folds.label.values


def load(path, probs):
    o = pd.read_csv(path, dtype={"case_id": str}).set_index("case_id").loc[folds.case_id]
    assert (o["true"].values == y).all()
    return o[classes].idxmax(axis=1).values if probs else o["pred"].values


preds = {"majority": load(R + "oof_v2_majority.csv", False),
         "meta_body": load(R + "oof_v2_meta_body.csv", False),
         "meta_rich": load(R + "oof_v2_meta_rich.csv", False),
         "frozen_mobilenet": load(R + "oof_v2_frozen_mobilenet_v3_small.csv", True),
         "frozen_resnet18": load(R + "oof_v2_frozen_resnet18.csv", True),
         "frozen_efficientnet": load(R + "oof_v2_frozen_efficientnet_b0.csv", True)}
ft = load(R + "oof_v2_finetune_mobilenet.csv", True)

rng = np.random.default_rng(SEED)
grp = {g: np.where(folds.group.values == g)[0] for g in folds.group.unique()}
gl = list(grp)
boots = [np.concatenate([grp[g] for g in rng.choice(gl, len(gl))]) for _ in range(N_BOOT)]
bacc = lambda yy, pp: balanced_accuracy_score(yy, pp)
b_ft = np.array([bacc(y[b], ft[b]) for b in boots])

print("paired bootstrap: balanced accuracy of fine-tuned MobileNetV3-S minus each baseline "
      "(same resampled groups, 95% CI)")
for name, p in preds.items():
    d = b_ft - np.array([bacc(y[b], p[b]) for b in boots])
    lo, hi = np.percentile(d, [2.5, 97.5])
    pt = bacc(y, ft) - bacc(y, p)
    print(f"  vs {name:20s} diff {pt:+.3f}  [{lo:+.3f}, {hi:+.3f}]  "
          f"{'CI excludes 0' if lo > 0 or hi < 0 else 'CI includes 0'}")

rec = recall_score(y, ft, labels=classes, average=None, zero_division=0)
R_b = np.array([recall_score(y[b], ft[b], labels=classes, average=None, zero_division=0) for b in boots])
lo, hi = np.percentile(R_b, [2.5, 97.5], axis=0)
print("\nfine-tuned per-class recall:")
for c, r, a, b in zip(classes, rec, lo, hi):
    print(f"  {c:30s} {r:.3f} [{a:.3f}, {b:.3f}]")
print(pd.crosstab(pd.Series(y, name="true"), pd.Series(ft, name="pred")).to_string())
print("\npredicted-class share:", pd.Series(ft).value_counts(normalize=True).round(3).to_dict())