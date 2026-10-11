import numpy as np
import pandas as pd
from sklearn.metrics import balanced_accuracy_score

SEED, N_BOOT = 42, 2000
R = "outputs/reports/"
folds = pd.read_csv("manifests/folds_v2.csv", dtype={"case_id": str, "group": str})
classes = sorted(folds.label.unique())
y = folds.label.values


def load(path, probs):
    o = pd.read_csv(path, dtype={"case_id": str}).set_index("case_id").loc[folds.case_id]
    assert (o["true"].values == y).all()
    return o[classes].idxmax(axis=1).values if probs else o["pred"].values


P = {"meta_body_bal": load(R + "oof_v2bal_meta_body.csv", False),
     "meta_rich_bal": load(R + "oof_v2bal_meta_rich.csv", False),
     "frozenbal_mobilenet": load(R + "oof_v2_frozenbal_mobilenet_v3_small.csv", True),
     "frozenbal_resnet18": load(R + "oof_v2_frozenbal_resnet18.csv", True),
     "frozenbal_efficientnet": load(R + "oof_v2_frozenbal_efficientnet_b0.csv", True),
     "finetuned_mobilenet": load(R + "oof_v2_finetune_mobilenet.csv", True)}

rng = np.random.default_rng(SEED)
grp = {g: np.where(folds.group.values == g)[0] for g in folds.group.unique()}
gl = list(grp)
boots = [np.concatenate([grp[g] for g in rng.choice(gl, len(gl))]) for _ in range(N_BOOT)]
B = {k: np.array([balanced_accuracy_score(y[b], p[b]) for b in boots]) for k, p in P.items()}
pt = {k: balanced_accuracy_score(y, p) for k, p in P.items()}

print("balanced accuracy, all class-weighted (95% group-bootstrap CI)")
for k in P:
    lo, hi = np.percentile(B[k], [2.5, 97.5])
    print(f"  {k:24s} {pt[k]:.3f} [{lo:.3f}, {hi:.3f}]")

thr_body = np.percentile(B["meta_body_bal"], 97.5)
thr_rich = np.percentile(B["meta_rich_bal"], 97.5)
print(f"\nweight-matched claim rule thresholds: body-part upper {thr_body:.3f}, richer-metadata upper {thr_rich:.3f}")
for k in ("frozenbal_mobilenet", "frozenbal_resnet18", "frozenbal_efficientnet", "finetuned_mobilenet"):
    lo = np.percentile(B[k], 2.5)
    print(f"  {k:24s} lower {lo:.3f} | > body-part upper: {'yes' if lo > thr_body else 'no'}"
          f" | > richer-metadata upper: {'yes' if lo > thr_rich else 'no'}")


def paired(a, b):
    d = B[a] - B[b]
    lo, hi = np.percentile(d, [2.5, 97.5])
    print(f"  {a:24s} - {b:18s} diff {pt[a] - pt[b]:+.3f} [{lo:+.3f}, {hi:+.3f}] "
          f"{'CI excludes 0' if lo > 0 or hi < 0 else 'CI includes 0'}")


print("\npaired differences (same resampled groups):")
for b in ("meta_body_bal", "meta_rich_bal", "frozenbal_mobilenet", "frozenbal_resnet18", "frozenbal_efficientnet"):
    paired("finetuned_mobilenet", b)
for a in ("frozenbal_mobilenet", "frozenbal_resnet18", "frozenbal_efficientnet"):
    paired(a, "meta_body_bal")
    paired(a, "meta_rich_bal")