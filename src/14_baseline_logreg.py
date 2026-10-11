import warnings

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegressionCV
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score, recall_score
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

SEED, N_BOOT = 42, 2000
MODELS = ["mobilenet_v3_small", "resnet18", "efficientnet_b0"]
CS = [1e-4, 1e-3, 1e-2, 1e-1, 1, 10]

folds = pd.read_csv("manifests/folds_headneck_v1.csv", dtype={"case_id": str, "group": str})
classes = sorted(folds.label.unique())
cand = pd.read_csv("manifests/scin_candidate_images.csv", dtype={"case_id": str})
paths = pd.read_csv("data/interim/features_image_paths.csv").image_path.tolist()
idx = {p: i for i, p in enumerate(paths)}

img = cand[["case_id", "image_path"]].merge(
    folds[["case_id", "label", "fold", "group"]], on="case_id", validate="many_to_one")
img["row"] = img.image_path.map(idx)
assert len(img) == 269 and img.row.notna().all()
img["row"] = img.row.astype(int)
case_label = folds.set_index("case_id").label
print("cases:", len(folds), "| image rows:", len(img), "| classes:", classes)


def logreg_oof(X):
    out, chosen = [], []
    for f in sorted(folds.fold.unique()):
        tr, te = img[img.fold != f], img[img.fold == f]
        inner = list(StratifiedGroupKFold(3, shuffle=True, random_state=SEED)
                     .split(tr, tr.label, tr.group))
        clf = make_pipeline(StandardScaler(),
                            LogisticRegressionCV(Cs=CS, cv=inner, scoring="balanced_accuracy",
                                                 max_iter=3000))
        clf.fit(X[tr.row.values], tr.label.values)
        chosen.append(float(clf[-1].C_[0]))
        P = pd.DataFrame(clf.predict_proba(X[te.row.values]), columns=clf.classes_)
        P["case_id"] = te.case_id.values
        out.append(P.groupby("case_id").mean())  # case level = mean over the case's images
    P = pd.concat(out).loc[folds.case_id]
    return P, chosen


def majority_oof():
    pred = {}
    for f in sorted(folds.fold.unique()):
        maj = folds[folds.fold != f].label.value_counts().idxmax()
        for c in folds[folds.fold == f].case_id:
            pred[c] = maj
    return pd.Series(pred).loc[folds.case_id]


def stats(y, p):
    rec = recall_score(y, p, labels=classes, average=None, zero_division=0)
    return np.r_[accuracy_score(y, p), balanced_accuracy_score(y, p),
                 f1_score(y, p, labels=classes, average="macro", zero_division=0), rec]


names = ["accuracy", "balanced_acc", "macro_F1"] + [f"recall_{c}" for c in classes]
y = folds.label.values
grp = {g: np.where(folds.group.values == g)[0] for g in folds.group.unique()}
glist = list(grp)
rng = np.random.default_rng(SEED)
boot_idx = [np.concatenate([grp[g] for g in rng.choice(glist, len(glist))]) for _ in range(N_BOOT)]


def report(title, pred):
    pred = np.asarray(pred)
    point = stats(y, pred)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        B = np.array([stats(y[b], pred[b]) for b in boot_idx])
    lo, hi = np.percentile(B, [2.5, 97.5], axis=0)
    print(f"\n=== {title} (pooled out-of-fold, n={len(y)} cases) ===")
    for n, p, a, b in zip(names, point, lo, hi):
        print(f"  {n:42s} {p:.3f}  [{a:.3f}, {b:.3f}]")
    print(pd.crosstab(pd.Series(y, name="true"), pd.Series(pred, name="pred")).to_string())


report("majority-class baseline", majority_oof().values)
for m in MODELS:
    X = np.load(f"data/interim/features_{m}.npy")
    assert X.shape[0] == len(paths)
    P, chosen = logreg_oof(X)
    P.assign(true=y).to_csv(f"outputs/reports/oof_{m}.csv")
    print(f"\n[{m}] regularisation C chosen per fold (inner CV only): {chosen}")
    report(f"{m} + logistic regression", P.idxmax(axis=1).values)
print(f"\nCI: {N_BOOT} bootstrap resamples of duplicate-linked groups, seed {SEED}. "
      "No per-fold metrics reported (policy).")