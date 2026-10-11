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
BODY_UPPER, RICH_UPPER = 0.179, 0.232  # locked reference values from the metadata-only controls

folds = pd.read_csv("manifests/folds_v2.csv", dtype={"case_id": str, "group": str})
classes = sorted(folds.label.unique())
cand = pd.read_csv("manifests/v2_candidate_images.csv", dtype={"case_id": str})
paths = pd.read_csv("data/interim/v2_features_image_paths.csv").image_path.tolist()
idx = {p: i for i, p in enumerate(paths)}

img = cand[["case_id", "image_path"]].merge(
    folds[["case_id", "label", "fold", "group"]], on="case_id", validate="many_to_one")
img["row"] = img.image_path.map(idx)
assert img.row.notna().all()
img["row"] = img.row.astype(int)
assert len(img) == len(cand) == 2560 and img.case_id.nunique() == len(folds) == 1172
print("cases:", len(folds), "| image rows:", len(img), "| classes:", classes, flush=True)


def logreg_oof(X):
    out, chosen = [], []
    for f in sorted(folds.fold.unique()):
        tr, te = img[img.fold != f], img[img.fold == f]
        inner = list(StratifiedGroupKFold(3, shuffle=True, random_state=SEED)
                     .split(tr, tr.label, tr.group))
        clf = make_pipeline(StandardScaler(),
                            LogisticRegressionCV(Cs=CS, cv=inner, scoring="balanced_accuracy",
                                                 max_iter=3000))
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            clf.fit(X[tr.row.values], tr.label.values)
        chosen.append(float(clf[-1].C_[0]))
        P = pd.DataFrame(clf.predict_proba(X[te.row.values]), columns=clf.classes_)
        P["case_id"] = te.case_id.values
        out.append(P.groupby("case_id").mean())  # case level = mean over the case's images
        print(f"    fold {f} done", flush=True)
    return pd.concat(out).loc[folds.case_id], chosen


def stat(yy, pp):
    rec = recall_score(yy, pp, labels=classes, average=None, zero_division=0)
    return np.r_[accuracy_score(yy, pp), balanced_accuracy_score(yy, pp),
                 f1_score(yy, pp, labels=classes, average="macro", zero_division=0), rec]


names = ["accuracy", "balanced_acc", "macro_F1"] + [f"recall_{c}" for c in classes]
y = folds.label.values
group = folds.group.values
rng = np.random.default_rng(SEED)
grp = {g: np.where(group == g)[0] for g in np.unique(group)}
gl = list(grp)
boots = [np.concatenate([grp[g] for g in rng.choice(gl, len(gl))]) for _ in range(N_BOOT)]

for m in MODELS:
    X = np.load(f"data/interim/v2_features_{m}.npy")
    assert X.shape[0] == len(paths) and np.isfinite(X).all()
    print(f"\n[{m}] features {X.shape}", flush=True)
    P, chosen = logreg_oof(X)
    P.assign(true=y).to_csv(f"outputs/reports/oof_v2_frozen_{m}.csv")
    pred = P.idxmax(axis=1).values
    pt = stat(y, pred)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        B = np.array([stat(y[b], pred[b]) for b in boots])
    lo, hi = np.percentile(B, [2.5, 97.5], axis=0)
    print(f"C chosen per fold (inner CV only): {chosen}")
    print(f"=== {m} + logistic regression (pooled out-of-fold, n={len(y)} cases) ===")
    for n, p, a, b in zip(names, pt, lo, hi):
        print(f"  {n:42s} {p:.3f}  [{a:.3f}, {b:.3f}]")
    print(pd.crosstab(pd.Series(y, name="true"), pd.Series(pred, name="pred")).to_string())
    print(f"claim rule (balanced-acc CI lower bound > {BODY_UPPER}): "
          f"{'PASS' if lo[1] > BODY_UPPER else 'FAIL'} (lower bound {lo[1]:.3f}) "
          f"| above richer-metadata upper bound {RICH_UPPER}: {'yes' if lo[1] > RICH_UPPER else 'no'}")
print(f"\nCI: {N_BOOT} group-bootstrap resamples, seed {SEED}. No per-fold metrics reported (policy).")