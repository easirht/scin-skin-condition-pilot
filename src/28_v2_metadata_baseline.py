import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegressionCV
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score, recall_score
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

SEED, N_BOOT = 42, 2000
CS = [1e-4, 1e-3, 1e-2, 1e-1, 1, 10]

folds = pd.read_csv("manifests/folds_v2.csv", dtype={"case_id": str, "group": str})
raw = pd.read_csv("manifests/scin_merged_raw.csv", dtype={"case_id": str})
PREFIX = ("body_parts_", "textures_", "condition_symptoms_", "other_symptoms_")
body_cols = [c for c in raw.columns if c.startswith("body_parts_")]
rich_cols = [c for c in raw.columns if c.startswith(PREFIX)]
cat_cols = ["age_group", "sex_at_birth"]
d = folds.merge(raw[["case_id"] + rich_cols + cat_cols], on="case_id", how="left", validate="one_to_one")
assert len(d) == len(folds) == 1172

for c in rich_cols:  # this is where the earlier "True" vs "YES" bug hid, so report anything odd
    vals = set(d[c].dropna().astype(str).str.upper().unique())
    if not vals <= {"YES"}:
        print("WARNING unexpected values in", c, vals)


def yes(s):
    return s.astype(str).str.upper().eq("YES").astype(float).values


X_body = np.column_stack([yes(d[c]) for c in body_cols])
cat = pd.get_dummies(d[cat_cols].fillna("MISSING"), dtype=float)
X_rich = np.column_stack([yes(d[c]) for c in rich_cols] + [cat.values])
print("features: body parts", X_body.shape[1], "| richer metadata", X_rich.shape[1])

y = d.label.values
classes = sorted(set(y))
fold = d.fold.values
group = d.group.values


def oof_logreg(X):
    pred = np.empty(len(d), dtype=object)
    chosen = []
    for f in sorted(set(fold)):
        tr, te = np.where(fold != f)[0], np.where(fold == f)[0]
        inner = list(StratifiedGroupKFold(3, shuffle=True, random_state=SEED)
                     .split(tr, y[tr], group[tr]))
        clf = make_pipeline(StandardScaler(),
                            LogisticRegressionCV(Cs=CS, cv=inner, scoring="balanced_accuracy",
                                                 max_iter=3000))
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            clf.fit(X[tr], y[tr])
        chosen.append(float(clf[-1].C_[0]))
        pred[te] = clf.predict(X[te])
    return pred, chosen


def oof_majority():
    pred = np.empty(len(d), dtype=object)
    for f in sorted(set(fold)):
        tr = fold != f
        pred[~tr] = pd.Series(y[tr]).value_counts().idxmax()
    return pred


def stat(yy, pp):
    rec = recall_score(yy, pp, labels=classes, average=None, zero_division=0)
    return np.r_[accuracy_score(yy, pp), balanced_accuracy_score(yy, pp),
                 f1_score(yy, pp, labels=classes, average="macro", zero_division=0), rec]


names = ["accuracy", "balanced_acc", "macro_F1"] + [f"recall_{c}" for c in classes]
rng = np.random.default_rng(SEED)
grp = {g: np.where(group == g)[0] for g in np.unique(group)}
gl = list(grp)
boots = [np.concatenate([grp[g] for g in rng.choice(gl, len(gl))]) for _ in range(N_BOOT)]
Path("outputs/reports").mkdir(parents=True, exist_ok=True)


def report(tag, title, pred):
    pred = np.asarray(pred)
    pd.DataFrame({"case_id": d.case_id, "true": y, "pred": pred}).to_csv(
        f"outputs/reports/oof_v2_{tag}.csv", index=False)
    pt = stat(y, pred)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        B = np.array([stat(y[b], pred[b]) for b in boots])
    lo, hi = np.percentile(B, [2.5, 97.5], axis=0)
    print(f"\n=== {title} (pooled out-of-fold, n={len(y)} cases) ===")
    for n, p, a, b in zip(names, pt, lo, hi):
        print(f"  {n:42s} {p:.3f}  [{a:.3f}, {b:.3f}]")
    print(pd.crosstab(pd.Series(y, name="true"), pd.Series(pred, name="pred")).to_string())
    return hi[1]


report("majority", "majority class (training folds)", oof_majority())
p, ch = oof_logreg(X_body)
print("\nC chosen per fold (body parts):", ch)
report("meta_body", "metadata-only: body_parts_* + logistic regression", p)
p, ch = oof_logreg(X_rich)
print("\nC chosen per fold (richer metadata):", ch)
report("meta_rich", "metadata-only: body parts, textures, symptoms, age, sex + logistic regression", p)
print(f"\nCI: {N_BOOT} group-bootstrap resamples, seed {SEED}. No per-fold metrics reported (policy).")