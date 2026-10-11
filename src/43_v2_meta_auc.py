import warnings

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegressionCV
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

SEED, N_BOOT = 42, 2000
CS = [1e-4, 1e-3, 1e-2, 1e-1, 1, 10]
R = "outputs/reports/"

folds = pd.read_csv("manifests/folds_v2.csv", dtype={"case_id": str, "group": str})
raw = pd.read_csv("manifests/scin_merged_raw.csv", dtype={"case_id": str})
PREFIX = ("body_parts_", "textures_", "condition_symptoms_", "other_symptoms_")
body_cols = [c for c in raw.columns if c.startswith("body_parts_")]
rich_cols = [c for c in raw.columns if c.startswith(PREFIX)]
cat_cols = ["age_group", "sex_at_birth"]
d = folds.merge(raw[["case_id"] + rich_cols + cat_cols], on="case_id", how="left", validate="one_to_one")
assert len(d) == len(folds) == 1172


def yes(s):
    return s.astype(str).str.upper().eq("YES").astype(float).values


X_body = np.column_stack([yes(d[c]) for c in body_cols])
cat = pd.get_dummies(d[cat_cols].fillna("MISSING"), dtype=float)
X_rich = np.column_stack([yes(d[c]) for c in rich_cols] + [cat.values])
print("features: body parts", X_body.shape[1], "| richer metadata", X_rich.shape[1])

y = d.label.values
classes = sorted(set(y))
K = len(classes)
fold, group = d.fold.values, d.group.values
Y = np.column_stack([(y == c).astype(int) for c in classes])


def oof_proba(X, weighted):
    P = np.zeros((len(d), K))
    for f in sorted(set(fold)):
        tr, te = np.where(fold != f)[0], np.where(fold == f)[0]
        inner = list(StratifiedGroupKFold(3, shuffle=True, random_state=SEED).split(tr, y[tr], group[tr]))
        kw = {"class_weight": "balanced"} if weighted else {}
        clf = make_pipeline(StandardScaler(),
                            LogisticRegressionCV(Cs=CS, cv=inner, scoring="balanced_accuracy",
                                                 max_iter=3000, **kw))
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            clf.fit(X[tr], y[tr])
        assert list(clf.classes_) == classes
        P[te] = clf.predict_proba(X[te])
        print(f"    fold {f} done", flush=True)
    return P


def load_img(name):
    o = pd.read_csv(R + name, dtype={"case_id": str}).set_index("case_id").loc[d.case_id]
    assert (o["true"].values == y).all()
    return o[classes].values


def macro(P, idx):
    a, p = [], []
    for k in range(K):
        t = Y[idx, k]
        if 0 < t.sum() < len(t):
            a.append(roc_auc_score(t, P[idx, k]))
            p.append(average_precision_score(t, P[idx, k]))
    return np.mean(a), np.mean(p)


def per_class_auc(P):
    return [roc_auc_score(Y[:, k], P[:, k]) for k in range(K)]


rng = np.random.default_rng(SEED)
grp = {g: np.where(group == g)[0] for g in np.unique(group)}
gl = list(grp)
boots = [np.concatenate([grp[g] for g in rng.choice(gl, len(gl))]) for _ in range(N_BOOT)]
allidx = np.arange(len(y))

VARIANTS = {
    "Body parts (unweighted)": (X_body, False, "oof_v2_meta_body.csv", "body_unw"),
    "Richer metadata (unweighted)": (X_rich, False, "oof_v2_meta_rich.csv", "rich_unw"),
    "Body parts (weighted)": (X_body, True, "oof_v2bal_meta_body.csv", "body_w"),
    "Richer metadata (weighted)": (X_rich, True, "oof_v2bal_meta_rich.csv", "rich_w"),
}
IMG = {"Fine-tuned MobileNetV3-S (weighted)": "oof_v2_finetune_mobilenet.csv",
       "Frozen MobileNetV3-S (weighted)": "oof_v2_frozenbal_mobilenet_v3_small.csv",
       "Frozen ResNet-18 (weighted)": "oof_v2_frozenbal_resnet18.csv",
       "Frozen EfficientNet-B0 (weighted)": "oof_v2_frozenbal_efficientnet_b0.csv"}

probs, boot = {}, {}
with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    for name, f in IMG.items():
        probs[name] = load_img(f)
        boot[name] = np.array([macro(probs[name], b) for b in boots])
    print(f"consistency: fine-tuned macro AUC {macro(probs['Fine-tuned MobileNetV3-S (weighted)'], allidx)[0]:.3f} (expected 0.707)")
    for name, (X, w, saved, tag) in VARIANTS.items():
        print(f"\nrefitting {name}", flush=True)
        P = oof_proba(X, w)
        old = pd.read_csv(R + saved, dtype={"case_id": str}).set_index("case_id").loc[d.case_id]["pred"].values
        assert (np.array(classes)[P.argmax(1)] == old).all(), f"{name}: argmax differs from the saved predictions"
        pd.DataFrame(P, columns=classes, index=d.case_id).assign(true=y).to_csv(R + f"oof_v2prob_meta_{tag}.csv")
        probs[name] = P
        boot[name] = np.array([macro(P, b) for b in boots])

print("\nmacro ROC-AUC [95% CI] | macro AP (chance reference about 0.125; AUC chance 0.5)")
for name in list(VARIANTS) + list(IMG):
    a, p = macro(probs[name], allidx)
    lo, hi = np.percentile(boot[name][:, 0], [2.5, 97.5])
    print(f"  {name:38s} {a:.3f} [{lo:.3f}, {hi:.3f}] | AP {p:.3f}")

print("\nper-class AUC point estimates, metadata models:")
print("  " + " | ".join(c[:12] for c in classes))
for name in VARIANTS:
    print(f"  {name:30s} " + " ".join(f"{v:.3f}" for v in per_class_auc(probs[name])))


def paired(a, b):
    dd = boot[a][:, 0] - boot[b][:, 0]
    lo, hi = np.percentile(dd, [2.5, 97.5])
    pt = macro(probs[a], allidx)[0] - macro(probs[b], allidx)[0]
    print(f"  {a:36s} - {b:28s} {pt:+.3f} [{lo:+.3f}, {hi:+.3f}] {'CI excludes 0' if lo > 0 or hi < 0 else 'CI includes 0'}")


print("\npaired macro-AUC differences (class-weighted setting only, same resampled groups):")
paired("Fine-tuned MobileNetV3-S (weighted)", "Body parts (weighted)")
paired("Fine-tuned MobileNetV3-S (weighted)", "Richer metadata (weighted)")
for n in ("Frozen MobileNetV3-S (weighted)", "Frozen ResNet-18 (weighted)", "Frozen EfficientNet-B0 (weighted)"):
    paired(n, "Richer metadata (weighted)")