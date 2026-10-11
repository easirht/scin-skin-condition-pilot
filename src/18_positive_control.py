import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score, roc_auc_score
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

SEED, C, N_BOOT = 42, 0.01, 1000
V = {"face", "partial_face", "no_face", "unclear"}

full = pd.read_csv("manifests/face_review_full.csv", dtype=str, keep_default_na=False)
rep = full.is_repeat.str.lower().eq("true")
v1 = pd.read_csv("manifests/face_review_manual_annotated_v1.csv", dtype=str, keep_default_na=False)
v1 = v1[v1.judgement.isin(V)].drop_duplicates("image_path")
j = pd.concat([v1[["image_path", "judgement"]], full[~rep][["image_path", "judgement"]]])
assert j.image_path.is_unique

paths = pd.read_csv("data/interim/features_image_paths.csv").image_path.tolist()
folds = pd.read_csv("manifests/folds_headneck_v1.csv", dtype={"case_id": str, "group": str})
cand = pd.read_csv("manifests/scin_candidate_images.csv", dtype={"case_id": str})
g = cand[["case_id", "image_path"]].merge(folds[["case_id", "group"]], on="case_id") \
        .drop_duplicates("image_path")[["image_path", "group"]]

d = pd.DataFrame({"image_path": paths, "row": range(len(paths))}) \
      .merge(j, on="image_path", validate="one_to_one") \
      .merge(g, on="image_path", validate="one_to_one")
d = d[d.judgement != "unclear"].reset_index(drop=True)
d["y"] = d.judgement.isin(["face", "partial_face"]).astype(int)
print("images:", len(d), "| facial share: %.2f" % d.y.mean(), "(unclear excluded)")

rng = np.random.default_rng(SEED)
grp = {k: np.where(d.group.values == k)[0] for k in d.group.unique()}
gl = list(grp)
boots = [np.concatenate([grp[k] for k in rng.choice(gl, len(gl))]) for _ in range(N_BOOT)]

for m in ["mobilenet_v3_small", "resnet18", "efficientnet_b0"]:
    X = np.load(f"data/interim/features_{m}.npy")[d.row.values]
    p = np.zeros(len(d))
    for tr, te in StratifiedGroupKFold(5, shuffle=True, random_state=SEED).split(X, d.y, d.group):
        clf = make_pipeline(StandardScaler(),
                            LogisticRegression(C=C, class_weight="balanced", max_iter=3000))
        clf.fit(X[tr], d.y.values[tr])
        p[te] = clf.predict_proba(X[te])[:, 1]
    yy = d.y.values
    auc = roc_auc_score(yy, p)
    ba = balanced_accuracy_score(yy, p > 0.5)
    B = [(roc_auc_score(yy[b], p[b]), balanced_accuracy_score(yy[b], p[b] > 0.5))
         for b in boots if 0 < yy[b].sum() < len(b)]
    lo, hi = np.percentile(np.array(B), [2.5, 97.5], axis=0)
    print(f"{m:20s} AUC {auc:.3f} [{lo[0]:.3f}, {hi[0]:.3f}] | balanced acc {ba:.3f} [{lo[1]:.3f}, {hi[1]:.3f}]")