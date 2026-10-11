import numpy as np
import pandas as pd
from sklearn.metrics import cohen_kappa_score

FACIAL = {"face", "partial_face"}
VALID = {"face", "partial_face", "no_face", "unclear"}
MIN_FACIAL_CASES = 15


def wilson(k, n, z=1.96):
    if n == 0:
        return (float("nan"), float("nan"))
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (round(c - h, 2), round(c + h, 2))


full = pd.read_csv("manifests/face_review_full.csv", dtype=str, keep_default_na=False)
assert set(full.judgement) <= VALID and (full.judgement != "").all(), "incomplete annotation"
rep = full.is_repeat.str.lower().eq("true")

v1 = pd.read_csv("manifests/face_review_manual_annotated_v1.csv", dtype=str, keep_default_na=False)
v1 = v1[v1.judgement.isin(VALID)].drop_duplicates("image_path")

# primary judgement per unique image: v1 for the 39 reference images, full sheets for the other 228
j = pd.concat([v1[["image_path", "judgement"]].assign(source="review_v1"),
               full[~rep][["image_path", "judgement"]].assign(source="full_sheets")])
assert j.image_path.is_unique, "an image has two primary judgements"
print("unique images judged:", len(j))
print("\n== judgement distribution (unique images) ==")
print(j.judgement.value_counts().to_string())
print("share facial (face+partial): %.2f" % j.judgement.isin(FACIAL).mean(),
      "Wilson CI", wilson(int(j.judgement.isin(FACIAL).sum()), len(j)))

# intra-rater agreement
r = full[rep]
k_all = cohen_kappa_score(r.v1_judgement, r.judgement)
b1, b2 = r.v1_judgement.isin(FACIAL), r.judgement.isin(FACIAL)
print("\n== intra-rater (images judged twice) ==")
print("n =", len(r), "| exact 4-level agreement:", int((r.v1_judgement == r.judgement).sum()),
      "| facial/non-facial agreement:", int((b1 == b2).sum()))
print("Cohen kappa (4-level, n small, treat as rough):", round(k_all, 2))
print(pd.crosstab(r.v1_judgement, r.judgement).to_string())

# case level, with duplicate-linked groups (shared SHA-256 image)
aud = pd.read_csv("manifests/scin_candidate_images_audit.csv", dtype={"case_id": str})
m = aud.merge(j[["image_path", "judgement"]], on="image_path", how="left", validate="many_to_one")
assert m.judgement.notna().all()
parent = {c: c for c in m.case_id.unique()}


def find(x):
    while parent[x] != x:
        parent[x] = parent[parent[x]]
        x = parent[x]
    return x


for _, g in m.groupby("sha256"):
    cs = g.case_id.unique()
    for c in cs[1:]:
        parent[find(c)] = find(cs[0])
m["group"] = m.case_id.map(find)

m["facial_img"] = m.judgement.isin(FACIAL)
m["face_img"] = m.judgement.eq("face")
case = m.groupby("case_id").agg(label=("label", "first"), group=("group", "first"),
                                n_img=("image_path", "size"),
                                facial=("facial_img", "any"), face_only=("face_img", "any"))
case.to_csv("manifests/face_review_final.csv")

print("\n== class-wise counts (primary: >=1 image face/partial_face) ==")
rows = []
for lab, g in case.groupby("label"):
    rows.append({"class": lab, "cases": len(g), "groups": g.group.nunique(),
                 "facial_cases": int(g.facial.sum()),
                 "facial_groups": g[g.facial].group.nunique(),
                 "face_only_cases (sensitivity)": int(g.face_only.sum()),
                 "meets_rule": int(g.facial.sum()) >= MIN_FACIAL_CASES})
tab = pd.DataFrame(rows).sort_values("facial_cases", ascending=False)
print(tab.to_string(index=False))
kept = tab[tab.meets_rule]["class"].tolist()
print(f"\nrule (>= {MIN_FACIAL_CASES} facial cases): classes kept = {kept}")
print("fewer than 3 classes kept -> fallback rule triggered:", len(kept) < 3)

# Haar detector vs the full annotation (unclear excluded)
det = pd.read_csv("manifests/face_detect_haar.csv")
ref = j[j.judgement != "unclear"].merge(det, on="image_path", validate="one_to_one")
ref["truth"] = ref.judgement.isin(FACIAL)
tp = int((ref.truth & ref.det_pos).sum()); fn = int((ref.truth & ~ref.det_pos).sum())
fp = int((~ref.truth & ref.det_pos).sum()); tn = int((~ref.truth & ~ref.det_pos).sum())
print(f"\n== Haar detector vs full manual annotation (n={len(ref)}) ==")
print(f"TP {tp} FN {fn} FP {fp} TN {tn} | sensitivity {tp/max(tp+fn,1):.2f} CI {wilson(tp, tp+fn)} "
      f"| specificity {tn/max(tn+fp,1):.2f} CI {wilson(tn, tn+fp)}")