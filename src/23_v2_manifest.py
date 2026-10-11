import ast

import pandas as pd

MIN_CASES = 50
df = pd.read_csv("manifests/scin_merged_raw.csv", dtype={"case_id": str})
lab = df[df.has_label].copy()


def has_tie(s):
    d = ast.literal_eval(s)
    m = max(d.values())
    return sum(abs(v - m) < 1e-9 for v in d.values()) > 1


lab["tie"] = lab.weighted_skin_condition_label.apply(has_tie)
ok = lab[~lab.tie]
counts = ok.top1.value_counts()
classes = counts[counts >= MIN_CASES].index.tolist()
cand = ok[ok.top1.isin(classes)].copy()
print("classes:", classes)
print("cases:", len(cand))
print(cand.top1.value_counts().to_string())

rows = []
for _, r in cand.iterrows():
    for k in (1, 2, 3):
        p = r.get(f"image_{k}_path")
        if pd.notna(p):
            rows.append({"case_id": r.case_id, "label": r.top1, "top1_w": r.top1_w,
                         "image_idx": k, "shot_type": r.get(f"image_{k}_shot_type"),
                         "image_path": p})
img = pd.DataFrame(rows)
img.to_csv("manifests/v2_candidate_images.csv", index=False)
print("image rows:", len(img), "| unique files:", img.image_path.nunique())

keep = ["case_id", "top1", "top1_w", "n_images"] + [c for c in df.columns if c.startswith("body_parts_")]
cand[keep].to_csv("manifests/v2_cases.csv", index=False)
print("saved manifests/v2_candidate_images.csv and manifests/v2_cases.csv")