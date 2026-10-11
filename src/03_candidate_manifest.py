import ast
import pandas as pd



df = pd.read_csv("manifests/scin_merged_raw.csv")
hn = df[df.is_head_neck & df.has_label].copy()


def has_tie(s):
    d = ast.literal_eval(s)
    m = max(d.values())
    return sum(abs(v - m) < 1e-9 for v in d.values()) > 1


hn["top1_tie"] = hn["weighted_skin_condition_label"].apply(has_tie)
print("head/neck labelled:", len(hn), "| with top-1 tie:", int(hn.top1_tie.sum()))

MIN_CASES = 20
_counts = hn[~hn.top1_tie].top1.value_counts()
CLASSES = _counts[_counts >= MIN_CASES].index.tolist()
print("classes kept by policy:", CLASSES)

cand = hn[hn.top1.isin(CLASSES) & ~hn.top1_tie].copy()
print("\ncandidate cases per class (ties excluded):")
print(cand.top1.value_counts().to_string())
print("total candidate cases:", len(cand))

# one row per image
rows = []
for _, r in cand.iterrows():
    for k in (1, 2, 3):
        p = r.get(f"image_{k}_path")
        if pd.notna(p):
            rows.append({
                "case_id": r.case_id, "label": r.top1, "top1_w": r.top1_w,
                "head_neck_only": r.head_neck_only,
                "image_idx": k, "shot_type": r.get(f"image_{k}_shot_type"),
                "image_path": p,
            })
img = pd.DataFrame(rows)
img.to_csv("manifests/scin_candidate_images.csv", index=False)
print("\ncandidate images:", len(img))
print("\nsample paths:")
print(img.image_path.head(5).to_string(index=False))