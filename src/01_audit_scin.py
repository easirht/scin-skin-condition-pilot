import ast
from collections import Counter
from pathlib import Path

import pandas as pd

RAW = Path("data/raw/scin")
cases = pd.read_csv(RAW / "scin_cases.csv")
labels = pd.read_csv(RAW / "scin_labels.csv")

print("cases:", cases.shape, "| labels:", labels.shape)
print("unique case_id  cases:", cases.case_id.nunique(), "labels:", labels.case_id.nunique())
print("release:", cases["release"].value_counts().to_dict())

df = cases.merge(labels, on="case_id", how="inner", validate="one_to_one")
print("merged:", df.shape)


def as_bool(s):
    # SCIN encodes these columns as "YES" or blank (NaN)
    return s.astype(str).str.strip().str.upper().eq("YES")


# ---- body parts ----
body_cols = [c for c in df.columns if c.startswith("body_parts_")]
print("\n== body part counts (True) ==")
for c in body_cols:
    print(f"{c:40s} {as_bool(df[c]).sum()}")

df["is_head_neck"] = as_bool(df["body_parts_head_or_neck"])
df["n_body_parts"] = sum(as_bool(df[c]).astype(int) for c in body_cols)
df["head_neck_only"] = df["is_head_neck"] & (df["n_body_parts"] == 1)
print("\nhead_or_neck ticked:", df["is_head_neck"].sum())
print("head_or_neck ONLY (no other part):", df["head_neck_only"].sum())

# ---- images ----
img_cols = ["image_1_path", "image_2_path", "image_3_path"]
df["n_images"] = df[img_cols].notna().sum(axis=1)
print("\nimages per case:", df["n_images"].value_counts().sort_index().to_dict())
print("total images:", int(df["n_images"].sum()))
print("total images in head_or_neck cases:", int(df.loc[df.is_head_neck, "n_images"].sum()))


# ---- labels ----
def parse_w(x):
    if pd.isna(x):
        return {}
    try:
        d = ast.literal_eval(x)
        return d if isinstance(d, dict) else {}
    except Exception:
        return {}


df["w"] = df["weighted_skin_condition_label"].apply(parse_w)
df["has_label"] = df["w"].apply(len) > 0
df["top1"] = df["w"].apply(lambda d: max(d, key=d.get) if d else None)
df["top1_w"] = df["w"].apply(lambda d: max(d.values()) if d else None)

print("\n== label availability ==")
print("all cases with label       :", df.has_label.sum(), "/", len(df))
print("head_neck with label       :", (df.is_head_neck & df.has_label).sum())
print("head_neck_only with label  :", (df.head_neck_only & df.has_label).sum())

hn = df[df.is_head_neck & df.has_label]
print("\n== top-1 condition among head_or_neck labelled cases ==")
print(hn["top1"].value_counts().head(25).to_string())

weights = Counter()
for d in hn["w"]:
    weights.update(d)
print("\n== summed soft-label weight (head_or_neck) top 15 ==")
for k, v in weights.most_common(15):
    print(f"{k:45s} {v:.1f}")

print("\n== top-1 confidence (weight) among head_or_neck labelled ==")
print(hn["top1_w"].describe().round(3).to_string())

print("\n== user-reported related_category (head_or_neck) ==")
print(df.loc[df.is_head_neck, "related_category"].value_counts(dropna=False).to_string())

# ---- skin tone availability ----
efst_cols = [c for c in df.columns if c.startswith("dermatologist_fitzpatrick_skin_type_label_")]
df["has_efst"] = df[efst_cols].notna().any(axis=1)
df["has_monk_us"] = df["monk_skin_tone_label_us"].notna()
print("\n== skin tone availability among head_neck labelled ==")
print("eFST (any dermatologist):", (df.is_head_neck & df.has_label & df.has_efst).sum())
print("Monk (US grader)        :", (df.is_head_neck & df.has_label & df.has_monk_us).sum())

# ---- save merged manifest (no model/test info used) ----
out = df.drop(columns=["w"])
Path("manifests").mkdir(exist_ok=True)
out.to_csv("manifests/scin_merged_raw.csv", index=False)
print("\nSaved manifests/scin_merged_raw.csv")