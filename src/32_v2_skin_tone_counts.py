import pandas as pd

folds = pd.read_csv("manifests/folds_v2.csv", dtype={"case_id": str, "group": str})
raw = pd.read_csv("manifests/scin_merged_raw.csv", dtype={"case_id": str})
fst = [c for c in raw.columns if c.startswith("dermatologist_fitzpatrick_skin_type_label_")]
cols = ["case_id", "fitzpatrick_skin_type", "monk_skin_tone_label_us", "monk_skin_tone_label_india"] + fst
d = folds.merge(raw[[c for c in cols if c in raw.columns]], on="case_id", how="left", validate="one_to_one")
print("cases:", len(d))
print("\nself-reported Fitzpatrick (fitzpatrick_skin_type):")
print(d["fitzpatrick_skin_type"].value_counts(dropna=False).to_string())
print("\ndermatologist-estimated Fitzpatrick, label 1:")
print(d[fst[0]].value_counts(dropna=False).to_string())
print("\nMonk (US grader):")
print(d["monk_skin_tone_label_us"].value_counts(dropna=False).sort_index().to_string())
print("\ncases with any dermatologist Fitzpatrick estimate:", int(d[fst].notna().any(axis=1).sum()))
print("cases with a Monk (US) label:", int(d["monk_skin_tone_label_us"].notna().sum()))