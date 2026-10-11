import pandas as pd

folds = pd.read_csv("manifests/folds_v2.csv", dtype={"case_id": str, "group": str})
raw = pd.read_csv("manifests/scin_merged_raw.csv", dtype={"case_id": str})
cols = ["case_id", "dermatologist_fitzpatrick_skin_type_label_1", "monk_skin_tone_label_us"]
d = folds.merge(raw[cols], on="case_id", how="left", validate="one_to_one")

fst_map = {"FST1": "FST1-2", "FST2": "FST1-2", "FST3": "FST3", "FST4": "FST4",
           "FST5": "FST5-6", "FST6": "FST5-6"}
d["derm_fst_group"] = d["dermatologist_fitzpatrick_skin_type_label_1"].map(fst_map)


def monk(v):
    if pd.isna(v):
        return None
    v = int(v)
    return "Monk1-2" if v <= 2 else "Monk3" if v == 3 else "Monk4" if v == 4 else "Monk5-10"


d["monk_group"] = d["monk_skin_tone_label_us"].map(monk)
d.to_csv("manifests/v2_skintone_groups.csv", index=False)

for col in ("derm_fst_group", "monk_group"):
    tab = pd.crosstab(d[col], d.label)
    out = tab.copy()
    out["total"] = tab.sum(axis=1)
    out["classes>=5"] = (tab >= 5).sum(axis=1)
    out["passes_rule"] = (out["total"] >= 30) & (out["classes>=5"] >= 2)
    print(f"\n== {col}: cases per class (missing: {int(d[col].isna().sum())}) ==")
    print(out.to_string())