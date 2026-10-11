import ast

import pandas as pd

df = pd.read_csv("manifests/scin_merged_raw.csv", dtype={"case_id": str})
lab = df[df.has_label].copy()
print("labelled cases (all body parts):", len(lab))


def has_tie(s):
    d = ast.literal_eval(s)
    m = max(d.values())
    return sum(abs(v - m) < 1e-9 for v in d.values()) > 1


lab["tie"] = lab.weighted_skin_condition_label.apply(has_tie)
print("top-1 ties (excluded):", int(lab.tie.sum()))
ok = lab[~lab.tie]
print("cases after tie exclusion:", len(ok), "| images:", int(ok.n_images.sum()))

tab = ok.groupby("top1").agg(cases=("case_id", "nunique"), images=("n_images", "sum"),
                             head_neck=("is_head_neck", "sum")).sort_values("cases", ascending=False)
print("\ntop 25 conditions (case-level top-1):")
print(tab.head(25).to_string())

print("\nthreshold scenarios (cases per class, after tie exclusion):")
for t in (20, 30, 50, 75):
    k = tab[tab.cases >= t]
    print(f"  >= {t:>2} cases: {len(k):>2} classes, {int(k.cases.sum()):>4} cases, {int(k.images.sum()):>4} images")

# download size estimate: 321.1 MB for 269 images measured earlier
mb = 321.1 / 269
for t in (30, 50):
    k = tab[tab.cases >= t]
    print(f"  est. download for classes with >= {t} cases: ~{k.images.sum() * mb / 1000:.1f} GB (from 321.1 MB / 269 images)")