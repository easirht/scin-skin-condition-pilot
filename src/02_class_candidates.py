import pandas as pd

df = pd.read_csv("manifests/scin_merged_raw.csv")
hn = df[df.is_head_neck & df.has_label].copy()
print("head/neck + labelled cases:", len(hn))

tab = hn.groupby("top1").agg(
    cases=("case_id", "nunique"),
    images=("n_images", "sum"),
    head_neck_only=("head_neck_only", "sum"),
    mean_top1_w=("top1_w", "mean"),
    cases_w_ge_050=("top1_w", lambda s: int((s >= 0.5).sum())),
).sort_values("cases", ascending=False)

tab.round(2).to_csv("outputs/reports/top1_classes_headneck.csv")
print(tab.round(2).to_string())

print("\n== threshold scenarios (15% test share) ==")
for t in (10, 15, 20, 30):
    k = tab[tab.cases >= t]
    print(f"min {t:>2} cases/class -> {len(k):>2} classes, {int(k.cases.sum()):>3} cases, "
          f"smallest class ~{k.cases.min()*0.15:.1f} test cases")